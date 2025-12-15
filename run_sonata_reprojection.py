import sys
import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import open3d as o3d
from tqdm import tqdm

# ================= 0. 全局配置 =================

SONATA_LIB_PATH = "/backup/group_朱聪聪/hqhong/projects/AnyDynamics/submodules/sonata"
DA3_MODEL_NAME = "/backup/group_朱聪聪/hqhong/models/DA3-GIANT" 
SONATA_CKPT_PATH = "/backup/group_朱聪聪/hqhong/models/sonata/sonata.pth"
INPUT_VIDEO_DIR = "./demo/bear"
OUTPUT_DIR = "./demo/bear_test/reprojection_result"

# [关键参数]
CONSISTENCY_THRESH = 0.05  # 相对深度误差阈值 5%
CONF_THRESH = 0.8          # DA3 置信度阈值
CHECK_STRIDE = 3           # 前后第几帧用于校验

# ================= 1. 环境初始化 =================

if SONATA_LIB_PATH not in sys.path:
    sys.path.append(SONATA_LIB_PATH)

try:
    import sonata
    from sonata.transform import Compose, Collect
    from src.depth_anything_3.api import DepthAnything3
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# ================= 2. 核心算法 =================

def get_reprojection_mask(depth_src, depth_tgt, K_src, K_tgt, c2w_src, c2w_tgt):
    """
    计算源帧到目标帧的重投影误差掩码
    """
    H, W = depth_src.shape
    device = depth_src.device
    
    # 1. 构建源帧 3D 点
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    fx, fy = K_src[0, 0], K_src[1, 1]
    cx, cy = K_src[0, 2], K_src[1, 2]
    
    X = (x - cx) * depth_src / fx
    Y = (y - cy) * depth_src / fy
    Z = depth_src
    
    # Flatten
    points_src = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
    
    # 2. 变换到目标帧
    w2c_tgt = torch.linalg.inv(c2w_tgt)
    R = w2c_tgt[:3, :3] @ c2w_src[:3, :3]
    t = w2c_tgt[:3, :3] @ c2w_src[:3, 3] + w2c_tgt[:3, 3]
    points_tgt = (R @ points_src.T).T + t
    
    # 3. 投影
    fx_t, fy_t = K_tgt[0, 0], K_tgt[1, 1]
    cx_t, cy_t = K_tgt[0, 2], K_tgt[1, 2]
    
    z_tgt_proj = points_tgt[:, 2]
    x_img = (points_tgt[:, 0] * fx_t / (z_tgt_proj + 1e-6)) + cx_t
    y_img = (points_tgt[:, 1] * fy_t / (z_tgt_proj + 1e-6)) + cy_t
    
    # 4. 采样
    x_norm = 2 * (x_img / (W - 1)) - 1
    y_norm = 2 * (y_img / (H - 1)) - 1
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0).unsqueeze(0)
    
    depth_tgt_sampled = F.grid_sample(
        depth_tgt.unsqueeze(0).unsqueeze(0), 
        grid, 
        mode='nearest', 
        padding_mode='zeros', 
        align_corners=True
    ).reshape(-1)
    
    # 5. 误差计算
    valid_proj = (z_tgt_proj > 0) & (x_norm > -1) & (x_norm < 1) & (y_norm > -1) & (y_norm < 1)
    valid_depth = (depth_tgt_sampled > 0)
    
    diff = torch.abs(z_tgt_proj - depth_tgt_sampled)
    rel_diff = diff / (depth_tgt_sampled + 1e-6)
    
    is_static = (rel_diff < CONSISTENCY_THRESH) | (diff < 0.1)
    
    # Reshape back to H, W
    return (valid_proj & valid_depth).reshape(H, W), is_static.reshape(H, W)

def compute_normals_camera_space(depth, intrinsics):
    H, W = depth.shape
    depth_np = depth.cpu().numpy()
    fx, fy = intrinsics[0, 0].item(), intrinsics[1, 1].item()
    cx, cy = intrinsics[0, 2].item(), intrinsics[1, 2].item()
    
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    valid_mask = (depth_np > 0) & np.isfinite(depth_np)
    X, Y = np.zeros_like(depth_np), np.zeros_like(depth_np)
    X[valid_mask] = (x[valid_mask] - cx) * depth_np[valid_mask] / fx
    Y[valid_mask] = (y[valid_mask] - cy) * depth_np[valid_mask] / fy
    Z = depth_np.copy()

    ksize = 5
    dX_du = cv2.Sobel(X, cv2.CV_64F, 1, 0, ksize=ksize)
    dY_du = cv2.Sobel(Y, cv2.CV_64F, 1, 0, ksize=ksize)
    dZ_du = cv2.Sobel(Z, cv2.CV_64F, 1, 0, ksize=ksize)
    dX_dv = cv2.Sobel(X, cv2.CV_64F, 0, 1, ksize=ksize)
    dY_dv = cv2.Sobel(Y, cv2.CV_64F, 0, 1, ksize=ksize)
    dZ_dv = cv2.Sobel(Z, cv2.CV_64F, 0, 1, ksize=ksize)

    tu = np.stack([dX_du, dY_du, dZ_du], axis=-1)
    tv = np.stack([dX_dv, dY_dv, dZ_dv], axis=-1)
    normals = np.cross(tu, tv)
    norm_mag = np.linalg.norm(normals, axis=-1, keepdims=True)
    norm_mag[norm_mag < 1e-6] = 1e-6 
    normals = normals / norm_mag
    mask_flip = normals[..., 2] > 0
    normals[mask_flip] *= -1
    return torch.from_numpy(normals.astype(np.float32)).to(depth.device)

# ================= 3. 主流程 =================

def run_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Step 1: DA3 Inference ---
    print(f"\n[Step 1] Loading Depth Anything 3...")
    da3_model = DepthAnything3.from_pretrained(DA3_MODEL_NAME, dynamic=True).to(device)
    da3_model.eval()

    frames = sorted(glob.glob(os.path.join(INPUT_VIDEO_DIR, "*.jpg")) + 
                    glob.glob(os.path.join(INPUT_VIDEO_DIR, "*.png")))
    frames = frames[::1]
    num_frames = len(frames)
    
    print(f"Processing {num_frames} frames...")

    with torch.no_grad():
        prediction = da3_model.inference(
            image=frames,
            align_to_input_ext_scale=True,
            infer_gs=True, 
            process_res=504,
            export_dir=None, 
            export_format="mini_npz"
        )

    depths = torch.from_numpy(prediction.depth).to(device)
    confs = torch.from_numpy(prediction.conf).to(device)
    intrinsics = torch.from_numpy(prediction.intrinsics).to(device)
    
    extrinsics_np = prediction.extrinsics
    if extrinsics_np.shape[-2:] == (3, 4):
        bottom = np.array([[[0,0,0,1]]] * num_frames)
        extrinsics_np = np.concatenate([extrinsics_np, bottom], axis=1)
    w2c = torch.from_numpy(extrinsics_np).to(device).float()
    c2w = torch.linalg.inv(w2c)
    
    images = prediction.processed_images

    # --- Step 2: Fusing ---
    print(f"\n[Step 2] Fusing with Reprojection Consistency Check...")
    
    static_points_list = []
    static_colors_list = []
    static_normals_list = []
    
    dynamic_points_list = []
    dynamic_colors_list = []
    dynamic_normals_list = []

    for i in tqdm(range(num_frames)):
        depth_i = depths[i]
        conf_i = confs[i]
        K_i = intrinsics[i]
        c2w_i = c2w[i]
        
        base_mask = (depth_i > 0) & (conf_i > CONF_THRESH)
        is_static_pixel = torch.zeros_like(depth_i, dtype=torch.bool)
        
        check_indices = [idx for idx in [i - CHECK_STRIDE, i + CHECK_STRIDE] if 0 <= idx < num_frames]
        
        if not check_indices:
            is_static_pixel = torch.ones_like(depth_i, dtype=torch.bool)
        else:
            for j in check_indices:
                valid_proj, consistent = get_reprojection_mask(
                    depth_i, depths[j], K_i, intrinsics[j], c2w_i, c2w[j]
                )
                is_static_pixel = is_static_pixel | (valid_proj & consistent)
        
        final_static_mask = base_mask & is_static_pixel
        final_dynamic_mask = base_mask & (~is_static_pixel)
        
        normal_map = compute_normals_camera_space(depth_i, K_i)
        
        H, W = depth_i.shape
        y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        
        # Helper function to extract and transform points
        def extract_and_transform(mask, output_pts, output_col, output_norm):
            if mask.sum() > 0:
                z = depth_i[mask]
                x_ = x[mask]
                y_ = y[mask]
                
                X_c = (x_ - K_i[0, 2]) * z / K_i[0, 0]
                Y_c = (y_ - K_i[1, 2]) * z / K_i[1, 1]
                pts_c = torch.stack([X_c, Y_c, z], dim=-1)
                
                pts_w = (c2w_i[:3, :3] @ pts_c.T).T + c2w_i[:3, 3]
                norm_w = (c2w_i[:3, :3] @ normal_map[mask].T).T
                col = torch.from_numpy(images[i]).to(device)[mask]
                
                output_pts.append(pts_w.cpu().numpy())
                output_col.append(col.cpu().numpy())
                output_norm.append(norm_w.cpu().numpy())

        extract_and_transform(final_static_mask, static_points_list, static_colors_list, static_normals_list)
        extract_and_transform(final_dynamic_mask, dynamic_points_list, dynamic_colors_list, dynamic_normals_list)

    # --- Step 3: Statistics & Saving ---
    print(f"\n[Step 3] Saving Results...")

    # A. 聚合数据
    print("Aggregating points...")
    if static_points_list:
        static_pts = np.concatenate(static_points_list)
        static_col = np.concatenate(static_colors_list)
        static_norm = np.concatenate(static_normals_list)
    else:
        static_pts = np.empty((0, 3))
        static_col = np.empty((0, 3))
        static_norm = np.empty((0, 3))

    if dynamic_points_list:
        dynamic_pts = np.concatenate(dynamic_points_list)
        dynamic_col = np.concatenate(dynamic_colors_list)
        dynamic_norm = np.concatenate(dynamic_normals_list)
    else:
        dynamic_pts = np.empty((0, 3))
        dynamic_col = np.empty((0, 3))
        dynamic_norm = np.empty((0, 3))

    # B. [需求 1] 打印数量统计
    num_static = len(static_pts)
    num_dynamic = len(dynamic_pts)
    num_total = num_static + num_dynamic
    
    print("\n" + "="*30)
    print(f" STATISTICS")
    print("="*30)
    print(f" Static Points  : {num_static:10d} ({num_static/num_total*100:.1f}%)")
    print(f" Dynamic Points : {num_dynamic:10d} ({num_dynamic/num_total*100:.1f}%)")
    print(f" Total Points   : {num_total:10d}")
    print("="*30 + "\n")

    # C. [需求 2] 保存完整合并点云 (未降采样)
    if num_total > 0:
        print(f"Saving combined scene ({num_total} points)...")
        full_pts = np.vstack([p for p in [static_pts, dynamic_pts] if len(p) > 0])
        full_col = np.vstack([c for c in [static_col, dynamic_col] if len(c) > 0])
        full_norm = np.vstack([n for n in [static_norm, dynamic_norm] if len(n) > 0])
        
        pcd_full = o3d.geometry.PointCloud()
        pcd_full.points = o3d.utility.Vector3dVector(full_pts)
        pcd_full.colors = o3d.utility.Vector3dVector(full_col / 255.0)
        pcd_full.normals = o3d.utility.Vector3dVector(full_norm)
        
        o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, "scene_combined.ply"), pcd_full)
        print(f"-> Saved: {os.path.join(OUTPUT_DIR, 'scene_combined.ply')}")

    # D. 保存静态点云 (推荐降采样)
    if num_static > 0:
        pcd_static = o3d.geometry.PointCloud()
        pcd_static.points = o3d.utility.Vector3dVector(static_pts)
        pcd_static.colors = o3d.utility.Vector3dVector(static_col / 255.0)
        pcd_static.normals = o3d.utility.Vector3dVector(static_norm)
        
        print(f"Downsampling static background (voxel=0.02)...")
        pcd_static = pcd_static.voxel_down_sample(voxel_size=0.02)
        o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, "static_env.ply"), pcd_static)
        print(f"-> Saved: {os.path.join(OUTPUT_DIR, 'static_env.ply')} ({len(pcd_static.points)} points)")

    # E. 保存动态点云 (完整保留)
    if num_dynamic > 0:
        pcd_dyn = o3d.geometry.PointCloud()
        pcd_dyn.points = o3d.utility.Vector3dVector(dynamic_pts)
        pcd_dyn.colors = o3d.utility.Vector3dVector(dynamic_col / 255.0)
        pcd_dyn.normals = o3d.utility.Vector3dVector(dynamic_norm)
        
        o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, "dynamic_trace.ply"), pcd_dyn)
        print(f"-> Saved: {os.path.join(OUTPUT_DIR, 'dynamic_trace.ply')}")

    print("\nDone.")

if __name__ == "__main__":
    run_pipeline()