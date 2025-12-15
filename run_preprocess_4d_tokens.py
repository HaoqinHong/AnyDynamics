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
DA3_MODEL_NAME = "/backup/group_朱聪聪/hqhong/models/DA3NESTED-GIANT-LARGE" 
SONATA_CKPT_PATH = "/backup/group_朱聪聪/hqhong/models/sonata/sonata.pth"

INPUT_VIDEO_DIR = "/backup/group_朱聪聪/hqhong/datasets/DAVIS/JPEGImages/1080p/bmx-trees"
OUTPUT_ROOT = "./demo/bmx-trees/training_data" 

CONF_THRESH = 0.75       
MAX_DEPTH_DIST = 50.0    
VOXEL_SIZE = 0.02        
SAVE_VISUALIZATION = True 

# ================= 1. 环境初始化 =================

if SONATA_LIB_PATH not in sys.path:
    sys.path.append(SONATA_LIB_PATH)

try:
    import sonata
    from sonata.transform import Compose
    from src.depth_anything_3.api import DepthAnything3
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# ================= 2. 辅助函数 =================

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

def pool_dense_to_sparse(dense_data, inverse_indices, num_tokens):
    """
    聚合函数：将 Dense (N) 数据平均到 Sparse (M) Token 上
    """
    device = dense_data.device
    C = dense_data.shape[1]
    
    # 确保 dense_data 和 indices 在同一个设备
    if inverse_indices.device != device:
        inverse_indices = inverse_indices.to(device)
    
    pooled_sum = torch.zeros((num_tokens, C), device=device, dtype=dense_data.dtype)
    pooled_count = torch.zeros((num_tokens, 1), device=device, dtype=dense_data.dtype)
    ones_source = torch.ones((len(inverse_indices), 1), device=device, dtype=dense_data.dtype)
    
    # Index Add
    pooled_sum.index_add_(0, inverse_indices, dense_data)
    pooled_count.index_add_(0, inverse_indices, ones_source)
    
    pooled_data = pooled_sum / (pooled_count + 1e-6)
    return pooled_data

# ================= 3. 主流水线 =================

def run_pipeline():
    os.makedirs(os.path.join(OUTPUT_ROOT, "tokens"), exist_ok=True)
    if SAVE_VISUALIZATION:
        os.makedirs(os.path.join(OUTPUT_ROOT, "clean_ply"), exist_ok=True)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[Init] Loading Depth Anything 3 (Nested Giant)...")
    da3_model = DepthAnything3.from_pretrained(DA3_MODEL_NAME, dynamic=True).to(device)
    da3_model.eval()

    print(f"[Init] Loading Sonata Encoder...")
    sonata_model = sonata.load(SONATA_CKPT_PATH).to(device)
    sonata_model.eval()

    try:
        stem = sonata_model.embedding.stem
        in_ch = stem.linear.in_features if hasattr(stem, 'linear') else stem[0].in_features
    except:
        in_ch = 6
    feat_keys = ("coord", "color", "normal") if in_ch == 9 else ("coord", "color")
    print(f">> Sonata Input Channels: {in_ch}")

    # Transform: 注意这里不再需要 Copy/Collect origin，因为我们手动备份了
    sonata_transform = Compose([
        dict(type="CenterShift", apply_z=True),
        dict(
            type="GridSample",
            grid_size=VOXEL_SIZE, 
            hash_type="fnv",
            mode="train", 
            return_grid_coord=True,
            return_inverse=True, 
        ),
        dict(type="NormalizeColor"), 
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "color", "inverse"), 
            feat_keys=feat_keys,
        ),
    ])

    frames = sorted(glob.glob(os.path.join(INPUT_VIDEO_DIR, "*.jpg")) + 
                    glob.glob(os.path.join(INPUT_VIDEO_DIR, "*.png")))
    frames = frames[::1] 
    num_frames = len(frames)
    print(f">> Processing {num_frames} frames...")

    with torch.no_grad():
        prediction = da3_model.inference(
            image=frames,
            align_to_input_ext_scale=True,
            infer_gs=True, 
            process_res=504,
            export_format="mini_npz"
        )
    
    depths = torch.from_numpy(prediction.depth).to(device)
    # [Fix] use .conf not .confidence
    confs = torch.from_numpy(prediction.conf).to(device) if prediction.conf is not None else None
    
    extrinsics_np = prediction.extrinsics
    if extrinsics_np.shape[-2:] == (3, 4):
        bottom = np.array([[[0,0,0,1]]] * num_frames)
        extrinsics_np = np.concatenate([extrinsics_np, bottom], axis=1)
    w2c = torch.from_numpy(extrinsics_np).to(device).float()
    c2w = torch.linalg.inv(w2c)
    
    intrinsics = torch.from_numpy(prediction.intrinsics).to(device)
    images = prediction.processed_images 

    print(f"\n[Processing] Running Cleaning & Tokenization...")
    
    for i in tqdm(range(num_frames)):
        d = depths[i]
        K = intrinsics[i]
        c2w_mat = c2w[i]
        img = torch.from_numpy(images[i]).to(device).float() 
        
        # 1. Cleaning
        valid_mask = (d > 0) & (d < MAX_DEPTH_DIST)
        if confs is not None:
            valid_mask = valid_mask & (confs[i] > CONF_THRESH)
            
        indices = torch.nonzero(valid_mask).squeeze()
        if len(indices) > 200000:
            perm = torch.randperm(len(indices))[:200000]
            indices = indices[perm]
            
        y, x = indices[:, 0], indices[:, 1]
        z = d[y, x]
        
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]
        
        X_c = (x - cx) * z / fx
        Y_c = (y - cy) * z / fy
        pts_c = torch.stack([X_c, Y_c, z], dim=-1)
        pts_w = (c2w_mat[:3, :3] @ pts_c.T).T + c2w_mat[:3, 3]
        
        colors = img[y, x] 
        
        normals = None
        if in_ch == 9:
            n_map = compute_normals_camera_space(d, K)
            n_c = n_map[y, x]
            normals = (c2w_mat[:3, :3] @ n_c.T).T

        # [关键修复] 在 Transform 之前，备份 Dense Colors
        # 注意：这里的 colors 是 0-255，我们先转成 0-1 方便后续计算
        dense_colors_backup = colors.clone() / 255.0

        # 2. Tokenization
        input_dict = {
            "coord": pts_w.cpu().numpy().astype(np.float32), 
            "color": colors.cpu().numpy().astype(np.float32),
        }
        if normals is not None:
            input_dict["normal"] = normals.cpu().numpy().astype(np.float32)
            
        # GridSample 会把 input_dict 里的 color 降维，所以我们上面备份了
        input_dict = sonata_transform(input_dict)
        
        for k in input_dict.keys():
            if isinstance(input_dict[k], torch.Tensor):
                input_dict[k] = input_dict[k].to(device)
        input_dict["offset"] = torch.tensor([input_dict["coord"].shape[0]], device=device)
        
        # Encoder Forward
        with torch.no_grad():
            output = sonata_model(input_dict)
        
        token_feat = output.feat            
        token_coord = input_dict["coord"]
        if hasattr(output, "coord"):
            token_coord = output.coord

        # === [核心修复] 使用备份的 Dense Colors 进行聚合 ===
        inverse_idx = input_dict["inverse"] # (N,)
        num_tokens = output.feat.shape[0]   # (M,)
        
        # 使用 pool_dense_to_sparse 聚合颜色
        token_colors = pool_dense_to_sparse(dense_colors_backup, inverse_idx, num_tokens)

        # 保存 .pt
        frame_data = {
            "feat": token_feat.cpu().half(),   
            "coord": token_coord.cpu().half(),
            "color": token_colors.cpu().half(), # 聚合后的平均色
            "frame_idx": i,
            "timestamp": i / num_frames
        }
        torch.save(frame_data, os.path.join(OUTPUT_ROOT, "tokens", f"frame_{i:04d}.pt"))
        
        # 保存 PLY (可视化)
        if SAVE_VISUALIZATION:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(token_coord.cpu().numpy())
            pcd.colors = o3d.utility.Vector3dVector(token_colors.cpu().numpy())
            
            o3d.io.write_point_cloud(os.path.join(OUTPUT_ROOT, "clean_ply", f"frame_{i:04d}.ply"), pcd)

    print(f"\n[Done] All frames processed.")
    print(f"   • Token Data: {os.path.join(OUTPUT_ROOT, 'tokens')}")
    print(f"   • Visual Check: {os.path.join(OUTPUT_ROOT, 'clean_ply')}")

if __name__ == "__main__":
    run_pipeline()