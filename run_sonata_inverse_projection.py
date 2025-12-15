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
# [关键] 使用 Nested Giant 模型获取 Metric Depth
DA3_MODEL_NAME = "/backup/group_朱聪聪/hqhong/models/DA3NESTED-GIANT-LARGE" 
SONATA_CKPT_PATH = "/backup/group_朱聪聪/hqhong/models/sonata/sonata.pth"

# 数据集路径 (bmx-trees)
# INPUT_VIDEO_DIR = "/backup/group_朱聪聪/hqhong/datasets/DAVIS/JPEGImages/1080p/bmx-trees"
# OUTPUT_DIR = "./demo/bmx-trees/sonata_inverse_projection_result"

INPUT_VIDEO_DIR = "/backup/group_朱聪聪/hqhong/datasets/DAVIS/JPEGImages/1080p/bear"
OUTPUT_DIR = "./demo/bear/sonata_inverse_projection_result"

# [算法参数 - 宽松版 V6]
VOXEL_SIZE = 0.02        
MATCH_STRIDE = 5         

# 1. 降低相似度门槛 (树叶在不同视角特征变化大)
SIM_THRESH = 0.60        

# 2. 距离容差: 基础 0.15m + 距离的 8% 
# (例如: 10m处容忍 0.95m, 50m处容忍 4.15m)
DIST_THRESH_BASE = 0.15 
DIST_THRESH_RATIO = 0.08

# 3. [新增] 强制静态距离 (米)
# 超过这个距离的点，强制认为是背景 (解决天空/远景乱飘问题)
FORCE_STATIC_DIST = 30.0 

# 4. [新增] 置信度阈值 (过滤天空/边缘噪声)
CONF_THRESH = 0.7

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

# ================= 2. 核心算法 =================

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
    mask_flip = normals[..., 2] > 0
    normals[mask_flip] *= -1
    return torch.from_numpy(normals.astype(np.float32)).to(depth.device)

def compute_feature_consistency(tokens_src, tokens_tgt, coords_src, coords_tgt):
    """
    带远景抑制的判别逻辑
    """
    device = tokens_src.device
    tokens_tgt = tokens_tgt.to(device)
    coords_src = coords_src.to(device)
    coords_tgt = coords_tgt.to(device)

    # 1. 相似度
    feats_src = F.normalize(tokens_src, p=2, dim=-1)
    feats_tgt = F.normalize(tokens_tgt, p=2, dim=-1)
    
    # 避免 OOM (假设 Token 数不多，直接算)
    sim_matrix = torch.mm(feats_src, feats_tgt.transpose(0, 1)) 
    best_sim_val, best_sim_idx = torch.max(sim_matrix, dim=1)
    
    # 2. 距离检查
    matched_coords = coords_tgt[best_sim_idx]
    dist = torch.norm(coords_src - matched_coords, dim=-1) 
    
    # 3. 自适应阈值
    src_depths = torch.norm(coords_src, dim=-1) # 粗略距离
    adaptive_thresh = DIST_THRESH_BASE + src_depths * DIST_THRESH_RATIO
    
    # 4. [强力补丁] 远景强制静态
    # 如果距离超过30米，直接认为是背景（不用管相似度匹配了没）
    is_far_background = src_depths > FORCE_STATIC_DIST
    
    has_match = best_sim_val > SIM_THRESH
    
    # 静态判定: (匹配且误差小) OR (是远景)
    is_static = (has_match & (dist < adaptive_thresh)) | is_far_background
    
    # 动态判定: (匹配且误差大) AND (不是远景)
    # 注意：如果不匹配(has_match=False)，我们视为噪声或遮挡，不归入动态
    is_dynamic = has_match & (dist >= adaptive_thresh) & (~is_far_background)
    
    return is_static, is_dynamic

# ================= 3. 主流程 =================

def run_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Step 1: DA3 ---
    print(f"\n[Step 1] Loading Depth Anything 3 (Nested)...")
    da3_model = DepthAnything3.from_pretrained(DA3_MODEL_NAME, dynamic=True).to(device)
    da3_model.eval()

    frames = sorted(glob.glob(os.path.join(INPUT_VIDEO_DIR, "*.jpg")) + 
                    glob.glob(os.path.join(INPUT_VIDEO_DIR, "*.png")))
    frames = frames[::1] 
    num_frames = len(frames)
    
    with torch.no_grad():
        prediction = da3_model.inference(
            image=frames,
            align_to_input_ext_scale=True,
            infer_gs=True, 
            process_res=504,
            export_format="mini_npz"
        )
    
    depths_np = prediction.depth 
    # [修复点] 使用 .conf 而不是 .confidence
    conf_np = prediction.conf  
    
    if conf_np is None:
        print("Warning: No confidence map found, using default 1.0")
        conf_np = np.ones_like(depths_np)

    extrinsics_np = prediction.extrinsics
    if extrinsics_np.shape[-2:] == (3, 4):
        bottom = np.array([[[0,0,0,1]]] * num_frames)
        extrinsics_np = np.concatenate([extrinsics_np, bottom], axis=1)
    
    w2c = torch.from_numpy(extrinsics_np).to(device).float()
    c2w = torch.linalg.inv(w2c)
    
    depths = torch.from_numpy(depths_np).to(device)
    confs = torch.from_numpy(conf_np).to(device)
    intrinsics = torch.from_numpy(prediction.intrinsics).to(device)
    images = prediction.processed_images

    # --- Step 2: Sonata ---
    print(f"\n[Step 2] Extracting Features (Conf > {CONF_THRESH})...")
    sonata_model = sonata.load(SONATA_CKPT_PATH).to(device)
    sonata_model.eval()

    try:
        stem = sonata_model.embedding.stem
        in_ch = stem.linear.in_features if hasattr(stem, 'linear') else stem[0].in_features
    except:
        in_ch = 6
    feat_keys = ("coord", "color", "normal") if in_ch == 9 else ("coord", "color")
    if in_ch == 9: print(">> Using Normals.")

    transform = Compose([
        # 备份 origin 数据
        dict(type="Copy", keys_dict={"coord": "origin_coord", "color": "origin_color", "normal": "origin_normal"}), 
        
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
            keys=("coord", "grid_coord", "color", "inverse", "origin_coord", "origin_color", "origin_normal"), 
            feat_keys=feat_keys,
        ),
    ])

    frame_data_pool = [] 

    for i in tqdm(range(num_frames)):
        d_i = depths[i]
        c_i = confs[i] # Confidence Map
        K_i = intrinsics[i]
        c2w_i = c2w[i]
        n_map_c = compute_normals_camera_space(d_i, K_i)
        
        # [核心优化] 过滤掉置信度低的点 (天空、边缘)
        # 且限制最大点数
        mask = (d_i > 0) & (c_i > CONF_THRESH)
        
        indices = torch.nonzero(mask).squeeze()
        if len(indices) > 100000:
            perm = torch.randperm(len(indices))[:100000]
            indices = indices[perm]
        
        y, x = indices[:, 0], indices[:, 1]
        z = d_i[y, x]
        
        fx, fy = K_i[0,0], K_i[1,1]
        cx, cy = K_i[0,2], K_i[1,2]
        
        X_c = (x - cx) * z / fx
        Y_c = (y - cy) * z / fy
        pts_c = torch.stack([X_c, Y_c, z], dim=-1)
        pts_w = (c2w_i[:3, :3] @ pts_c.T).T + c2w_i[:3, 3] 
        
        norm_c = n_map_c[y, x]
        norm_w = (c2w_i[:3, :3] @ norm_c.T).T
        col = torch.from_numpy(images[i]).to(device)[y, x].float()

        input_dict = {
            "coord": pts_w.cpu().numpy().astype(np.float32), 
            "color": col.cpu().numpy().astype(np.float32),
            "normal": norm_w.cpu().numpy().astype(np.float32)
        }
        
        input_dict = transform(input_dict)
        
        with torch.no_grad():
            for k in input_dict.keys():
                if isinstance(input_dict[k], torch.Tensor):
                    input_dict[k] = input_dict[k].to(device)
            input_dict["offset"] = torch.tensor([input_dict["coord"].shape[0]], device=device)
            
            output = sonata_model(input_dict)
            
            if hasattr(output, "coord"):
                token_coord = output.coord
            else:
                token_coord = input_dict["coord"]

            frame_data_pool.append({
                "token_feat": output.feat,          
                "token_coord": token_coord,         
                "dense_coord": input_dict["origin_coord"], 
                "dense_color": input_dict["origin_color"], 
                "inverse_idx": input_dict["inverse"]       
            })

    # --- Step 3: 匹配 & 广播 ---
    print(f"\n[Step 3] Matching & Broadcasting (Far-Field Static)...")
    
    static_dense_cloud = []
    dynamic_dense_cloud = []
    
    for i in tqdm(range(len(frame_data_pool))):
        curr = frame_data_pool[i]
        
        tgt_idx = min(i + MATCH_STRIDE, len(frame_data_pool) - 1)
        if tgt_idx == i: tgt_idx = max(0, i - MATCH_STRIDE)
        tgt = frame_data_pool[tgt_idx]
        
        is_static_token, is_dynamic_token = compute_feature_consistency(
            curr["token_feat"], tgt["token_feat"],
            curr["token_coord"], tgt["token_coord"]
        )
        
        num_tokens = curr["token_feat"].shape[0]
        valid_indices_mask = curr["inverse_idx"] < num_tokens
        valid_inverse = curr["inverse_idx"][valid_indices_mask]
        
        # 广播结果到原始点
        is_static_dense = is_static_token[valid_inverse]
        is_dynamic_dense = is_dynamic_token[valid_inverse]
        
        dense_coords = curr["dense_coord"][valid_indices_mask]
        dense_colors = curr["dense_color"][valid_indices_mask]
        
        if is_static_dense.sum() > 0:
            static_dense_cloud.append({
                "coord": dense_coords[is_static_dense].cpu().numpy(),
                "color": dense_colors[is_static_dense].cpu().numpy()
            })
            
        if is_dynamic_dense.sum() > 0:
            dynamic_dense_cloud.append({
                "coord": dense_coords[is_dynamic_dense].cpu().numpy(),
                "color": dense_colors[is_dynamic_dense].cpu().numpy()
            })

    # --- Step 4: 保存 ---
    print(f"\n[Step 4] Saving...")
    
    def save_cloud(data_list, filename):
        if not data_list: 
            print(f"Skipping {filename} (empty)")
            return
        pts = np.concatenate([p["coord"] for p in data_list])
        cols = np.concatenate([p["color"] for p in data_list])
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols / 255.0)
        
        # 随机降采样
        if len(pcd.points) > 1000000:
             pcd = pcd.random_down_sample(fraction=0.5)

        path = os.path.join(OUTPUT_DIR, filename)
        o3d.io.write_point_cloud(path, pcd)
        print(f"Saved {filename}: {len(pcd.points)} points")

    save_cloud(static_dense_cloud, "final_static_dense.ply")
    save_cloud(dynamic_dense_cloud, "final_dynamic_dense.ply")
    
    n_s = sum([len(p["coord"]) for p in static_dense_cloud])
    n_d = sum([len(p["coord"]) for p in dynamic_dense_cloud])
    print(f"\nStats:\n Static Points: {n_s}\n Dynamic Points: {n_d}")
    print("\nDone.")

if __name__ == "__main__":
    run_pipeline()