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
INPUT_VIDEO_DIR = "./demo/bear"
OUTPUT_DIR = "./demo/bear_test/feature_matching_result"

# [算法参数]
VOXEL_SIZE = 0.02        
MATCH_STRIDE = 5         
SIM_THRESH = 0.75        
DIST_THRESH_STATIC = 0.10 

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
    device = tokens_src.device
    tokens_tgt = tokens_tgt.to(device)
    coords_src = coords_src.to(device)
    coords_tgt = coords_tgt.to(device)

    # 1. 归一化
    feats_src = F.normalize(tokens_src, p=2, dim=-1)
    feats_tgt = F.normalize(tokens_tgt, p=2, dim=-1)
    
    # 2. 相似度
    try:
        sim_matrix = torch.mm(feats_src, feats_tgt.transpose(0, 1)) 
    except RuntimeError:
        feats_src = feats_src.cpu()
        feats_tgt = feats_tgt.cpu()
        sim_matrix = torch.mm(feats_src, feats_tgt.transpose(0, 1))
        sim_matrix = sim_matrix.to(device)

    # 3. 最佳匹配
    best_sim_val, best_sim_idx = torch.max(sim_matrix, dim=1)
    
    # 4. 坐标差分
    matched_coords = coords_tgt[best_sim_idx]
    dist = torch.norm(coords_src - matched_coords, dim=-1)
    
    has_match = best_sim_val > SIM_THRESH
    is_static = has_match & (dist < DIST_THRESH_STATIC)
    is_dynamic = has_match & (dist >= DIST_THRESH_STATIC)
    
    return is_static, is_dynamic, dist, best_sim_val

def get_nearest_colors(target_coords, source_coords, source_colors):
    """
    为 target_coords 找到 source_coords 中最近邻的颜色
    """
    # 简单暴力法 (GPU上通常够快)
    # 分块处理以防 OOM
    chunk_size = 1000
    N = target_coords.shape[0]
    result_colors = torch.zeros((N, 3), device=target_coords.device, dtype=source_colors.dtype)
    
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        tgt_chunk = target_coords[i:end].unsqueeze(1) # (B, 1, 3)
        
        # 计算距离
        dists = torch.norm(tgt_chunk - source_coords.unsqueeze(0), dim=-1) # (B, M)
        min_dist, min_idx = torch.min(dists, dim=1)
        
        result_colors[i:end] = source_colors[min_idx]
        
    return result_colors

# ================= 3. 主流程 =================

def run_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Step 1: DA3 (Nested) ---
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
    
    extrinsics_np = prediction.extrinsics
    if extrinsics_np.shape[-2:] == (3, 4):
        bottom = np.array([[[0,0,0,1]]] * num_frames)
        extrinsics_np = np.concatenate([extrinsics_np, bottom], axis=1)
    
    w2c = torch.from_numpy(extrinsics_np).to(device).float()
    c2w = torch.linalg.inv(w2c)
    
    depths = torch.from_numpy(depths_np).to(device)
    intrinsics = torch.from_numpy(prediction.intrinsics).to(device)
    images = prediction.processed_images

    # --- Step 2: Sonata ---
    print(f"\n[Step 2] Extracting Sonata Features...")
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
        # 只需要 Copy 用于最后的最近邻查询
        dict(type="Copy", keys_dict={"coord": "origin_coord", "color": "origin_color"}), 
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
            keys=("coord", "grid_coord", "color", "inverse", "origin_coord", "origin_color"), 
            feat_keys=feat_keys,
        ),
    ])

    frame_data_pool = [] 

    for i in tqdm(range(num_frames)):
        d_i = depths[i]
        K_i = intrinsics[i]
        c2w_i = c2w[i]
        n_map_c = compute_normals_camera_space(d_i, K_i)
        
        # 降采样
        mask = (d_i > 0)
        indices = torch.nonzero(mask).squeeze()
        if len(indices) > 50000:
            perm = torch.randperm(len(indices))[:50000]
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
            
            # --- [核心修改] 健壮的数据获取逻辑 ---
            
            # 1. 获取 Output Token Coords
            # Sonata (PointTransformer) 输出对象通常有 .coord 属性 (Strided Coords)
            if hasattr(output, "coord"):
                token_coord = output.coord
            else:
                # 如果没有，说明没有下采样，可以直接用 input grid coord
                # 但要小心 size mismatch
                if output.feat.shape[0] == input_dict["coord"].shape[0]:
                    token_coord = input_dict["coord"]
                else:
                    print(f"Warning: Frame {i} output feature size {output.feat.shape} != input coord {input_dict['coord'].shape}. And no output.coord found.")
                    # Fallback: 假设前 N 个对应? (危险)
                    # 或者跳过此帧
                    continue

            # 2. 获取 Output Token Features
            token_feat = output.feat
            
            # 3. 获取 Color (通过 Nearest Neighbor 匹配 Input Dense Points)
            # input_dict["origin_coord"] 和 ["origin_color"] 是原始密集点
            # 我们找到每个 token_coord 对应的最近原始点颜色
            token_color = get_nearest_colors(
                token_coord, 
                input_dict["origin_coord"], 
                input_dict["origin_color"]
            )

            frame_data_pool.append({
                "feat": token_feat,   
                "coord": token_coord, 
                "color": token_color
            })

    # --- Step 3: 匹配 ---
    print(f"\n[Step 3] Matching features across time (Stride={MATCH_STRIDE})...")
    
    static_pool = []
    dynamic_pool = []
    
    for i in tqdm(range(len(frame_data_pool))):
        curr_frame = frame_data_pool[i]
        
        tgt_idx = min(i + MATCH_STRIDE, len(frame_data_pool) - 1)
        if tgt_idx == i: tgt_idx = max(0, i - MATCH_STRIDE)
        
        tgt_frame = frame_data_pool[tgt_idx]
        
        is_static, is_dynamic, dists, sims = compute_feature_consistency(
            curr_frame["feat"], tgt_frame["feat"],
            curr_frame["coord"], tgt_frame["coord"]
        )
        
        def extract(mask):
            return {
                "coord": curr_frame["coord"][mask].cpu().numpy(),
                "color": curr_frame["color"][mask].cpu().numpy()
            }

        if is_static.sum() > 0:
            static_pool.append(extract(is_static))
            
        if is_dynamic.sum() > 0:
            dynamic_pool.append(extract(is_dynamic))

    # --- Step 4: 保存 ---
    print(f"\n[Step 4] Saving...")
    
    def save_pool(pool, filename):
        if not pool: 
            print(f"Skipping {filename} (empty)")
            return
        pts = np.concatenate([p["coord"] for p in pool])
        cols = np.concatenate([p["color"] for p in pool])
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols / 255.0)
        
        o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, filename), pcd)
        print(f"Saved {filename}: {len(pcd.points)} points")

    save_pool(static_pool, "matched_static.ply")
    save_pool(dynamic_pool, "matched_dynamic_objects.ply")
    
    n_static = sum([len(p["coord"]) for p in static_pool])
    n_dynamic = sum([len(p["coord"]) for p in dynamic_pool])
    print(f"\nStats:\n Static Tokens: {n_static}\n Dynamic Tokens: {n_dynamic}")
    print("\nDone.")

if __name__ == "__main__":
    run_pipeline()