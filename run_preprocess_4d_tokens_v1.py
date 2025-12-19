import sys
import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import open3d as o3d
from tqdm import tqdm

# ================= 0. 全局配置 =================

SONATA_LIB_PATH = "/opt/data/private/Ours-Projects/Physics-Simulator-World-Model/AnyDynamics/submodules/sonata"
DA3_MODEL_NAME = "/opt/data/private/models/depthanything3/DA3-GIANT" 
SONATA_CKPT_PATH = "/opt/data/private/models/sonata/sonata.pth"

INPUT_VIDEO_DIR = "/opt/data/private/datasets/davis_2016/DAVIS_2016/JPEGImages/1080p/bmx-trees"
OUTPUT_ROOT = "./demo/bmx-trees/scene_token_data" 

# [核心参数]
VOXEL_SIZE = 0.02        # 体素大小
POINTS_PER_FRAME = 50000 # 显存控制
CONF_THRESH = 0.70       
MAX_DEPTH_DIST = 50.0    

# [Embedding 参数]
CAM_EMBED_DIM = 16       
TIME_EMBED_DIM = 8       

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

# ================= 2. 辅助组件 =================

def get_camera_embedding(c2w, dim=16):
    flat = c2w[:3, :4].flatten()
    if flat.shape[0] < dim:
        return F.pad(flat, (0, dim - flat.shape[0]), "constant", 0)
    else:
        return flat[:dim]

def get_time_embedding(t_norm, dim=8):
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim)).to(t_norm.device)
    pe = torch.zeros(dim).to(t_norm.device)
    pe[0::2] = torch.sin(t_norm * 100 * div_term)
    pe[1::2] = torch.cos(t_norm * 100 * div_term)
    return pe

# ================= 3. 主流水线 =================

def run_pipeline():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Models ---
    print(f"\n[Init] Loading Models...")
    da3_model = DepthAnything3.from_pretrained(DA3_MODEL_NAME, dynamic=True).to(device)
    da3_model.eval()
    
    sonata_model = sonata.load(SONATA_CKPT_PATH).to(device)
    sonata_model.eval()

    # --- Data Prep ---
    frames = sorted(glob.glob(os.path.join(INPUT_VIDEO_DIR, "*.jpg")) + 
                    glob.glob(os.path.join(INPUT_VIDEO_DIR, "*.png")))
    num_frames = len(frames)
    print(f">> Processing {num_frames} frames (All-in-One Mode)...")

    # --- Step 1: 批量 DA3 预测 & 数据收集 ---
    all_coords = []
    all_colors = []
    all_cam_embs = []  
    all_time_embs = [] 
    
    print(f">> Step 1: Generating Dense Point Clouds...")
    with torch.no_grad():
        prediction = da3_model.inference(
            image=frames,
            align_to_input_ext_scale=True,
            infer_gs=True, 
            process_res=504,
            export_format="mini_npz"
        )
    
    depths = torch.from_numpy(prediction.depth).to(device)
    confs = torch.from_numpy(prediction.conf).to(device) if prediction.conf is not None else None
    
    extrinsics_np = prediction.extrinsics
    if extrinsics_np.shape[-2:] == (3, 4):
        bottom = np.array([[[0,0,0,1]]] * num_frames)
        extrinsics_np = np.concatenate([extrinsics_np, bottom], axis=1)
    w2c = torch.from_numpy(extrinsics_np).to(device).float()
    c2w = torch.linalg.inv(w2c)
    
    intrinsics = torch.from_numpy(prediction.intrinsics).to(device)
    images = prediction.processed_images 

    for i in tqdm(range(num_frames)):
        d = depths[i]
        K = intrinsics[i]
        c2w_mat = c2w[i]
        img = torch.from_numpy(images[i]).to(device).float() / 255.0 
        
        mask = (d > 0) & (d < MAX_DEPTH_DIST)
        if confs is not None:
            mask = mask & (confs[i] > CONF_THRESH)
            
        indices = torch.nonzero(mask).squeeze()
        if len(indices) > POINTS_PER_FRAME:
            perm = torch.randperm(len(indices))[:POINTS_PER_FRAME]
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
        
        N_points = pts_w.shape[0]
        cam_vec = get_camera_embedding(c2w_mat, dim=CAM_EMBED_DIM) 
        cam_block = cam_vec.unsqueeze(0).expand(N_points, -1)      
        
        t_norm = torch.tensor(i / (num_frames - 1), device=device).float()
        time_vec = get_time_embedding(t_norm, dim=TIME_EMBED_DIM) 
        time_block = time_vec.unsqueeze(0).expand(N_points, -1)    
        
        all_coords.append(pts_w)
        all_colors.append(colors)
        all_cam_embs.append(cam_block)
        all_time_embs.append(time_block)

    print(f">> Merging {num_frames} frames into one Giant Point Cloud...")
    big_coords = torch.cat(all_coords, dim=0)    
    big_colors = torch.cat(all_colors, dim=0)    
    big_cam_embs = torch.cat(all_cam_embs, dim=0) 
    big_time_embs = torch.cat(all_time_embs, dim=0) 
    
    print(f"   Total Points: {big_coords.shape[0]}")

    # --- Step 2: Sonata Global Processing ---
    print(f">> Step 2: Running Sonata GridSample & Encoding (Global)...")
    
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
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "inverse"), 
            feat_keys=("coord", "color"), 
        ),
    ])
    
    # [Fix 1]: Numpy conversion to avoid CenterShift crash
    input_dict = {
        "coord": big_coords.cpu().numpy(), 
        "color": big_colors.cpu().numpy()  
    }
    
    input_dict = sonata_transform(input_dict)
    
    # [Fix 2]: Channel Padding (6 -> 9)
    feat = input_dict["feat"] 
    if feat.shape[1] == 6:
        print(f"   [Auto-Fix] Padding features from 6 to 9 channels (adding dummy normals)...")
        dummy_normal = torch.zeros((feat.shape[0], 3), dtype=feat.dtype)
        input_dict["feat"] = torch.cat([feat, dummy_normal], dim=1)

    for k in input_dict.keys():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = input_dict[k].to(device)
    input_dict["offset"] = torch.tensor([input_dict["coord"].shape[0]], device=device)
    
    # Forward
    with torch.no_grad():
        output = sonata_model(input_dict)
    
    token_geo_feat = output.feat  
    
    # [Fix 3 - CRITICAL]: 使用 input coord 大小作为 Token 数量基准
    # 避免 inverse_idx (基于 input coord) 超过 token_geo_feat (模型输出) 的范围
    # inverse_idx 的值域是 [0, num_voxels - 1]
    num_voxels = input_dict["coord"].shape[0]
    num_out_feat = token_geo_feat.shape[0]
    inverse_idx = input_dict["inverse"] 
    
    print(f"   Input Voxels: {num_voxels}, Model Output Feats: {num_out_feat}")
    
    # 强制聚合容器的大小等于 Voxel 数量，确保 inverse_idx 不越界
    pool_count = num_voxels

    # --- Step 3: Aggregate Embeddings (Pool) ---
    print(f">> Step 3: Pooling Embeddings to Tokens...")
    
    def pool_data(data, idx, count):
        C = data.shape[1]
        out = torch.zeros((count, C), device=device, dtype=data.dtype)
        ones = torch.ones((len(idx), 1), device=device, dtype=data.dtype)
        cnt = torch.zeros((count, 1), device=device, dtype=data.dtype)
        
        # [Safety Check] 确保 idx 不超过 count
        if idx.max() >= count:
            print(f"[Error] Index out of bounds: max_idx={idx.max()}, count={count}")
            # 紧急扩展 count 防止 crash (仅作最后一道防线)
            count = idx.max().item() + 1
            out = torch.zeros((count, C), device=device, dtype=data.dtype)
            cnt = torch.zeros((count, 1), device=device, dtype=data.dtype)

        out.index_add_(0, idx, data)
        cnt.index_add_(0, idx, ones)
        return out / (cnt + 1e-6)

    token_color = pool_data(big_colors, inverse_idx, pool_count)
    token_cam = pool_data(big_cam_embs, inverse_idx, pool_count)
    token_time = pool_data(big_time_embs, inverse_idx, pool_count)
    
    # 对齐检查：如果模型输出特征数少于体素数 (如发生了 stride)
    # 我们通常只保存前 num_out_feat 个，或者需要插值。
    # 这里假设我们只关心几何特征对应的部分 (或者直接保存所有体素的属性)
    if num_out_feat != pool_count:
        print(f"   [Warning] Feature size mismatch! Keeping geometric features separate.")
        # 在这种情况下，我们必须确保保存的数据长度一致，否则 Dataset 读取会报错
        # 简单策略：Pad 几何特征到 pool_count (如果 pool_count 更大)
        if num_out_feat < pool_count:
             pad_feat = torch.zeros((pool_count - num_out_feat, token_geo_feat.shape[1]), 
                                    device=device, dtype=token_geo_feat.dtype)
             token_geo_feat = torch.cat([token_geo_feat, pad_feat], dim=0)
        else:
             token_geo_feat = token_geo_feat[:pool_count]

    # --- Step 4: Save Scene Latent ---
    print(f">> Step 4: Saving Scene Latent...")
    
    scene_data = {
        "geo_feat": token_geo_feat.cpu().half(),  
        "coord": input_dict["coord"].cpu().half(), # 使用 input coord 也就是 Voxel Grid Coords
        "color": token_color.cpu().half(),        
        "cam_emb": token_cam.cpu().half(),       
        "time_emb": token_time.cpu().half(),      
        "num_frames": num_frames
    }
    
    save_path = os.path.join(OUTPUT_ROOT, "scene_latent.pt")
    torch.save(scene_data, save_path)
    
    # 可视化
    try:
        pcd = o3d.geometry.PointCloud()
        # 使用 input_dict["coord"] 作为可视化坐标，因为它与 color 是一一对应的
        pcd.points = o3d.utility.Vector3dVector(input_dict["coord"].cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(token_color.cpu().numpy())
        o3d.io.write_point_cloud(os.path.join(OUTPUT_ROOT, "scene_vis.ply"), pcd)
    except Exception as e:
        print(f"[Warning] Visualization failed: {e}")

    print(f"\n[Done] Scene Latent saved to: {save_path}")
    print(f"       Visualization saved to: {os.path.join(OUTPUT_ROOT, 'scene_vis.ply')}")

if __name__ == "__main__":
    run_pipeline()