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

# [路径配置]
SONATA_LIB_PATH = "/opt/data/private/Ours-Projects/Physics-Simulator-World-Model/AnyDynamics/submodules/sonata"
DA3_MODEL_NAME = "/opt/data/private/models/depthanything3/DA3-GIANT" 
SONATA_CKPT_PATH = "/opt/data/private/models/sonata/sonata.pth"

INPUT_VIDEO_DIR = "/opt/data/private/datasets/davis_2016/DAVIS_2016/JPEGImages/1080p/bear"
OUTPUT_ROOT = "./demo/bear/scene_token_data" 

# [核心参数]
VOXEL_SIZE = 0.005        
POINTS_PER_FRAME = 50000 
CONF_THRESH = 0.70       
MAX_DEPTH_DIST = 50.0    

# [Embedding 参数]
RAY_EMBED_DIM = 16       
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

def get_ray_embedding(pts_w, c2w, dim=16):
    """计算相对视线向量编码 (Ray Direction Embedding)"""
    cam_pos = c2w[:3, 3].to(pts_w.device) 
    view_dirs = pts_w - cam_pos.unsqueeze(0) 
    view_dirs = F.normalize(view_dirs, p=2, dim=1) 
    if dim > 3:
        return F.pad(view_dirs, (0, dim - 3), "constant", 0)
    else:
        return view_dirs[:, :dim]

def get_time_embedding(t_norm, dim=8):
    """时间正弦编码"""
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim)).to(t_norm.device)
    pe = torch.zeros(dim).to(t_norm.device)
    pe[0::2] = torch.sin(t_norm * 100 * div_term)
    pe[1::2] = torch.cos(t_norm * 100 * div_term)
    return pe

def get_pca_color(feat, brightness=1.25, center=True):
    """PCA 降维可视化"""
    feat = feat.float()
    try:
        u, s, v = torch.pca_lowrank(feat, center=center, q=3, niter=5)
        projection = feat @ v
        min_val = projection.min(dim=0, keepdim=True)[0]
        max_val = projection.max(dim=0, keepdim=True)[0]
        div = torch.clamp(max_val - min_val, min=1e-6)
        color = (projection - min_val) / div
        color = color * brightness
        color = color.clamp(0.0, 1.0)
        return color.cpu().numpy()
    except Exception as e:
        print(f"[PCA Error] {e}, returning random colors")
        return np.random.rand(feat.shape[0], 3)

# ================= 3. 主流水线 =================

def run_pipeline():
    # 创建输出目录
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    # [New] 创建每帧点云的子目录
    frames_output_dir = os.path.join(OUTPUT_ROOT, "frames")
    os.makedirs(frames_output_dir, exist_ok=True)
    
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
    print(f">> Processing {num_frames} frames...")

    # --- Step 1: DA3 & Point Cloud Generation ---
    all_coords = []
    all_colors = []
    all_ray_embs = []  
    all_time_embs = [] 
    
    # 原始全量点云缓存 (用于最后合并生成 Ground Truth)
    original_vis_coords = []
    original_vis_colors = []
    
    print(f">> Step 1: Generating Dense Point Clouds & Saving Per-Frame...")
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
        
        # Ray Embedding
        ray_vecs = get_ray_embedding(pts_w, c2w_mat, dim=RAY_EMBED_DIM) 
        
        t_norm = torch.tensor(i / (num_frames - 1), device=device).float()
        time_vec = get_time_embedding(t_norm, dim=TIME_EMBED_DIM) 
        time_block = time_vec.unsqueeze(0).expand(N_points, -1)    
        
        all_coords.append(pts_w)
        all_colors.append(colors)
        all_ray_embs.append(ray_vecs)
        all_time_embs.append(time_block)
        
        # [CPU Data] 用于保存和合并
        pts_cpu = pts_w.cpu().numpy()
        col_cpu = colors.cpu().numpy()
        
        # 1. 缓存用于最后的大合并
        # 为了避免内存爆炸，我们还是降采样存全量 (比如每帧都存)
        original_vis_coords.append(pts_cpu)
        original_vis_colors.append(col_cpu)
        
        # 2. [New] 立即保存当前帧点云 (Per-Frame Saving)
        pcd_frame = o3d.geometry.PointCloud()
        pcd_frame.points = o3d.utility.Vector3dVector(pts_cpu)
        pcd_frame.colors = o3d.utility.Vector3dVector(col_cpu)
        o3d.io.write_point_cloud(os.path.join(frames_output_dir, f"frame_{i:03d}.ply"), pcd_frame)

    print(f">> Merging frames...")
    big_coords = torch.cat(all_coords, dim=0)     
    big_colors = torch.cat(all_colors, dim=0)     
    big_ray_embs = torch.cat(all_ray_embs, dim=0) 
    big_time_embs = torch.cat(all_time_embs, dim=0) 

    # --- Step 2: Sonata Processing ---
    print(f">> Step 2: Running Sonata GridSample & Encoding...")
    
    sonata_transform = Compose([
        dict(type="CenterShift", apply_z=True),
        dict(type="GridSample", grid_size=VOXEL_SIZE, hash_type="fnv", mode="train", 
             return_grid_coord=True, return_inverse=True),
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord", "inverse"), feat_keys=("coord", "color")),
    ])
    
    input_dict = {
        "coord": big_coords.cpu().numpy(), 
        "color": big_colors.cpu().numpy()  
    }
    input_dict = sonata_transform(input_dict)
    
    feat = input_dict["feat"] 
    if feat.shape[1] == 6:
        dummy_normal = torch.zeros((feat.shape[0], 3), dtype=feat.dtype)
        input_dict["feat"] = torch.cat([feat, dummy_normal], dim=1)

    for k in input_dict.keys():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = input_dict[k].to(device)
    input_dict["offset"] = torch.tensor([input_dict["coord"].shape[0]], device=device)
    
    with torch.no_grad():
        output = sonata_model(input_dict)
    
    print(f"   Starting feature upcasting...")
    point = output
    while "pooling_parent" in point.keys():
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent

    token_geo_feat = point.feat
    
    num_output_feats = token_geo_feat.shape[0]
    inverse_idx = input_dict["inverse"] 
    print(f"   Feature Dim: {token_geo_feat.shape[1]}") 

    # --- Step 3: Pooling (Centroid Calculation) ---
    print(f">> Step 3: Pooling Embeddings (Calculating Geometric Centroids)...")
    
    def pool_data(data, idx, count):
        C = data.shape[1]
        out = torch.zeros((count, C), device=device, dtype=data.dtype)
        ones = torch.ones((len(idx), 1), device=device, dtype=data.dtype)
        cnt = torch.zeros((count, 1), device=device, dtype=data.dtype)
        if idx.max() >= count:
             idx = torch.clamp(idx, max=count-1)
        out.index_add_(0, idx, data)
        cnt.index_add_(0, idx, ones)
        return out / (cnt + 1e-6)

    token_color = pool_data(big_colors, inverse_idx, num_output_feats)
    token_coord = pool_data(big_coords, inverse_idx, num_output_feats)
    token_ray = pool_data(big_ray_embs, inverse_idx, num_output_feats)
    token_time = pool_data(big_time_embs, inverse_idx, num_output_feats)

    # --- Step 4: Save Scene Latent (Separated & Visualized) ---
    print(f">> Step 4: Saving Separated Features & Visualizing...")
    
    dims = [48, 96, 192, 384, 512]
    
    if token_geo_feat.shape[1] == sum(dims):
        print(f"   Splitting features into 5 stages: {dims}")
        split_feats = torch.split(token_geo_feat, dims, dim=1)
        
        # 1. 保存 .pt
        scene_data = {
            "coord": token_coord.cpu().half(),
            "color": token_color.cpu().half(),
            "ray_emb": token_ray.cpu().half(), 
            "time_emb": token_time.cpu().half(),
            "num_frames": num_frames,
            # 分层特征
            "geo_feat_all": token_geo_feat.cpu().half(),
            "geo_feat_stage_0": split_feats[0].cpu().half(), 
            "geo_feat_stage_1": split_feats[1].cpu().half(), 
            "geo_feat_stage_2": split_feats[2].cpu().half(), 
            "geo_feat_stage_3": split_feats[3].cpu().half(), 
            "geo_feat_stage_4": split_feats[4].cpu().half(), 
        }
        
        save_path = os.path.join(OUTPUT_ROOT, "scene_latent.pt")
        torch.save(scene_data, save_path)
        print(f"   [Saved] Latent data saved to {save_path}")

        # 2. 可视化每一层
        print(f"   Generating visualizations...")
        vis_coord = token_coord.cpu().numpy()
        
        for i, feat_tensor in enumerate(split_feats):
            stage_name = f"stage_{i}"
            pca_rgb = get_pca_color(feat_tensor) 
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vis_coord)
            pcd.colors = o3d.utility.Vector3dVector(pca_rgb)
            o3d.io.write_point_cloud(os.path.join(OUTPUT_ROOT, f"vis_{stage_name}.ply"), pcd)
        
        # 3. 可视化修复后的 RGB
        pcd_rgb = o3d.geometry.PointCloud()
        pcd_rgb.points = o3d.utility.Vector3dVector(vis_coord)
        pcd_rgb.colors = o3d.utility.Vector3dVector(token_color.cpu().numpy())
        o3d.io.write_point_cloud(os.path.join(OUTPUT_ROOT, "vis_voxel_rgb.ply"), pcd_rgb)
        
        # 4. 可视化 DA3 原始点云 (Ground Truth)
        print(f"   Saving original DA3 point cloud for comparison...")
        orig_coords_np = np.concatenate(original_vis_coords, axis=0)
        orig_colors_np = np.concatenate(original_vis_colors, axis=0)
        
        pcd_orig = o3d.geometry.PointCloud()
        pcd_orig.points = o3d.utility.Vector3dVector(orig_coords_np)
        pcd_orig.colors = o3d.utility.Vector3dVector(orig_colors_np)
        o3d.io.write_point_cloud(os.path.join(OUTPUT_ROOT, "vis_original_da3.ply"), pcd_orig)
        
        print(f"   [Done] All Visualizations & Per-Frame Clouds saved to {OUTPUT_ROOT}")

    else:
        # Fallback
        scene_data = {
            "geo_feat": token_geo_feat.cpu().half(),
            "coord": token_coord.cpu().half(),
            "color": token_color.cpu().half(),
            "ray_emb": token_ray.cpu().half(),
            "time_emb": token_time.cpu().half(),
            "num_frames": num_frames
        }
        torch.save(scene_data, os.path.join(OUTPUT_ROOT, "scene_latent.pt"))
        print(f"   [Saved] Combined latent saved.")

if __name__ == "__main__":
    run_pipeline()