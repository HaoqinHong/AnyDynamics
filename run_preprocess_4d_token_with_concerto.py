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

# ================= 0. 全局配置 (请根据您的环境修改路径) =================

# [路径配置]
# 请确保指向包含 concerto 文件夹的父目录 (即 Pointcept 根目录或您解压 Concerto 的目录)
CONCERTO_LIB_PATH = "/opt/data/private/Ours-Projects/Physics-Simulator-World-Model/AnyDynamics/submodules/Concerto" 

# DA3 模型路径
DA3_MODEL_NAME = "/opt/data/private/models/depthanything3/DA3-GIANT" # 或您的本地绝对路径

# Concerto 模型配置
# 建议使用 "concerto_large" 以获得最佳的 2D-3D 语义对齐
CONCERTO_MODEL_NAME = "concerto_large" 
# 如果您已经下载了权重，可以指定本地路径，否则留空会自动下载
CONCERTO_WEIGHT_PATH = "/opt/data/private/models/concerto/concerto_large.pth"  

# 输入输出
INPUT_VIDEO_DIR = "/opt/data/private/datasets/davis_2016/DAVIS_2016/JPEGImages/1080p/bear"
OUTPUT_ROOT = "./demo/bear/scene_token_data_concerto"

# [核心参数]
VOXEL_SIZE = 0.02        # 体素大小 (决定 Token 的密度)
POINTS_PER_FRAME = 50000 # 显存控制 (每帧采样点数)
CONF_THRESH = 0.70       # DA3 置信度阈值
MAX_DEPTH_DIST = 50.0    # 深度截断

# [Embedding 参数]
CAM_EMBED_DIM = 16       
TIME_EMBED_DIM = 8       

# ================= 1. 环境初始化 =================

# 添加 Concerto 路径
if CONCERTO_LIB_PATH not in sys.path:
    sys.path.append(CONCERTO_LIB_PATH)

try:
    import concerto
    from concerto.transform import Compose
    from src.depth_anything_3.api import DepthAnything3
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Please check CONCERTO_LIB_PATH: {CONCERTO_LIB_PATH}")
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
    
    # 1. Load DA3
    print(f">> Loading Depth Anything 3...")
    da3_model = DepthAnything3.from_pretrained(DA3_MODEL_NAME, dynamic=True).to(device)
    da3_model.eval()
    
    # 2. Load Concerto
    print(f">> Loading Concerto ({CONCERTO_MODEL_NAME})...")
    # 尝试检测 flash attention
    try:
        import flash_attn
        enable_flash = True
    except ImportError:
        enable_flash = False
        print("   [Info] FlashAttention not found, using standard attention.")

    if CONCERTO_WEIGHT_PATH and os.path.exists(CONCERTO_WEIGHT_PATH):
        # 本地加载
        concerto_model = concerto.model.load(CONCERTO_WEIGHT_PATH).to(device)
    else:
        # HuggingFace 加载
        custom_config = None
        if not enable_flash:
            custom_config = dict(
                enc_patch_size=[1024 for _ in range(5)], 
                enable_flash=False
            )
        concerto_model = concerto.model.load(
            CONCERTO_MODEL_NAME, 
            repo_id="Pointcept/Concerto", 
            custom_config=custom_config
        ).to(device)
    
    concerto_model.eval()

    # --- Data Prep ---
    frames = sorted(glob.glob(os.path.join(INPUT_VIDEO_DIR, "*.jpg")) + 
                    glob.glob(os.path.join(INPUT_VIDEO_DIR, "*.png")))
    num_frames = len(frames)
    if num_frames == 0:
        print(f"[Error] No images found in {INPUT_VIDEO_DIR}")
        return
        
    print(f">> Processing {num_frames} frames (All-in-One Mode)...")

    # --- Step 1: 批量 DA3 预测 & 数据收集 (Dense Point Cloud) ---
    all_coords = []
    all_colors = []
    all_cam_embs = []  
    all_time_embs = [] 
    
    print(f">> Step 1: Generating Dense Point Clouds with DA3...")
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
    
    # 处理外参
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
        
        # 深度过滤与置信度过滤
        mask = (d > 0) & (d < MAX_DEPTH_DIST)
        if confs is not None:
            mask = mask & (confs[i] > CONF_THRESH)
            
        indices = torch.nonzero(mask).squeeze()
        # 随机采样以控制显存
        if len(indices) > POINTS_PER_FRAME:
            perm = torch.randperm(len(indices))[:POINTS_PER_FRAME]
            indices = indices[perm]
            
        y, x = indices[:, 0], indices[:, 1]
        z = d[y, x]
        
        # 反投影
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]
        X_c = (x - cx) * z / fx
        Y_c = (y - cy) * z / fy
        pts_c = torch.stack([X_c, Y_c, z], dim=-1)
        pts_w = (c2w_mat[:3, :3] @ pts_c.T).T + c2w_mat[:3, 3] 
        
        colors = img[y, x]
        
        N_points = pts_w.shape[0]
        
        # 准备 Embeddings
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

    # --- Step 2: Concerto Global Processing (Tokenization) ---
    print(f">> Step 2: Running Concerto GridSample & Encoding...")
    
    # 构建 Concerto 预处理 Pipeline
    # 使用与 demo 类似的配置，但适配您的 Voxel Size
    concerto_transform = Compose([
        dict(type="CenterShift", apply_z=True),
        dict(
            type="GridSample",
            grid_size=VOXEL_SIZE, 
            hash_type="fnv",
            mode="train", # 必须是 train 模式以生成 index
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
    
    # 准备输入数据 (Numpy)
    input_dict = {
        "coord": big_coords.cpu().numpy(), 
        "color": big_colors.cpu().numpy()  
    }
    
    # 执行 Transform (Raw -> Grid)
    input_dict = concerto_transform(input_dict)
    
    # [Auto-Fix] Channel Padding (6 -> 9)
    # Concerto Large 可能期望 (coord, color, normal)，如果缺少 normal 需补零
    feat = input_dict["feat"] 
    if feat.shape[1] == 6:
        print(f"   [Auto-Fix] Padding features from 6 to 9 channels (adding dummy normals for Concerto)...")
        dummy_normal = torch.zeros((feat.shape[0], 3), dtype=feat.dtype)
        input_dict["feat"] = torch.cat([feat, dummy_normal], dim=1)

    # Move to GPU
    for k in input_dict.keys():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = input_dict[k].to(device)
    input_dict["offset"] = torch.tensor([input_dict["coord"].shape[0]], device=device)
    
    # --- Forward Pass ---
    print("   Running Concerto Inference...")
    with torch.no_grad():
        # Concerto forward return a Point structure
        output_point = concerto_model(input_dict)
        
    # --- Step 3: Upcast Features (关键步骤: 恢复到 Grid 分辨率) ---
    print("   [Critical] Upcasting features back to Grid Resolution...")
    # Concerto/PTv3 会在网络内部下采样，我们需要 undo 这些 pooling
    # 直到特征数量变回与 input_dict["coord"] 一致 (即 0.02m Grid 的数量)
    
    # 循环上采样，逻辑参考 Concerto demo/0_pca.py
    # 我们只运行这个循环，直到不再有 pooling_parent (回到最精细层)
    point = output_point
    
    # 1. 显式上采样循环
    while "pooling_parent" in point.keys():
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        # 将当前层特征(point.feat) 映射回 父层(parent)
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1) # 或者是直接 point.feat[inverse] 取决于架构，Concerto demo 是 cat
        point = parent

    # 此时 point.feat 应该是最精细层 (Grid Level) 的特征
    token_geo_feat = point.feat
    
    # 验证对齐
    num_voxels = input_dict["coord"].shape[0]
    num_out_feat = token_geo_feat.shape[0]
    inverse_idx = input_dict["inverse"] # Raw -> Grid 索引
    
    print(f"   Input Voxels (Grid): {num_voxels}")
    print(f"   Output Features    : {num_out_feat}")
    
    if num_out_feat != num_voxels:
        print(f"[Warning] Size mismatch after upcasting! {num_out_feat} vs {num_voxels}")
        # 如果仍有不匹配，通常是因为最后一次 GridSample 导致的轻微差异或网络结构不同
        # 但通常执行完 while 循环后应当对齐
        if num_out_feat < num_voxels:
            print("   Padding features to match voxel count...")
            pad = torch.zeros((num_voxels - num_out_feat, token_geo_feat.shape[1]), device=device, dtype=token_geo_feat.dtype)
            token_geo_feat = torch.cat([token_geo_feat, pad], dim=0)
        else:
            token_geo_feat = token_geo_feat[:num_voxels]
    else:
        print("   [Success] Features aligned perfectly with Grid.")

    # --- Step 4: Pooling Attributes to Tokens ---
    print(f">> Step 4: Pooling Attribute Embeddings to Tokens...")
    
    pool_count = num_voxels

    def pool_data(data, idx, count):
        C = data.shape[1]
        out = torch.zeros((count, C), device=device, dtype=data.dtype)
        ones = torch.ones((len(idx), 1), device=device, dtype=data.dtype)
        cnt = torch.zeros((count, 1), device=device, dtype=data.dtype)
        
        if idx.max() >= count:
             count = idx.max().item() + 1
             out = torch.zeros((count, C), device=device, dtype=data.dtype)
             cnt = torch.zeros((count, 1), device=device, dtype=data.dtype)

        out.index_add_(0, idx, data)
        cnt.index_add_(0, idx, ones)
        return out / (cnt + 1e-6)

    # 聚合 Dense 属性到 Voxel Grid Token
    token_color = pool_data(big_colors, inverse_idx, pool_count)
    token_cam = pool_data(big_cam_embs, inverse_idx, pool_count)
    token_time = pool_data(big_time_embs, inverse_idx, pool_count)

    # --- Step 5: Save Scene Latent ---
    print(f">> Step 5: Saving Scene Latent (Concerto Encoded)...")
    
    scene_data = {
        "geo_feat": token_geo_feat.cpu().half(),   # Concerto 特征 (FP16)
        "coord": input_dict["coord"].cpu().half(), # Voxel Grid 坐标
        "color": token_color.cpu().half(),         # 平均颜色
        "cam_emb": token_cam.cpu().half(),       
        "time_emb": token_time.cpu().half(),      
        "num_frames": num_frames
    }
    
    save_path = os.path.join(OUTPUT_ROOT, "scene_latent.pt")
    torch.save(scene_data, save_path)
    
    # 可视化 (保存点云以供检查)
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(input_dict["coord"].cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(token_color.cpu().numpy())
        o3d.io.write_point_cloud(os.path.join(OUTPUT_ROOT, "scene_vis.ply"), pcd)
    except Exception as e:
        print(f"[Warning] Visualization failed: {e}")

    print(f"\n[Done] Scene Latent saved to: {save_path}")
    print(f"       Visualization saved to: {os.path.join(OUTPUT_ROOT, 'scene_vis.ply')}")

if __name__ == "__main__":
    run_pipeline()