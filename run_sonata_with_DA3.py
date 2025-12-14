import sys
import os
import glob
import torch
import numpy as np
import cv2
import open3d as o3d
from tqdm import tqdm

# ================= 0. 全局配置 (请修改这里) =================

# 1. Sonata 源代码路径 (包含 sonata 文件夹的父目录)
SONATA_LIB_PATH = "/backup/group_朱聪聪/hqhong/projects/AnyDynamics/submodules/sonata"

# 2. DA3 模型名称或路径
DA3_MODEL_NAME = "/backup/group_朱聪聪/hqhong/models/DA3-GIANT" 

# 3. Sonata 预训练权重路径 (.pth 文件)
SONATA_CKPT_PATH = "/backup/group_朱聪聪/hqhong/models/sonata/sonata.pth"

# 4. 输入视频帧文件夹
INPUT_VIDEO_DIR = "./demo/bear"

# 5. 输出文件夹
OUTPUT_DIR = "./demo/bear_test/fused_sonata_result"

# ================= 1. 环境初始化 =================

if SONATA_LIB_PATH not in sys.path:
    sys.path.append(SONATA_LIB_PATH)

try:
    import sonata
    from sonata.transform import Compose, Collect
    from src.depth_anything_3.api import DepthAnything3
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please check your file paths and environment.")
    sys.exit(1)

# ================= 2. 核心算法函数 =================

def compute_normals_camera_space(depth, intrinsics):
    """
    从深度图计算法线 (Camera Space)
    """
    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Back-project
    valid_mask = (depth > 0) & np.isfinite(depth)
    X = np.zeros_like(depth)
    Y = np.zeros_like(depth)
    X[valid_mask] = (x[valid_mask] - cx) * depth[valid_mask] / fx
    Y[valid_mask] = (y[valid_mask] - cy) * depth[valid_mask] / fy
    Z = depth.copy()

    # Gradients
    ksize = 5
    dX_du = cv2.Sobel(X, cv2.CV_64F, 1, 0, ksize=ksize)
    dY_du = cv2.Sobel(Y, cv2.CV_64F, 1, 0, ksize=ksize)
    dZ_du = cv2.Sobel(Z, cv2.CV_64F, 1, 0, ksize=ksize)
    dX_dv = cv2.Sobel(X, cv2.CV_64F, 0, 1, ksize=ksize)
    dY_dv = cv2.Sobel(Y, cv2.CV_64F, 0, 1, ksize=ksize)
    dZ_dv = cv2.Sobel(Z, cv2.CV_64F, 0, 1, ksize=ksize)

    # Cross Product
    tu = np.stack([dX_du, dY_du, dZ_du], axis=-1)
    tv = np.stack([dX_dv, dY_dv, dZ_dv], axis=-1)
    normals = np.cross(tu, tv)

    # Normalize
    norm_mag = np.linalg.norm(normals, axis=-1, keepdims=True)
    norm_mag[norm_mag < 1e-6] = 1e-6 
    normals = normals / norm_mag
    
    # Orient towards camera
    mask_flip = normals[..., 2] > 0
    normals[mask_flip] *= -1

    return normals.astype(np.float32)

def transform_to_world(points, normals, w2c_matrix):
    """
    将点和法线从相机系转到世界系
    w2c_matrix: (4, 4) or (3, 4) World-to-Camera matrix
    """
    # [修复关键点] 检查矩阵形状，如果是 3x4 则补全为 4x4
    if w2c_matrix.shape == (3, 4):
        # 添加最后一行 [0, 0, 0, 1]
        bottom_row = np.array([[0, 0, 0, 1]], dtype=w2c_matrix.dtype)
        w2c_matrix = np.vstack([w2c_matrix, bottom_row])
    
    # 1. 计算 C2W (Camera-to-World)
    # DA3 输出通常是 w2c，求逆得到 c2w
    try:
        c2w = np.linalg.inv(w2c_matrix)
    except np.linalg.LinAlgError:
        print("Error: Singular matrix, cannot invert W2C matrix.")
        return points, normals # 失败时返回原值

    rotation = c2w[:3, :3]
    translation = c2w[:3, 3]

    # 2. 变换点坐标: P_world = R * P_cam + T
    # points shape: (N, 3)
    points_world = (rotation @ points.T).T + translation

    # 3. 变换法线: N_world = R * N_cam (法线不受平移影响)
    normals_world = (rotation @ normals.T).T

    return points_world, normals_world

# ================= 3. 主要处理流程 =================

def run_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Step 1: 加载 DA3 并进行推理 (获取位姿) ---
    print(f"\n[Step 1] Loading Depth Anything 3 ({DA3_MODEL_NAME})...")
    da3_model = DepthAnything3.from_pretrained(DA3_MODEL_NAME, dynamic=True).to(device)
    da3_model.eval()

    # 读取图片
    frames = sorted(glob.glob(os.path.join(INPUT_VIDEO_DIR, "*.jpg")) + 
                    glob.glob(os.path.join(INPUT_VIDEO_DIR, "*.png")))
    
    if not frames:
        print(f"Error: No images found in {INPUT_VIDEO_DIR}")
        return

    frames = frames[::1] 
    print(f"Found {len(frames)} frames. Running inference...")

    with torch.no_grad():
        prediction = da3_model.inference(
            image=frames,
            align_to_input_ext_scale=True,
            infer_gs=True, 
            process_res=504,
            export_dir=None, 
            export_format="mini_npz"
        )

    depths = prediction.depth
    intrinsics = prediction.intrinsics
    extrinsics = prediction.extrinsics # 可能返回 (N, 3, 4) 或 (N, 4, 4)
    images = prediction.processed_images

    # --- Step 2: 多帧融合 (Temporal Fusion) ---
    print(f"\n[Step 2] Fusing {len(frames)} frames into World Space...")
    
    all_points = []
    all_colors = []
    all_normals = []

    for i in tqdm(range(len(frames))):
        # A. 基础数据
        d_map = depths[i]
        K = intrinsics[i]
        img = images[i]
        w2c = extrinsics[i]

        # B. 计算法线 (Camera Space)
        n_map = compute_normals_camera_space(d_map, K)

        # C. 反投影 (Camera Space Points)
        H, W = d_map.shape
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        
        valid_mask = (d_map > 0) & np.isfinite(d_map)
        valid_indices = np.where(valid_mask.reshape(-1))[0]
        # 降采样
        if len(valid_indices) > 50000:
            chosen = np.random.choice(valid_indices, 50000, replace=False)
        else:
            chosen = valid_indices

        x_flat = x.reshape(-1)[chosen]
        y_flat = y.reshape(-1)[chosen]
        z_flat = d_map.reshape(-1)[chosen]
        
        X_cam = (x_flat - cx) * z_flat / fx
        Y_cam = (y_flat - cy) * z_flat / fy
        Z_cam = z_flat
        pts_cam = np.stack([X_cam, Y_cam, Z_cam], axis=-1)
        
        col_flat = img.reshape(-1, 3)[chosen]
        norm_flat = n_map.reshape(-1, 3)[chosen]

        # D. 变换到世界系 (World Space)
        # 这里的 w2c 可能是 3x4，transform_to_world 会自动修复它
        pts_world, norm_world = transform_to_world(pts_cam, norm_flat, w2c)

        # E. 收集
        all_points.append(pts_world)
        all_colors.append(col_flat)
        all_normals.append(norm_world)

    # 合并所有帧数据
    if not all_points:
        print("Error: No valid points generated.")
        return

    full_coord = np.concatenate(all_points, axis=0).astype(np.float32)
    full_color = np.concatenate(all_colors, axis=0).astype(np.float32)
    full_normal = np.concatenate(all_normals, axis=0).astype(np.float32)

    print(f"Fused Point Cloud Size: {full_coord.shape[0]} points")

    # --- Step 3: 加载 Sonata 并提取特征 ---
    print(f"\n[Step 3] Loading Sonata Model from {SONATA_CKPT_PATH}...")
    
    if not os.path.exists(SONATA_CKPT_PATH):
        print(f"Error: Checkpoint not found at {SONATA_CKPT_PATH}")
        return

    sonata_model = sonata.load(SONATA_CKPT_PATH).to(device)
    sonata_model.eval()

    try:
        if hasattr(sonata_model, 'embedding') and hasattr(sonata_model.embedding, 'stem') and hasattr(sonata_model.embedding.stem, 'linear'):
            in_ch = sonata_model.embedding.stem.linear.in_features
        else:
            in_ch = sonata_model.embedding.stem[0].in_features
    except:
        in_ch = 6
    
    if in_ch == 9:
        print(">> Strategy: Using Coord + Color + Normal")
        feat_keys = ("coord", "color", "normal")
    else:
        print(">> Strategy: Using Coord + Color (Model doesn't support Normal input)")
        feat_keys = ("coord", "color")

    transform = Compose([
        dict(type="CenterShift", apply_z=True),
        dict(
            type="GridSample",
            grid_size=0.02,
            hash_type="fnv",
            mode="train",    # <--- 必须是 train
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

    input_dict = {
        "coord": full_coord,
        "color": full_color,
        "normal": full_normal
    }

    print("Running Sonata Preprocessing (Grid Sampling)...")
    input_dict = transform(input_dict)
    
    print("Running Sonata Inference...")
    with torch.no_grad():
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].to(device, non_blocking=True)
        
        input_dict["offset"] = torch.tensor([input_dict["coord"].shape[0]], device=device)
        
        output = sonata_model(input_dict)
        features = output.feat

    # --- Step 4: 保存结果 ---
    print("\n[Step 4] Saving results...")
    
    pcd_fused = o3d.geometry.PointCloud()
    pcd_fused.points = o3d.utility.Vector3dVector(full_coord)
    pcd_fused.colors = o3d.utility.Vector3dVector(full_color / 255.0)
    pcd_fused.normals = o3d.utility.Vector3dVector(full_normal)
    o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, "fused_scene.ply"), pcd_fused)
    
    torch.save(features.cpu(), os.path.join(OUTPUT_DIR, "sonata_features.pt"))
    
    print(f"Fused PLY saved to: {os.path.join(OUTPUT_DIR, 'fused_scene.ply')}")
    print(f"Features saved to: {os.path.join(OUTPUT_DIR, 'sonata_features.pt')}")
    print(f"Original Points: {len(full_coord)}")
    print(f"Tokenized Features: {features.shape}")

if __name__ == "__main__":
    run_pipeline()