import sys
import os
import glob
import torch
import numpy as np
import cv2
import open3d as o3d
from tqdm import tqdm

# ================= 0. 全局配置 (参数已调优) =================

SONATA_LIB_PATH = "/backup/group_朱聪聪/hqhong/projects/AnyDynamics/submodules/sonata"
DA3_MODEL_NAME = "/backup/group_朱聪聪/hqhong/models/DA3-GIANT" 
SONATA_CKPT_PATH = "/backup/group_朱聪聪/hqhong/models/sonata/sonata.pth"
INPUT_VIDEO_DIR = "./demo/bear"
OUTPUT_DIR = "./demo/bear_test/optimized_result"

# [关键调整] 动静分离参数
# 1. 判定用的体素大小 (米)
#    越小越灵敏，物体稍微一动就被算作动态。建议 0.02 (2cm)
SEPARATION_VOXEL_SIZE = 0.02  

# 2. 静态阈值比例 (0.0 ~ 1.0)
#    如果一个位置在 (总帧数 * Ratio) 的帧里都出现过，才算静态。
#    调大这个值 (如 0.3)，会让更多“半动半静”的物体被归类为动态。
STATIC_THRESHOLD_RATIO = 0.15 

# [新增] 导出时的降采样参数
# 静态背景通常需要大幅去重，动态轨迹希望保留细节
EXPORT_STATIC_VOXEL_SIZE = 0.02  # 2cm 降采样，大幅减小文件体积
EXPORT_DYNAMIC_VOXEL_SIZE = 0.0  # 0 表示不降采样，保留所有动态细节

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

# ================= 2. 算法函数 =================

def compute_normals_camera_space(depth, intrinsics):
    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    valid_mask = (depth > 0) & np.isfinite(depth)
    X, Y = np.zeros_like(depth), np.zeros_like(depth)
    X[valid_mask] = (x[valid_mask] - cx) * depth[valid_mask] / fx
    Y[valid_mask] = (y[valid_mask] - cy) * depth[valid_mask] / fy
    Z = depth.copy()

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
    return normals.astype(np.float32)

def transform_to_world(points, normals, w2c_matrix):
    if w2c_matrix.shape == (3, 4):
        bottom_row = np.array([[0, 0, 0, 1]], dtype=w2c_matrix.dtype)
        w2c_matrix = np.vstack([w2c_matrix, bottom_row])
    
    try:
        c2w = np.linalg.inv(w2c_matrix)
    except np.linalg.LinAlgError:
        return points, normals

    rotation = c2w[:3, :3]
    translation = c2w[:3, 3]
    points_world = (rotation @ points.T).T + translation
    normals_world = (rotation @ normals.T).T
    return points_world, normals_world

def separate_static_dynamic(points, frame_indices, voxel_size, threshold):
    print(f"Separating... Grid: {voxel_size}m, Static Thresh: >{threshold} frames")
    
    # 1. 坐标体素化
    grid_coords = np.floor(points / voxel_size).astype(np.int64)
    
    # 2. 获取每个点的体素ID
    # 使用 axis=0 对 (N, 3) 数组进行 unique
    # return_inverse 得到的 voxel_ids 是每个点对应的唯一体素索引
    _, voxel_ids = np.unique(grid_coords, axis=0, return_inverse=True)
    
    # 3. 统计每个体素被多少个唯一帧观测到
    # 构造 (Voxel_ID, Frame_ID)
    # 使用 int64 避免溢出，将两个 ID以此哈希或其他方式组合，或者直接用 np.unique on rows
    pairs = np.stack([voxel_ids, frame_indices], axis=1)
    unique_pairs = np.unique(pairs, axis=0) # 去重：同一个体素在同一帧里有多个点只算1次
    
    # 统计 Voxel_ID 出现的次数 (即它包含的 Unique Frame 数量)
    # unique_pairs[:, 0] 是去重后的 pair 里的 Voxel_ID 列
    unique_voxel_ids_in_pairs = unique_pairs[:, 0]
    
    # 统计每个体素出现的次数。注意：bincount 的长度是 max_id + 1
    frame_counts_per_voxel = np.bincount(unique_voxel_ids_in_pairs)
    
    # 4. 判定
    # 映射回每个点：查看该点所属的 voxel 的 frame_count
    # 注意处理边界：bincount 长度可能小于 voxel_ids 的最大值吗？
    # 不会，因为 voxel_ids 来源于 unique，unique_pairs 包含了所有存在的 voxel_id
    
    point_frame_counts = frame_counts_per_voxel[voxel_ids]
    
    is_static = point_frame_counts > threshold
    
    return is_static

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
    
    if not frames:
        print("No frames found.")
        return
    
    step = 1
    frames = frames[::step]
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

    depths = prediction.depth
    intrinsics = prediction.intrinsics
    extrinsics = prediction.extrinsics
    images = prediction.processed_images

    # --- Step 2: Fusion & Separation ---
    print(f"\n[Step 2] Fusing & Analyzing...")
    
    all_points = []
    all_colors = []
    all_normals = []
    all_frame_indices = []

    for i in tqdm(range(len(frames))):
        d_map = depths[i]
        K = intrinsics[i]
        img = images[i]
        w2c = extrinsics[i]

        n_map = compute_normals_camera_space(d_map, K)

        H, W = d_map.shape
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        
        valid_mask = (d_map > 0) & np.isfinite(d_map)
        valid_indices = np.where(valid_mask.reshape(-1))[0]
        
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

        pts_world, norm_world = transform_to_world(pts_cam, norm_flat, w2c)

        all_points.append(pts_world)
        all_colors.append(col_flat)
        all_normals.append(norm_world)
        
        frame_ids = np.full((pts_world.shape[0],), i, dtype=np.int32)
        all_frame_indices.append(frame_ids)

    # 合并
    full_coord = np.concatenate(all_points, axis=0).astype(np.float32)
    full_color = np.concatenate(all_colors, axis=0).astype(np.float32)
    full_normal = np.concatenate(all_normals, axis=0).astype(np.float32)
    full_indices = np.concatenate(all_frame_indices, axis=0)

    print(f"Raw Fused Points: {len(full_coord)}")

    # [执行分离]
    # 计算阈值: 总帧数的 15%
    static_threshold_count = int(num_frames * STATIC_THRESHOLD_RATIO)
    # 至少要 2 帧以上才能算 static，防止单帧误判
    static_threshold_count = max(2, static_threshold_count) 
    
    static_mask = separate_static_dynamic(
        full_coord, 
        full_indices, 
        voxel_size=SEPARATION_VOXEL_SIZE, 
        threshold=static_threshold_count
    )
    
    # 拆分数据
    static_coord = full_coord[static_mask]
    static_color = full_color[static_mask]
    static_normal = full_normal[static_mask]
    
    dynamic_coord = full_coord[~static_mask]
    dynamic_color = full_color[~static_mask]
    dynamic_normal = full_normal[~static_mask]

    print(f"Static Points (Raw): {len(static_coord)}")
    print(f"Dynamic Points (Raw): {len(dynamic_coord)}")

    # --- Step 3: 优化与保存 (Downsampling) ---
    print(f"\n[Step 3] Optimizing & Saving...")

    def save_optimized_pcd(coords, colors, normals, filename, voxel_size=0.0):
        if len(coords) == 0:
            print(f"Skipping {filename} (empty)")
            return
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        
        # 降采样
        if voxel_size > 0:
            print(f"  Downsampling {filename} with voxel={voxel_size}...")
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        
        # 保存 (Binary format is smaller and faster)
        # write_ascii=False 是默认的，但这里显式确认一下
        save_path = os.path.join(OUTPUT_DIR, filename)
        o3d.io.write_point_cloud(save_path, pcd, write_ascii=False)
        print(f"  Saved {filename}: {len(pcd.points)} points")
        return pcd

    # 1. 静态背景：强力降采样 (去重)
    pcd_static_opt = save_optimized_pcd(
        static_coord, static_color, static_normal, 
        "scene_static_optimized.ply", 
        voxel_size=EXPORT_STATIC_VOXEL_SIZE
    )

    # 2. 动态物体：保留细节 (不降采样或轻微降采样)
    save_optimized_pcd(
        dynamic_coord, dynamic_color, dynamic_normal, 
        "scene_dynamic_trajectory.ply", 
        voxel_size=EXPORT_DYNAMIC_VOXEL_SIZE
    )

    # --- Step 4: Sonata Feature Extraction (针对动态部分) ---
    # 通常我们更关心动态物体的特征
    print(f"\n[Step 4] Extracting features for Dynamic Objects...")
    
    if len(dynamic_coord) > 0 and os.path.exists(SONATA_CKPT_PATH):
        sonata_model = sonata.load(SONATA_CKPT_PATH).to(device)
        sonata_model.eval()
        
        # 构建 transform
        # 注意：这里我们重新把数据转为 Sonata 格式
        # Sonata 内部也会做 GridSample，但这只是为了 Tokenization
        transform = Compose([
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.02, # Token size
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
                feat_keys=("coord", "color", "normal") if sonata_model.embedding.stem.linear.in_features==9 else ("coord", "color"),
            ),
        ])

        input_dict = {
            "coord": dynamic_coord,
            "color": dynamic_color,
            "normal": dynamic_normal
        }
        
        input_dict = transform(input_dict)
        
        with torch.no_grad():
            for k in input_dict.keys():
                if isinstance(input_dict[k], torch.Tensor):
                    input_dict[k] = input_dict[k].to(device)
            input_dict["offset"] = torch.tensor([input_dict["coord"].shape[0]], device=device)
            
            out = sonata_model(input_dict)
            feats = out.feat
            
        torch.save(feats.cpu(), os.path.join(OUTPUT_DIR, "dynamic_features.pt"))
        print(f"Extracted {len(feats)} features for dynamic points.")

    print("Done.")

if __name__ == "__main__":
    run_pipeline()