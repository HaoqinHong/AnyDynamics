import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from src.depth_anything_3.api import DepthAnything3

# ================= 工具函数：计算法线与保存 PLY =================

def compute_normals(depth, intrinsics):
    """
    从深度图和内参计算表面法线 (Camera Coordinate System)
    """
    H, W = depth.shape
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    # 1. 生成图像坐标网格
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    # 2. 反投影到 3D 坐标 (X, Y, Z)
    valid_mask = (depth > 0) & np.isfinite(depth)
    
    X = np.zeros_like(depth)
    Y = np.zeros_like(depth)
    
    X[valid_mask] = (x[valid_mask] - cx) * depth[valid_mask] / fx
    Y[valid_mask] = (y[valid_mask] - cy) * depth[valid_mask] / fy
    Z = depth.copy()

    # 3. 计算梯度 (利用 Sobel 算子抗噪)
    ksize = 5
    dX_du = cv2.Sobel(X, cv2.CV_64F, 1, 0, ksize=ksize)
    dY_du = cv2.Sobel(Y, cv2.CV_64F, 1, 0, ksize=ksize)
    dZ_du = cv2.Sobel(Z, cv2.CV_64F, 1, 0, ksize=ksize)

    dX_dv = cv2.Sobel(X, cv2.CV_64F, 0, 1, ksize=ksize)
    dY_dv = cv2.Sobel(Y, cv2.CV_64F, 0, 1, ksize=ksize)
    dZ_dv = cv2.Sobel(Z, cv2.CV_64F, 0, 1, ksize=ksize)

    # 4. 构造切向量并叉积得到法线
    tu = np.stack([dX_du, dY_du, dZ_du], axis=-1)
    tv = np.stack([dX_dv, dY_dv, dZ_dv], axis=-1)
    normals = np.cross(tu, tv)

    # 5. 归一化
    norm_mag = np.linalg.norm(normals, axis=-1, keepdims=True)
    norm_mag[norm_mag < 1e-6] = 1e-6 
    normals = normals / norm_mag
    
    # 6. 方向修正：确保法线指向相机 (Z < 0)
    mask_flip = normals[..., 2] > 0
    normals[mask_flip] *= -1

    return normals.astype(np.float32)

def save_ply_with_normals(filename, points, colors, normals):
    """
    保存带法线的 PLY 文件
    """
    header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property float nx
property float ny
property float nz
end_header
""".format(len(points))

    with open(filename, 'w') as f:
        f.write(header)
        for i in range(len(points)):
            p = points[i]
            c = colors[i]
            n = normals[i]
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]} {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")

# ================= 主程序 =================

def main():
    # Set args
    start_idx = 10
    max_frames = 10
    step = 1
    read_dir = './demo/bear'
    # 修改输出路径，专门存放 ply
    save_dir = f"./demo/bear_test/results/start-{start_idx}_max-{max_frames}_step-{step}_ply"
    
    os.makedirs(save_dir, exist_ok=True)

    # Load frame paths
    frames = []
    if os.path.exists(read_dir):
        for file in sorted(os.listdir(read_dir)):
            if file.endswith((".jpg", ".png")):
                frames.append(os.path.join(read_dir, file))
    else:
        print(f"Directory {read_dir} does not exist.")
        return

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading model from pretrained...")
    # 请确保这里的路径是正确的，可以是本地路径或 HuggingFace ID
    model = DepthAnything3.from_pretrained("/opt/data/private/models/depthanything3/DA3-GIANT", dynamic=True)
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # Inference
    print("Running inference...")
    with torch.no_grad():
        prediction = model.inference(
            image=frames,
            align_to_input_ext_scale=True,
            infer_gs=True, 
            process_res=504,
            process_res_method="upper_bound_resize",
            # 1. export_dir = None: 禁止模型内部自动导出
            # 2. export_format = "mini_npz": 给一个有效字符串
            export_dir=None, 
            export_format="mini_npz", 
        )

    # ================= 自定义导出：带法线的 PLY =================
    print("\nStarting export of PLY with Normals...")
    
    if hasattr(prediction, 'depth') and hasattr(prediction, 'processed_images'):
        depths = prediction.depth
        intrinsics_list = prediction.intrinsics
        images = prediction.processed_images
        
        num_frames = depths.shape[0]
        
        # 定义输出子文件夹
        ply_out_dir = os.path.join(save_dir, "ply_normals")
        os.makedirs(ply_out_dir, exist_ok=True)

        for i in tqdm(range(num_frames), desc="Exporting PLYs"):
            depth = depths[i]       # (H, W)
            K = intrinsics_list[i]  # (3, 3)
            img = images[i]         # (H, W, 3)

            # 1. 计算法线
            normals_map = compute_normals(depth, K) # (H, W, 3)

            # 2. 反投影点云
            H, W = depth.shape
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            X = (x - cx) * depth / fx
            Y = (y - cy) * depth / fy
            Z = depth
            
            points_3d = np.stack([X, Y, Z], axis=-1)

            # 3. Flatten & Filter
            pts_flat = points_3d.reshape(-1, 3)
            col_flat = img.reshape(-1, 3)
            norm_flat = normals_map.reshape(-1, 3)
            depth_flat = depth.reshape(-1)

            mask = (depth_flat > 0) & np.isfinite(depth_flat)
            pts_valid = pts_flat[mask]
            col_valid = col_flat[mask]
            norm_valid = norm_flat[mask]

            # 4. 降采样 (可选)
            max_points = 200_000
            if len(pts_valid) > max_points:
                indices = np.random.choice(len(pts_valid), max_points, replace=False)
                pts_valid = pts_valid[indices]
                col_valid = col_valid[indices]
                norm_valid = norm_valid[indices]

            # 5. [新增] 坐标系旋转 (Camera -> World Z-up)
            # 为了适配 Sonata，将相机坐标系 (Y向下, Z向前) 转换为世界坐标系 (Z向上)
            # 变换：New Y = Old Z (前), New Z = -Old Y (上)
            pts_rotated = pts_valid.copy()
            pts_rotated[:, 1] = pts_valid[:, 2]
            pts_rotated[:, 2] = -pts_valid[:, 1]

            norm_rotated = norm_valid.copy()
            norm_rotated[:, 1] = norm_valid[:, 2]
            norm_rotated[:, 2] = -norm_valid[:, 1]

            # 6. 保存
            save_path = os.path.join(ply_out_dir, f"frame_{i:04d}.ply")
            save_ply_with_normals(save_path, pts_rotated, col_valid, norm_rotated)
            
        print(f"Custom PLYs saved to {ply_out_dir}")

if __name__ == "__main__":
    main()