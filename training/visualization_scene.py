import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==============================================================================
# 1. 核心修复：强制解决路径冲突
# ==============================================================================
# 获取当前脚本所在目录 (.../AnyDynamics/training)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (.../AnyDynamics)
project_root = os.path.dirname(current_dir)

# 强制插入到第 0 位，确保 import training 优先使用本地文件夹
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 适配 Concerto 路径 (防止 dataset 内部报错)
CONCERTO_ROOT = "/opt/data/private/Ours-Projects/Physics-Simulator-World-Model/AnyDynamics/submodules/Concerto"
if CONCERTO_ROOT not in sys.path:
    sys.path.insert(0, CONCERTO_ROOT)

# ==============================================================================
# 2. 正常导入
# ==============================================================================
from training.dataset import IntegratedVideoDataset

def plot_camera(ax, c2w, color='blue', scale=0.1):
    """画一个相机金字塔"""
    c2w = c2w.cpu().numpy()
    center = c2w[:3, 3]
    R = c2w[:3, :3]
    
    axes_len = scale * 2
    
    # X(红), Y(绿), Z(蓝)
    x_end = center + R @ np.array([axes_len, 0, 0])
    ax.plot([center[0], x_end[0]], [center[1], x_end[1]], [center[2], x_end[2]], color='red')
    
    y_end = center + R @ np.array([0, axes_len, 0])
    ax.plot([center[0], y_end[0]], [center[1], y_end[1]], [center[2], y_end[2]], color='green')
    
    z_end = center + R @ np.array([0, 0, axes_len])
    ax.plot([center[0], z_end[0]], [center[1], z_end[1]], [center[2], z_end[2]], color='blue')

def visualize():
    # === 配置区 ===
    VIDEO_DIR = "/opt/data/private/datasets/davis_2016/DAVIS_2016/JPEGImages/1080p/bear"
    DA3_PATH = "/opt/data/private/models/depthanything3/DA3-GIANT"
    CONCERTO_PATH = "/opt/data/private/models/concerto/concerto_large.pth"
    DINO_PATH = "/opt/data/private/models/dinov2-base"
    VOXEL_SIZE = 0.02
    
    print("--- 正在加载数据集 ---")
    dataset = IntegratedVideoDataset(
        video_dir=VIDEO_DIR,
        da3_model_path=DA3_PATH,
        concerto_model_path=CONCERTO_PATH,
        dino_model_path=DINO_PATH,
        voxel_size=VOXEL_SIZE,
        device='cuda'
    )
    
    print("\n--- 诊断报告 ---")
    
    # 1. 检查数据对齐情况
    data = dataset[0]
    coords = data["coords"]
    
    coords_np = coords.cpu().numpy()
    print(f"Token 数量: {data['tokens'].shape}")
    print(f"坐标点 数量: {coords_np.shape}")
    print(f"点云范围 (Coords Range): Min {coords_np.min(0)}, Max {coords_np.max(0)}")
    print(f"点云中心 (Coords Mean): {coords_np.mean(0)}")
    
    # 2. 检查相机
    c2w_list = [dataset[i]["c2w"] for i in range(len(dataset))]
    c2ws = torch.stack(c2w_list)
    c2ws_np = c2ws[:, :3, 3].cpu().numpy()
    
    print(f"相机中心范围 (Camera Centers): Min {c2ws_np.min(0)}, Max {c2ws_np.max(0)}")
    
    # 3. 绘图
    print("\n--- 正在绘制场景 (保存为 scene_alignment.png) ---")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 画点云
    if len(coords_np) > 5000:
        idx = np.random.choice(len(coords_np), 5000, replace=False)
        pts = coords_np[idx]
    else:
        pts = coords_np
        
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c='gray', alpha=0.5, label='Scene Points')
    
    # 画相机
    indices = [0, len(dataset)//2, len(dataset)-1]
    for i in indices:
        plot_camera(ax, dataset[i]["c2w"], scale=0.2)
        
    ax.plot(c2ws_np[:, 0], c2ws_np[:, 1], c2ws_np[:, 2], c='purple', label='Camera Trajectory')
    
    # 设置显示范围
    all_vals = np.concatenate([pts, c2ws_np])
    min_v, max_v = all_vals.min(), all_vals.max()
    ax.set_xlim(min_v, max_v); ax.set_ylim(min_v, max_v); ax.set_zlim(min_v, max_v)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(f"Check: Points [{coords_np.min():.2f}, {coords_np.max():.2f}]")
    
    plt.savefig("scene_alignment.png")
    print("完成! 请查看生成的 'scene_alignment.png' 和上方日志。")

if __name__ == "__main__":
    visualize()