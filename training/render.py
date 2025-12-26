import sys
import os
import torch
import numpy as np
import imageio
from tqdm import tqdm
import struct

# ==============================================================================
# 1. 路径修复 (防止命名冲突)
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 确保 Concerto 路径也在 (如果 dataset.py 里没配好)
CONCERTO_ROOT = "/opt/data/private/Ours-Projects/Physics-Simulator-World-Model/AnyDynamics/submodules/Concerto"
if CONCERTO_ROOT not in sys.path:
    sys.path.insert(0, CONCERTO_ROOT)

# ==============================================================================
# 2. 导入模块
# ==============================================================================
from training.model import FreeTimeGSModel
from training.dataset import IntegratedVideoDataset
from depth_anything_3.model.utils.gs_renderer import render_3dgs

# ==============================================================================
# 3. PLY 保存辅助函数 (标准 3DGS 格式)
# ==============================================================================
def save_ply(path, means, scales, rotations, opacities, shs):
    """
    将高斯保存为标准 PLY 格式。
    注意：标准查看器通常期望 Scale 是 Log 空间，Opacity 是 Logit 空间。
    """
    mkdir_p(os.path.dirname(path))
    
    xyz = means.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    
    # 颜色 (f_dc): [N, 3, 1] -> [N, 3]
    f_dc = shs.detach().cpu().numpy().reshape(-1, 3)
    
    # Opacity: Model 输出的是 Sigmoid 后的，PLY 需要 Logit
    # logit(p) = log(p / (1 - p))
    opac = opacities.detach().cpu().numpy().reshape(-1, 1)
    opac = np.clip(opac, 1e-6, 1 - 1e-6)
    opac = np.log(opac / (1 - opac))
    
    # Scale: Model 输出的是 Exp 后的，PLY 需要 Log
    # log(exp(s)) = s
    scale = scales.detach().cpu().numpy()
    scale = np.log(np.clip(scale, 1e-6, 1e8))
    
    # Rotation: Quaternion [N, 4]
    rot = rotations.detach().cpu().numpy()

    # 构建结构化数组
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
             ('opacity', 'f4'),
             ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
             ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')]
    
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['nx'] = normals[:, 0]
    elements['ny'] = normals[:, 1]
    elements['nz'] = normals[:, 2]
    elements['f_dc_0'] = f_dc[:, 0]
    elements['f_dc_1'] = f_dc[:, 1]
    elements['f_dc_2'] = f_dc[:, 2]
    elements['opacity'] = opac[:, 0]
    elements['scale_0'] = scale[:, 0]
    elements['scale_1'] = scale[:, 1]
    elements['scale_2'] = scale[:, 2]
    elements['rot_0'] = rot[:, 0]
    elements['rot_1'] = rot[:, 1]
    elements['rot_2'] = rot[:, 2]
    elements['rot_3'] = rot[:, 3]

    # 写入文件
    with open(path, 'wb') as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n".encode())
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        f.write(b"property float nx\n")
        f.write(b"property float ny\n")
        f.write(b"property float nz\n")
        f.write(b"property float f_dc_0\n")
        f.write(b"property float f_dc_1\n")
        f.write(b"property float f_dc_2\n")
        f.write(b"property float opacity\n")
        f.write(b"property float scale_0\n")
        f.write(b"property float scale_1\n")
        f.write(b"property float scale_2\n")
        f.write(b"property float rot_0\n")
        f.write(b"property float rot_1\n")
        f.write(b"property float rot_2\n")
        f.write(b"property float rot_3\n")
        f.write(b"end_header\n")
        elements.tofile(f)

def mkdir_p(path):
    os.makedirs(path, exist_ok=True)

# ==============================================================================
# 4. 主渲染逻辑
# ==============================================================================
def render_video():
    # --- 配置 ---
    VIDEO_DIR = "/opt/data/private/datasets/davis_2016/DAVIS_2016/JPEGImages/1080p/bear" 
    DA3_PATH = "/opt/data/private/models/depthanything3/DA3-GIANT" 
    CONCERTO_PATH = "/opt/data/private/models/concerto/concerto_large.pth"
    DINO_PATH = "/opt/data/private/models/dinov2-base"
    
    CHECKPOINT_PATH = "./checkpoints/bear_result/final_model.pth"
    
    # 输出路径配置
    OUTPUT_ROOT = "./outputs"
    OUTPUT_VIDEO = os.path.join(OUTPUT_ROOT, "bear_output.mp4")
    OUTPUT_PLY_DIR = os.path.join(OUTPUT_ROOT, "ply_sequence")
    
    DEVICE = "cuda"
    
    # 1. 准备数据
    print("--- 1. Re-loading Data ---")
    dataset = IntegratedVideoDataset(
        video_dir=VIDEO_DIR,
        da3_model_path=DA3_PATH,
        concerto_model_path=CONCERTO_PATH,
        dino_model_path=DINO_PATH,
        voxel_size=0.02, 
        device=DEVICE
    )
    
    # 2. 加载模型
    print(f"--- 2. Loading Model from {CHECKPOINT_PATH} ---")
    token_dim = dataset.scene_tokens.shape[-1]
    model = FreeTimeGSModel(input_dim=token_dim).to(DEVICE)
    
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
        
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 创建输出目录
    mkdir_p(OUTPUT_ROOT)
    mkdir_p(OUTPUT_PLY_DIR)
    
    # 3. 渲染循环
    print("--- 3. Rendering & Saving PLYs ---")
    frames = []
    
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        
        # 准备输入
        tokens = data["tokens"].unsqueeze(0).to(DEVICE)
        coords = data["coords"].unsqueeze(0).to(DEVICE)
        t = data["t"].unsqueeze(0).to(DEVICE)
        
        c2w = data["c2w"].unsqueeze(0).to(DEVICE)
        K = data["K"].unsqueeze(0).to(DEVICE)
        w2c = torch.linalg.inv(c2w)
        
        _, H, W = data["gt_image"].shape
        K_norm = K.clone()
        K_norm[..., 0, :] /= W
        K_norm[..., 1, :] /= H
        
        with torch.no_grad():
            # A. 预测高斯
            gaussians = model(tokens, coords, t)
            
            # B. 保存 PLY (每帧一个文件)
            # 格式: frame_000.ply
            ply_path = os.path.join(OUTPUT_PLY_DIR, f"frame_{i:03d}.ply")
            save_ply(
                ply_path, 
                gaussians.means[0], 
                gaussians.scales[0], 
                gaussians.rotations[0], 
                gaussians.opacities[0], 
                gaussians.harmonics[0]
            )
            
            # C. 渲染视频帧
            render_out, _ = render_3dgs(
                extrinsics=w2c,
                intrinsics=K_norm,
                image_shape=(H, W),
                gaussian=gaussians,
                num_view=1,
                background_color=torch.zeros(1, 3).to(DEVICE)
            )
            
            rgb = render_out.squeeze(1).squeeze(0)
            rgb = rgb.permute(1, 2, 0).cpu().numpy()
            rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
            frames.append(rgb)
            
    # 4. 保存视频
    print(f"--- 4. Saving video to {OUTPUT_VIDEO} ---")
    imageio.mimwrite(OUTPUT_VIDEO, frames, fps=24, quality=8)
    print(f"Done! Check your results in '{OUTPUT_ROOT}'")

if __name__ == "__main__":
    render_video()