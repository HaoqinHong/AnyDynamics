import sys
import os
import torch
import numpy as np
import imageio
from tqdm import tqdm
import struct

# Path Fix
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path: sys.path.insert(0, project_root)
CONCERTO_ROOT = "/opt/data/private/Ours-Projects/Physics-Simulator-World-Model/AnyDynamics/submodules/Concerto"
if CONCERTO_ROOT not in sys.path: sys.path.insert(0, CONCERTO_ROOT)

from training.model import FreeTimeGSModel
from training.dataset import IntegratedVideoDataset
from depth_anything_3.model.utils.gs_renderer import render_3dgs

def save_ply(path, means, scales, rotations, opacities, shs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    xyz = means.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = shs.detach().cpu().numpy().reshape(-1, 3)
    
    # Opacity (Sigmoid -> Logit)
    opac = opacities.detach().cpu().numpy().reshape(-1, 1)
    opac = np.clip(opac, 1e-6, 1 - 1e-6)
    opac = np.log(opac / (1 - opac))
    
    # Scale (Sigmoid -> Logit, 假设 viewer 会做 sigmoid)
    # 注意：这里有点 trick，标准 3DGS 存的是 log_scale，渲染时用 exp。
    # 我们模型输出的是 sigmoid 后的值。为了适配标准 Viewer，我们需要反推一个值 x
    # 使得 exp(x) ≈ sigmoid(model_out)。所以 x = log(sigmoid(model_out))
    # 但我们模型里 final_scale = min + (max-min)*sigmoid. 
    # 为了简单可视化，直接存 log(final_scale) 即可。
    scale = scales.detach().cpu().numpy()
    scale = np.log(np.clip(scale, 1e-8, 1e8))
    
    rot = rotations.detach().cpu().numpy()

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
             ('opacity', 'f4'),
             ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
             ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')]
    
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements['x'] = xyz[:, 0]; elements['y'] = xyz[:, 1]; elements['z'] = xyz[:, 2]
    elements['nx'] = normals[:, 0]; elements['ny'] = normals[:, 1]; elements['nz'] = normals[:, 2]
    elements['f_dc_0'] = f_dc[:, 0]; elements['f_dc_1'] = f_dc[:, 1]; elements['f_dc_2'] = f_dc[:, 2]
    elements['opacity'] = opac[:, 0]
    elements['scale_0'] = scale[:, 0]; elements['scale_1'] = scale[:, 1]; elements['scale_2'] = scale[:, 2]
    elements['rot_0'] = rot[:, 0]; elements['rot_1'] = rot[:, 1]; elements['rot_2'] = rot[:, 2]; elements['rot_3'] = rot[:, 3]

    with open(path, 'wb') as f:
        f.write(b"ply\n"); f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n".encode())
        f.write(b"property float x\n"); f.write(b"property float y\n"); f.write(b"property float z\n")
        f.write(b"property float nx\n"); f.write(b"property float ny\n"); f.write(b"property float nz\n")
        f.write(b"property float f_dc_0\n"); f.write(b"property float f_dc_1\n"); f.write(b"property float f_dc_2\n")
        f.write(b"property float opacity\n")
        f.write(b"property float scale_0\n"); f.write(b"property float scale_1\n"); f.write(b"property float scale_2\n")
        f.write(b"property float rot_0\n"); f.write(b"property float rot_1\n"); f.write(b"property float rot_2\n"); f.write(b"property float rot_3\n")
        f.write(b"end_header\n")
        elements.tofile(f)

def render_video():
    VIDEO_DIR = "/opt/data/private/datasets/davis_2016/DAVIS_2016/JPEGImages/1080p/bear" 
    DA3_PATH = "/opt/data/private/models/depthanything3/DA3-GIANT" 
    CONCERTO_PATH = "/opt/data/private/models/concerto/concerto_large.pth"
    DINO_PATH = "/opt/data/private/models/dinov2-base"
    CHECKPOINT_PATH = "./checkpoints/bear_result/final_model.pth"
    OUTPUT_ROOT = "./outputs"
    DEVICE = "cuda"
    
    print("--- 1. Re-loading Data ---")
    dataset = IntegratedVideoDataset(VIDEO_DIR, DA3_PATH, CONCERTO_PATH, DINO_PATH, 0.02, DEVICE)
    
    print(f"--- 2. Loading Model ---")
    model = FreeTimeGSModel(dataset.scene_tokens.shape[-1]).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    
    print("--- 3. Rendering ---")
    frames = []
    os.makedirs(f"{OUTPUT_ROOT}/ply_sequence", exist_ok=True)
    
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        tokens = data["tokens"].unsqueeze(0).to(DEVICE)
        coords = data["coords"].unsqueeze(0).to(DEVICE)
        t = data["t"].unsqueeze(0).to(DEVICE)
        c2w = data["c2w"].unsqueeze(0).to(DEVICE)
        K = data["K"].unsqueeze(0).to(DEVICE)
        w2c = torch.linalg.inv(c2w)
        _, H, W = data["gt_image"].shape
        K_norm = K.clone(); K_norm[..., 0, :] /= W; K_norm[..., 1, :] /= H
        
        with torch.no_grad():
            gaussians = model(tokens, coords, t)
            
            # Save PLY
            save_ply(f"{OUTPUT_ROOT}/ply_sequence/frame_{i:03d}.ply", 
                     gaussians.means[0], gaussians.scales[0], gaussians.rotations[0], 
                     gaussians.opacities[0], gaussians.harmonics[0])
            
            render_out, _ = render_3dgs(w2c, K_norm, (H, W), gaussians, 1, torch.zeros(1, 3).to(DEVICE))
            
            rgb = render_out.squeeze(1).squeeze(0).permute(1, 2, 0).cpu().numpy()
            frames.append(np.clip(rgb * 255, 0, 255).astype(np.uint8))
            
    imageio.mimwrite(f"{OUTPUT_ROOT}/bear_output.mp4", frames, fps=24, quality=8)
    print("Done!")

if __name__ == "__main__":
    render_video()