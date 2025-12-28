import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# ================= 路径配置 =================
# 1. 添加项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. 添加 src 目录 (确保可以直接 import depth_anything_3)
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# ================= 模块导入 =================
from training.loss import DINOMetricLoss, ssim
from training.model import FreeTimeGSModel
from training.dataset import IntegratedVideoDataset

# [修复] 正确的导入路径: 只要到 gs_renderer 文件名为止
from depth_anything_3.model.utils.gs_renderer import render_3dgs

def train():
    # === 配置区 ===
    VIDEO_DIR = "/opt/data/private/datasets/davis_2016/DAVIS_2016/JPEGImages/1080p/bear" 
    DA3_PATH = "/opt/data/private/models/depthanything3/DA3-GIANT" 
    CONCERTO_PATH = "/opt/data/private/models/concerto/concerto_large.pth"
    DINO_PATH = "/opt/data/private/models/dinov2-base" 
    OUTPUT_DIR = "./checkpoints/bear_result"
    
    DEVICE = "cuda"
    EPOCHS = 100
    LR = 1e-3
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Dataset
    print("--- 1. Pipeline Initialization ---")
    dataset = IntegratedVideoDataset(
        video_dir=VIDEO_DIR, da3_model_path=DA3_PATH, concerto_model_path=CONCERTO_PATH,
        dino_model_path=DINO_PATH, voxel_size=0.02, device=DEVICE
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 2. Model
    print("--- 2. Model Initialization ---")
    token_dim = dataset.scene_tokens.shape[-1]
    model = FreeTimeGSModel(input_dim=token_dim).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    # 3. Critic
    print(f"--- 3. Initializing Critic ---")
    critic = DINOMetricLoss(model_path=DINO_PATH, device=DEVICE)
    
    # 4. Train
    print("--- 4. Training Loop ---")
    model.train()
    
    for epoch in range(1, EPOCHS + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            tokens = batch["tokens"].to(DEVICE)
            coords = batch["coords"].to(DEVICE)
            t = batch["t"].to(DEVICE)
            gt_image = batch["gt_image"].to(DEVICE)
            gt_feat = batch["gt_feat"].to(DEVICE)
            c2w = batch["c2w"].to(DEVICE)
            K = batch["K"].to(DEVICE)
            
            optimizer.zero_grad()
            
            gaussians = model(tokens, coords, t)
            
            # Render
            w2c = torch.linalg.inv(c2w)
            _, _, H, W = gt_image.shape
            K_norm = K.clone(); K_norm[...,0,:]/=W; K_norm[...,1,:]/=H
            
            render_out, _ = render_3dgs(
                extrinsics=w2c, intrinsics=K_norm, image_shape=(H,W),
                gaussian=gaussians, num_view=1,
                background_color=torch.zeros(1,3).to(DEVICE)
            )
            render_out = render_out.squeeze(1)
            
            # === Loss Calculation ===
            loss_l1 = (render_out - gt_image).abs().mean()
            val_ssim = ssim(render_out, gt_image)
            loss_ssim = 1.0 - val_ssim
            loss_feat = critic(render_out, gt_feats=gt_feat)
            
            # 正则化
            reg_scale = gaussians.scales.mean()
            reg_opac = gaussians.opacities.mean()
            
            # [策略] 降低正则化权重，优先保证物体可见性
            loss = 1.0 * loss_l1 + 0.2 * loss_ssim + 0.1 * loss_feat + \
                   0.0001 * reg_scale + 0.0001 * reg_opac
            
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({
                "L1": f"{loss_l1.item():.3f}", 
                "SSIM": f"{val_ssim.item():.3f}",
                "DINO": f"{loss_feat.item():.3f}"
            })
            
        if epoch % 50 == 0 or epoch == EPOCHS:
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/final_model.pth")
            
    print("Done! Model saved.")

if __name__ == "__main__":
    train()