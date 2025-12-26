import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 确保能引用 training 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.loss import DINOMetricLoss, ssim
from training.model import FreeTimeGSModel
from training.dataset import IntegratedVideoDataset
from src.depth_anything_3.model.utils.gs_renderer import render_3dgs

def train():
    # ================= 配置区 =================
    # 视频帧目录
    VIDEO_DIR = "/opt/data/private/datasets/davis_2016/DAVIS_2016/JPEGImages/1080p/bear" 
    
    # 权重路径
    DA3_PATH = "/opt/data/private/models/depthanything3/DA3-GIANT" 
    CONCERTO_PATH = "/opt/data/private/models/concerto/concerto_large.pth"
    
    # DINO路径: 可以是 'facebook/dinov2-base' 也可以是您本地的 HF 文件夹路径
    # 如果您下载了 HF 版本到本地，请填入本地文件夹路径
    DINO_PATH = "/opt/data/private/models/dinov2-base" 
    
    OUTPUT_DIR = "./checkpoints/bear_result"
    DEVICE = "cuda"
    EPOCHS = 100
    LR = 1e-3
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 初始化全自动数据集
    print("--- 1. Pipeline Initialization ---")
    dataset = IntegratedVideoDataset(
        video_dir=VIDEO_DIR,
        da3_model_path=DA3_PATH,
        concerto_model_path=CONCERTO_PATH,
        dino_model_path=DINO_PATH,
        voxel_size=0.02, # 可调节精度
        device=DEVICE
    )
    
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 2. 初始化模型
    print("--- 2. Model Initialization ---")
    token_dim = dataset.scene_tokens.shape[-1]
    model = FreeTimeGSModel(input_dim=token_dim).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    # 3. 初始化 Critic (用于提取渲染图特征)
    # GT 特征已经在 Dataset 里准备好了，所以这里主要用于渲染图
    print(f"--- 3. Initializing Critic with: {DINO_PATH} ---")
    critic = DINOMetricLoss(model_path=DINO_PATH, device=DEVICE)
    
    # 4. 训练
    print("--- 4. Training Loop ---")
    model.train()
    
    for epoch in range(1, EPOCHS + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            tokens = batch["tokens"].to(DEVICE)
            coords = batch["coords"].to(DEVICE)
            t = batch["t"].to(DEVICE)
            gt_image = batch["gt_image"].to(DEVICE)
            gt_feat = batch["gt_feat"].to(DEVICE) # [1, 768] (GT 特征已预存)
            c2w = batch["c2w"].to(DEVICE)
            K = batch["K"].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward
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
            
            # Loss
            loss_l1 = (render_out - gt_image).abs().mean()
            loss_ssim = 1.0 - ssim(render_out, gt_image)
            # Critic 只需要算渲染图特征，GT 特征直接传进去
            loss_feat = critic(render_out, gt_feats=gt_feat) 
            
            loss = loss_l1 + 0.2 * loss_ssim + 0.1 * loss_feat
            
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({"L1": f"{loss_l1.item():.3f}", "DINO": f"{loss_feat.item():.3f}"})
            
        if epoch % 50 == 0 or epoch == EPOCHS:
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/final_model.pth")
            
    print("Done! Model saved.")

if __name__ == "__main__":
    train()