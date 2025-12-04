# tools/run_vggt4d_guided.py
"""
Usage:
python tools/run_vggt4d_guided.py \
    --model-name da3nested-giant-large \
    --model-path /opt/data/private/models/depthanything3/DA3NESTED-GIANT-LARGE \
    --image-dir ./demo/kling \
    --output-dir ./analysis/vis_results_guided \
    --fps 10
"""
import argparse
import types
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import gc
import glob
from pathlib import Path
from PIL import Image

# DA3 Imports
from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.dinov2.layers.attention import Attention

# ==========================================
# PART 1: 特征提取 (保持不变)
# ==========================================
def custom_attention_forward_hybrid(self, x, pos=None, attn_mask=None):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q, k = self.q_norm(q), self.k_norm(k)

    if self.rope is not None and pos is not None:
        q = self.rope(q, pos)
        k = self.rope(k, pos)

    if getattr(self, "collect_stats", False):
        with torch.no_grad():
            scale = self.scale
            q_t = q.transpose(-2, -1)
            k_t = k.transpose(-2, -1)
            
            acc_m_qk = torch.zeros(B, self.num_heads, N, device=q.device, dtype=torch.float32)
            acc_m_qq = torch.zeros(B, self.num_heads, N, device=q.device, dtype=torch.float32)
            acc_v_qq = torch.zeros(B, self.num_heads, N, device=q.device, dtype=torch.float32)
            acc_m_kk = torch.zeros(B, self.num_heads, N, device=q.device, dtype=torch.float32)
            acc_v_qk = torch.zeros(B, self.num_heads, N, device=q.device, dtype=torch.float32)
            
            chunk_size = 1024 
            for i in range(0, N, chunk_size):
                end = min(i + chunk_size, N)
                q_chunk = q[:, :, i:end, :]
                k_chunk = k[:, :, i:end, :]
                
                attn_qk = torch.matmul(q_chunk, k_t) * scale
                acc_m_qk[:, :, i:end] = attn_qk.mean(dim=-1)
                acc_v_qk[:, :, i:end] = attn_qk.var(dim=-1)
                del attn_qk
                
                gram_qq = torch.matmul(q_chunk, q_t) * scale
                acc_m_qq[:, :, i:end] = gram_qq.mean(dim=-1)
                acc_v_qq[:, :, i:end] = gram_qq.var(dim=-1)
                del gram_qq
                
                gram_kk = torch.matmul(k_chunk, k_t) * scale
                acc_m_kk[:, :, i:end] = gram_kk.mean(dim=-1)
                del gram_kk

            self._captured_stats = {
                "qk_mean": acc_m_qk.mean(dim=1).cpu(),
                "v_qk":    acc_v_qk.mean(dim=1).cpu(),
                "s_qq":    acc_m_qq.mean(dim=1).cpu(),
                "v_qq":    acc_v_qq.mean(dim=1).cpu(),
                "s_kk":    acc_m_kk.mean(dim=1).cpu(),
            }

    if self.fused_attn:
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=((attn_mask)[:, None].repeat(1, self.num_heads, 1, 1) if attn_mask is not None else None))
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

def apply_monkey_patch(model):
    print("Applying Monkey Patch...")
    for module in model.modules():
        if isinstance(module, Attention):
            module.forward = types.MethodType(custom_attention_forward_hybrid, module)
            module.collect_stats = False
            module._captured_stats = {}

# ==========================================
# PART 2: [核心] Guided Filter 实现
# ==========================================
def infer_grid(total_tokens, num_frames, aspect_ratio):
    tokens_per_frame = total_tokens // num_frames
    for offset in range(17):
        valid = tokens_per_frame - offset
        if valid <= 0: continue
        h = int(np.round(np.sqrt(valid / aspect_ratio)))
        if h > 0 and valid % h == 0:
            w = valid // h
            if abs((w/h) - aspect_ratio)/aspect_ratio < 0.1:
                return (h, w), offset
    return None, 0

def normalize_and_reshape(raw_flat, num_frames, grid_size, offset):
    h, w = grid_size
    tokens_per_frame = raw_flat.shape[0] // num_frames
    reshaped = raw_flat.reshape(num_frames, tokens_per_frame)
    spatial = reshaped[:, offset:]
    flat = spatial.flatten()
    lower, upper = np.percentile(flat, 2), np.percentile(flat, 98)
    norm = (np.clip(spatial, lower, upper) - lower) / (upper - lower + 1e-6)
    return norm.reshape(num_frames, h, w)

def guided_filter(I, p, r, eps):
    """
    引导滤波算法 (Guided Filter)
    I: 引导图 (RGB, normalized 0-1)
    p: 输入图 (Coarse Mask, normalized 0-1)
    r: 滤波半径
    eps: 正则化参数
    """
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
    
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    
    q = mean_a * I + mean_b
    return q

def apply_guided_refinement(mask_lowres, rgb_img, W, H):
    """
    使用 Guided Filter 优化 Mask 边缘
    """
    # 1. 准备引导图 (RGB转灰度作为引导，或直接用单通道)
    # 缩放到目标尺寸
    rgb_resized = cv2.resize(rgb_img, (W, H))
    I = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    # 2. 准备输入 Mask (上采样)
    p = cv2.resize(mask_lowres, (W, H), interpolation=cv2.INTER_CUBIC)
    
    # 3. 运行 Guided Filter
    # r: 半径，越大越平滑 (通常设为图像尺寸的 1-2%)
    # eps: 阈值，越小越贴合边缘
    r = int(max(W, H) * 0.02)
    eps = 1e-4
    refined = guided_filter(I, p, r, eps)
    
    # 4. 后处理 (截断并二值化以获得清晰轮廓)
    refined = np.clip(refined, 0, 1)
    # 稍微加强对比度
    refined = (refined - refined.min()) / (refined.max() - refined.min() + 1e-6)
    
    return refined

def save_heatmap(data, path):
    img_uint8 = (data * 255).astype(np.uint8)
    color = cv2.applyColorMap(img_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(path, color)

def create_video_from_images(image_folder, video_output_path, suffix, fps=10):
    images = sorted(glob.glob(os.path.join(image_folder, f"*{suffix}")))
    if not images: return
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
    for image in images: video.write(cv2.imread(image))
    video.release()
    print(f"  [Video] Generated: {os.path.basename(video_output_path)}")

# ==========================================
# PART 3: 主程序
# ==========================================
def load_images(image_dir):
    files = sorted(list(Path(image_dir).glob("*.png")) + list(Path(image_dir).glob("*.jpg")))
    if not files: raise ValueError("No images found")
    return [Image.open(p).convert("RGB") for p in files], files

@torch.no_grad()
def main(args):
    # 1. 加载模型
    if args.model_path:
        print(f"Loading local model from: {args.model_path}...")
        load_source = args.model_path
    else:
        print(f"Loading HF model: depth-anything/{args.model_name.upper()}...")
        load_source = f"depth-anything/{args.model_name.upper()}"

    torch.cuda.empty_cache()
    gc.collect()
    
    model = DepthAnything3.from_pretrained(load_source).to(args.device)
    model.eval()
    apply_monkey_patch(model.model)
    
    pil_imgs, img_paths = load_images(args.image_dir)
    print(f"Loaded {len(pil_imgs)} images.")
    
    for module in model.model.modules():
        if isinstance(module, Attention): module.collect_stats = True
        
    print("Running inference...")
    _ = model.inference(pil_imgs, infer_gs=False)
    
    print("Processing results...")
    if hasattr(model.model, 'da3'): vit = model.model.da3.backbone.pretrained
    else: vit = model.model.backbone.pretrained

    root_dir = Path(args.output_dir)
    img_dir = root_dir / "images"
    vid_dir = root_dir / "videos"
    img_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(exist_ok=True)

    meta = {"H": pil_imgs[0].height, "W": pil_imgs[0].width, "num_frames": len(pil_imgs)}
    aspect_ratio = meta["W"] / meta["H"]
    
    selected_layers = [1, 9, 21]

    for idx, block in enumerate(vit.blocks):
        if idx not in selected_layers: 
            block.attn._captured_stats = {}
            continue
            
        if hasattr(block.attn, "_captured_stats") and block.attn._captured_stats:
            captured = block.attn._captured_stats
            key = f"layer_{idx:02d}"
            print(f"  Processing {key}...")
            
            raw_qk_mean = captured["qk_mean"].float().numpy().flatten()
            raw_s_qq = captured["s_qq"].float().numpy().flatten()
            raw_v_qq = captured["v_qq"].float().numpy().flatten()
            raw_s_kk = captured["s_kk"].float().numpy().flatten()
            raw_v_qk = captured["v_qk"].float().numpy().flatten()
            
            grid, offset = infer_grid(raw_qk_mean.shape[0], meta["num_frames"], aspect_ratio)
            if not grid: continue
            
            map_qk_mean = normalize_and_reshape(raw_qk_mean, meta["num_frames"], grid, offset)
            map_s_qq = normalize_and_reshape(raw_s_qq, meta["num_frames"], grid, offset)
            map_v_qq = normalize_and_reshape(raw_v_qq, meta["num_frames"], grid, offset)
            map_s_kk = normalize_and_reshape(raw_s_kk, meta["num_frames"], grid, offset)
            map_v_qk = normalize_and_reshape(raw_v_qk, meta["num_frames"], grid, offset)
            
            for t in range(meta["num_frames"]):
                base = str(img_dir / f"{key}_frame{t:03d}")
                
                # 读取当前帧的 RGB 原图
                rgb_img = cv2.imread(str(img_paths[t]))
                
                # 1. Avg Mask (基础)
                mask_std = 1.0 - map_qk_mean[t]
                w_shallow = (1.0 - map_s_kk[t]) * map_v_qk[t]
                w_middle = 1.0 - map_s_qq[t]
                w_deep_dyn = 1.0 - (map_s_qq[t] * (1.0 - map_v_qq[t]))
                mask_vggt = w_shallow * w_middle * w_deep_dyn
                
                mask_avg = (mask_std + mask_vggt) / 2.0
                save_heatmap(mask_avg, base + "_avg.png")
                
                # 2. Guided Filter Refinement (利用 RGB 优化边缘)
                mask_guided = apply_guided_refinement(mask_avg, rgb_img, meta["W"], meta["H"])
                save_heatmap(mask_guided, base + "_guided.png")

            block.attn._captured_stats = {}

    print("Generating videos...")
    for suffix in ["_avg.png", "_guided.png"]:
        for layer in selected_layers:
            key = f"layer_{layer:02d}"
            vid_name = f"{key}{suffix.replace('.png', '.mp4')}"
            create_video_from_images(str(img_dir), str(vid_dir / vid_name), f"{key}_*{suffix}", args.fps)

    print(f"\nAll Done! Results in {root_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="da3nested-giant-large")
    parser.add_argument("--model-path", default="", help="Local model path")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", default="./analysis/vis_results_guided")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    main(args)