# tools/run_vggt4d_robust.py
"""
Usage:
python tools/run_vggt4d_robust.py \
    --image-dir ./demo/kling \
    --output-dir ./analysis/vis_results_robust \
    --model-path /opt/data/private/models/depthanything3/DA3NESTED-GIANT-LARGE \
    --window-size 2
"""
import argparse
import types
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import glob
import gc
import sys
from PIL import Image
from tqdm import tqdm

# 移除 scipy 依赖，改用 numpy 实现
# from scipy.ndimage import gaussian_filter1d 

from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.dinov2.layers.attention import Attention

# =========================================================================
# Part 1: 手写工具函数 (去依赖)
# =========================================================================
def temporal_smooth_numpy(data, sigma=1.0):
    """
    用 Numpy 实现简单的时序高斯平滑 (替代 scipy)
    data: (T, H, W)
    """
    T = data.shape[0]
    # 生成高斯核
    radius = int(4 * sigma + 0.5)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    
    # 对 T 维度进行卷积
    # 为了效率，我们只对非零区域做，或者直接循环
    out = np.zeros_like(data)
    for t in range(T):
        # 简单加权平均
        val = 0
        weight_sum = 0
        for k in range(-radius, radius + 1):
            idx = t + k
            if 0 <= idx < T:
                w = kernel[k + radius]
                val += data[idx] * w
                weight_sum += w
        out[t] = val / (weight_sum + 1e-6)
    return out

def guided_filter(I, p, r, eps):
    # 标准 Guided Filter
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

def normalize_map(data):
    flat = data.flatten()
    lower, upper = np.percentile(flat, 2), np.percentile(flat, 98)
    return np.clip((data - lower) / (upper - lower + 1e-6), 0, 1)

# =========================================================================
# Part 2: Monkey Patch
# =========================================================================
def fused_attention_forward(self, x, pos=None, attn_mask=None):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q, k = self.q_norm(q), self.k_norm(k)

    if self.rope is not None and pos is not None:
        q = self.rope(q, pos)
        k = self.rope(k, pos)

    # Mode A: Collect Stats
    if getattr(self, "collect_stats", False):
        with torch.no_grad():
            scale = self.scale
            num_frames = self.num_frames
            window = getattr(self, "window_size", 2)
            
            # 适配 Local (B=Frames) 或 Global (B=1)
            if B == num_frames and B > 1:
                q_in = q.permute(1, 0, 2, 3).reshape(1, self.num_heads, -1, C // self.num_heads)
                k_in = k.permute(1, 0, 2, 3).reshape(1, self.num_heads, -1, C // self.num_heads)
                total_tokens = B * N
            else:
                q_in, k_in = q, k
                total_tokens = N
            
            tokens_per_frame = total_tokens // num_frames
            
            # 初始化 (fp32 避免溢出)
            stats = {
                "qk_mean": torch.zeros(1, self.num_heads, total_tokens, device=q.device, dtype=torch.float32),
                "qk_var":  torch.zeros(1, self.num_heads, total_tokens, device=q.device, dtype=torch.float32),
                "qq_mean": torch.zeros(1, self.num_heads, total_tokens, device=q.device, dtype=torch.float32),
                "qq_var":  torch.zeros(1, self.num_heads, total_tokens, device=q.device, dtype=torch.float32),
                "kk_mean": torch.zeros(1, self.num_heads, total_tokens, device=q.device, dtype=torch.float32),
            }
            
            q_t = q_in.transpose(-2, -1)
            k_t = k_in.transpose(-2, -1)

            for t in range(num_frames):
                t_start, t_end = t * tokens_per_frame, (t+1) * tokens_per_frame
                w_start = max(0, t - window) * tokens_per_frame
                w_end = min(num_frames, t + window + 1) * tokens_per_frame
                
                q_curr = q_in[:, :, t_start:t_end]
                k_curr = k_in[:, :, t_start:t_end]
                q_win = q_t[:, :, :, w_start:w_end]
                k_win = k_t[:, :, :, w_start:w_end]

                # QK
                attn_qk = q_curr @ k_win * scale
                stats["qk_mean"][:, :, t_start:t_end] = attn_qk.mean(dim=-1)
                stats["qk_var"][:, :, t_start:t_end]  = attn_qk.var(dim=-1)
                # QQ
                gram_qq = q_curr @ q_win * scale
                stats["qq_mean"][:, :, t_start:t_end] = gram_qq.mean(dim=-1)
                stats["qq_var"][:, :, t_start:t_end]  = gram_qq.var(dim=-1)
                # KK
                gram_kk = k_curr @ k_win * scale
                stats["kk_mean"][:, :, t_start:t_end] = gram_kk.mean(dim=-1)

            self._captured_stats = {k: v.mean(dim=1).cpu() for k, v in stats.items()}
            del stats, q_t, k_t, q_in, k_in

    # Mode B: Inject Mask
    is_mask_injected = False
    if getattr(self, "apply_masking", False) and hasattr(self, "dynamic_mask"):
        mask_map = self.dynamic_mask.to(x.device)
        S, N_p = mask_map.shape
        
        # 修正 Mask 维度逻辑
        if B == S: # Local
            bias = torch.zeros(S, 1, 1, N_p, device=x.device)
            bias.masked_fill_(mask_map.view(S, 1, 1, N_p).bool(), float("-inf"))
        else: # Global
            bias = torch.zeros(1, 1, 1, S*N_p, device=x.device)
            bias.masked_fill_(mask_map.view(1, 1, 1, S*N_p).bool(), float("-inf"))
            
        if attn_mask is None: attn_mask = bias
        else: attn_mask = attn_mask + bias
        is_mask_injected = True

    if self.fused_attn:
        if not is_mask_injected and attn_mask is not None:
             attn_mask = attn_mask[:, None].repeat(1, self.num_heads, 1, 1)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=attn_mask)
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if attn_mask is not None: attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

def apply_patch(model, num_frames, window_size=2):
    for module in model.modules():
        if isinstance(module, Attention):
            module.forward = types.MethodType(fused_attention_forward, module)
            module.collect_stats = False
            module.apply_masking = False
            module.num_frames = num_frames
            module.window_size = window_size
            module._captured_stats = {}

# =========================================================================
# Part 3: 主流程
# =========================================================================
@torch.no_grad()
def main(args):
    print(f"--> Loading Model: {args.model_name}")
    load_source = args.model_path if args.model_path else f"depth-anything/{args.model_name.upper()}"
    model = DepthAnything3.from_pretrained(load_source).to(args.device)
    model.eval()
    
    print(f"--> Loading Images from {args.image_dir}")
    img_files = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")) + glob.glob(os.path.join(args.image_dir, "*.png")))
    if len(img_files) == 0:
        print("Error: No images found!")
        return
    pil_imgs = [Image.open(p).convert("RGB") for p in img_files]
    num_frames = len(pil_imgs)
    print(f"    Loaded {num_frames} frames.")
    
    apply_patch(model.model, num_frames, window_size=args.window_size)
    
    # --- Pass 1 ---
    print("--> [Pass 1] Running Inference & Extracting Stats...")
    for m in model.model.modules():
        if isinstance(m, Attention): m.collect_stats = True
            
    results_p1 = model.inference(pil_imgs, infer_gs=False)
    
    # 强制清理显存
    torch.cuda.empty_cache()
    gc.collect()
    
    # --- Mask Generation ---
    print("--> Computing Masks...")
    if hasattr(model.model, 'da3'): vit = model.model.da3.backbone.pretrained 
    else: vit = model.model.backbone.pretrained
    
    H_feat, W_feat = results_p1.depth.shape[1] // 14, results_p1.depth.shape[2] // 14
    accum_gram, accum_qk = 0, 0
    
    for idx, block in enumerate(vit.blocks):
        if not block.attn._captured_stats: continue
        stats = block.attn._captured_stats
        
        def get_map(name):
            d = stats[name].view(num_frames, -1)
            if d.shape[1] == (H_feat * W_feat + 1): d = d[:, 1:]
            d = d.view(num_frames, H_feat, W_feat)
            return F.interpolate(d.unsqueeze(1), size=results_p1.depth.shape[1:], mode='bilinear').squeeze(1).to(args.device)

        if idx in [1, 9, 21]: accum_qk += get_map("qk_mean")
        if idx == 1: accum_gram += (1 - get_map("kk_mean")) * get_map("qk_var")
        if 6 <= idx <= 13: accum_gram += (1 - get_map("qq_mean"))
        if 30 <= idx <= 37: accum_gram += (1 - (1 - get_map("qq_var")) * get_map("qq_mean"))
        
        # 释放每层的统计量
        block.attn._captured_stats = {}

    # Normalize
    mask_gram = normalize_map(accum_gram.cpu().numpy())
    mask_std = 1.0 - normalize_map(accum_qk.cpu().numpy())
    mask_coarse = (mask_gram + mask_std) / 2.0
    
    # Temporal Smoothing (Numpy implementation)
    print("--> Applying Temporal Smoothing (Numpy)...")
    mask_smooth = temporal_smooth_numpy(mask_coarse, sigma=1.0)
    
    # Spatial Refinement
    print("--> Applying Guided Filter...")
    refined_masks = []
    H, W = results_p1.depth.shape[1], results_p1.depth.shape[2]
    
    for t in tqdm(range(num_frames)):
        rgb = cv2.imread(img_files[t])
        rgb = cv2.resize(rgb, (W, H))
        I = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        refined = guided_filter(I, mask_smooth[t], r=int(W*0.02), eps=1e-4)
        refined = np.clip(refined, 0, 1)
        refined_masks.append(refined > 0.4) # Binarize
        
    refined_masks_np = np.stack(refined_masks).astype(np.float32)
    
    # --- Pass 2 ---
    print("--> [Pass 2] Masked Inference...")
    mask_tensor = torch.from_numpy(refined_masks_np).to(args.device)
    mask_small = F.interpolate(mask_tensor.unsqueeze(1), size=(H_feat, W_feat), mode='nearest').view(num_frames, -1)
    cls_mask = torch.zeros(num_frames, 1, device=args.device)
    mask_tokens = torch.cat([cls_mask, mask_small], dim=1)
    
    for idx, block in enumerate(vit.blocks):
        block.attn.collect_stats = False
        if 1 <= idx <= 8:
            block.attn.apply_masking = True
            block.attn.dynamic_mask = mask_tokens
        else:
            block.attn.apply_masking = False
            
    results_p2 = model.inference(pil_imgs, infer_gs=False)
    
    # --- Saving ---
    print(f"--> Saving Results to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    for t in range(num_frames):
        # Save Mask
        cv2.imwrite(os.path.join(args.output_dir, f"mask_{t:03d}.png"), (refined_masks_np[t]*255).astype(np.uint8))
        
        # Save Depth Compare
        d1 = normalize_map(results_p1.depth[t])
        d2 = normalize_map(results_p2.depth[t])
        vis = np.concatenate([d1, d2], axis=1)
        cv2.imwrite(os.path.join(args.output_dir, f"depth_compare_{t:03d}.png"), (vis*255).astype(np.uint8))
        
    print("--> All Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="da3nested-giant-large")
    parser.add_argument("--model-path", default="", help="Local model path")
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args)