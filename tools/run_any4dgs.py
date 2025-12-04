# tools/run_vggt4d_final_smooth.py
"""
Usage:
python tools/run_vggt4d_final_smooth.py \
    --image-dir ./demo/kling \
    --output-dir ./analysis/vis_results_smooth \
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
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d # 需要 scipy

from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.dinov2.layers.attention import Attention

# =========================================================================
# Part 1: 高级图像处理工具
# =========================================================================
def guided_filter(I, p, r, eps):
    """
    引导滤波: 利用 I (RGB) 的纹理边缘去修正 p (Mask)
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

def apply_hysteresis_threshold(prob_map, low=0.3, high=0.6):
    """
    双阈值滞后处理：
    1. 强阈值(high)确定的区域直接保留
    2. 弱阈值(low)区域，只有与强区域相连才保留
    这能有效保持物体完整性并去除背景噪点
    """
    # 1. 强弱区域
    strong_mask = (prob_map >= high).astype(np.uint8)
    weak_mask = ((prob_map >= low) & (prob_map < high)).astype(np.uint8)
    
    # 2. 连通域分析
    # 将 strong 作为种子，通过 weak 区域扩散
    # cv2.connectedComponents 只能处理二值，这里我们用形态学重建的思路
    # 或者简单点：用 connectedComponentsWithStats
    
    # 组合 mask
    all_candidates = strong_mask + weak_mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(all_candidates, connectivity=8)
    
    final_mask = np.zeros_like(prob_map)
    
    for i in range(1, num_labels): # 跳过背景 0
        # 获取该连通域的 mask
        component_mask = (labels == i)
        
        # 检查该连通域是否包含至少一个 strong pixel
        # 如果包含，则整个连通域（包括 weak 部分）都保留
        if np.any(strong_mask & component_mask):
            final_mask[component_mask] = 1.0
            
    return final_mask

def normalize_map(data):
    flat = data.flatten()
    lower, upper = np.percentile(flat, 2), np.percentile(flat, 98)
    return np.clip((data - lower) / (upper - lower + 1e-6), 0, 1)

# =========================================================================
# Part 2: 核心 Patch (保持 Fused Sliding 逻辑)
# =========================================================================
def fused_attention_forward(self, x, pos=None, attn_mask=None):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q, k = self.q_norm(q), self.k_norm(k)

    if self.rope is not None and pos is not None:
        q = self.rope(q, pos)
        k = self.rope(k, pos)

    # --- Mode A: 统计收集 ---
    if getattr(self, "collect_stats", False):
        with torch.no_grad():
            scale = self.scale
            num_frames = self.num_frames
            window = getattr(self, "window_size", 2)
            
            if B == num_frames and B > 1:
                q_in = q.permute(1, 0, 2, 3).reshape(1, self.num_heads, -1, C // self.num_heads)
                k_in = k.permute(1, 0, 2, 3).reshape(1, self.num_heads, -1, C // self.num_heads)
                total_tokens = B * N
            else:
                q_in, k_in = q, k
                total_tokens = N
            
            tokens_per_frame = total_tokens // num_frames
            
            stats = {
                "qk_mean": torch.zeros(1, self.num_heads, total_tokens, device=q.device),
                "qk_var":  torch.zeros(1, self.num_heads, total_tokens, device=q.device),
                "qq_mean": torch.zeros(1, self.num_heads, total_tokens, device=q.device),
                "qq_var":  torch.zeros(1, self.num_heads, total_tokens, device=q.device),
                "kk_mean": torch.zeros(1, self.num_heads, total_tokens, device=q.device),
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
            del stats, q_t, k_t

    # --- Mode B: Mask 注入 ---
    is_mask_injected = False
    if getattr(self, "apply_masking", False) and hasattr(self, "dynamic_mask"):
        mask_map = self.dynamic_mask.to(x.device)
        S, N_p = mask_map.shape
        if B == S: 
            bias = torch.zeros(S, 1, 1, N_p, device=x.device)
            bias.masked_fill_(mask_map.view(S, 1, 1, N_p).bool(), float("-inf"))
        else: 
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
    print(f"[1/4] Loading Model: {args.model_name}")
    load_source = args.model_path if args.model_path else f"depth-anything/{args.model_name.upper()}"
    model = DepthAnything3.from_pretrained(load_source).to(args.device)
    model.eval()
    
    img_files = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")) + glob.glob(os.path.join(args.image_dir, "*.png")))
    pil_imgs = [Image.open(p).convert("RGB") for p in img_files]
    num_frames = len(pil_imgs)
    print(f"Processing {num_frames} frames (Window={args.window_size})")
    
    apply_patch(model.model, num_frames, window_size=args.window_size)
    
    # --- Pass 1: 提取特征 ---
    print("[2/4] Pass 1: Extracting Cues...")
    for m in model.model.modules():
        if isinstance(m, Attention): m.collect_stats = True
            
    results_p1 = model.inference(pil_imgs, infer_gs=False)
    
    # --- Mask Generation (Sequence Level) ---
    print("[3/4] Fusing Masks & Temporal Smoothing...")
    vit = model.model.da3.backbone.pretrained if hasattr(model.model, 'da3') else model.model.backbone.pretrained
    H_feat, W_feat = results_p1.depth.shape[1] // 14, results_p1.depth.shape[2] // 14
    
    accum_shallow, accum_middle, accum_deep = 0, 0, 0
    accum_qk_mean = 0 
    
    for idx, block in enumerate(vit.blocks):
        if not block.attn._captured_stats: continue
        stats = block.attn._captured_stats
        def get_map(name):
            d = stats[name].view(num_frames, -1)
            if d.shape[1] == (H_feat * W_feat + 1): d = d[:, 1:]
            d = d.view(num_frames, H_feat, W_feat)
            return F.interpolate(d.unsqueeze(1), size=results_p1.depth.shape[1:], mode='bilinear').squeeze(1).to(args.device)

        if idx in [1, 9, 21]: accum_qk_mean += get_map("qk_mean")
        if idx == 1: accum_shallow += (1 - get_map("kk_mean")) * get_map("qk_var")
        if 6 <= idx <= 13: accum_middle += (1 - get_map("qq_mean"))
        if 30 <= idx <= 37: accum_deep += (1 - (1 - get_map("qq_var")) * get_map("qq_mean"))
        block.attn._captured_stats = {}

    # 1. 初始 Coarse Mask
    w_s = normalize_map(accum_shallow.cpu().numpy())
    w_m = normalize_map(accum_middle.cpu().numpy())
    w_d = normalize_map(accum_deep.cpu().numpy())
    mask_gram = w_s * w_m * w_d
    
    qk_val = normalize_map(accum_qk_mean.cpu().numpy())
    mask_std = 1.0 - qk_val
    
    # 融合
    mask_sequence_raw = (mask_gram + mask_std) / 2.0 # (S, H, W)
    
    # 2. [关键] 时序平滑 (Temporal Smoothing)
    # sigma=1.0 约等于前后各1帧的加权平均，平滑闪烁
    print("  -> Applying Temporal Gaussian Smoothing...")
    mask_sequence_smooth = gaussian_filter1d(mask_sequence_raw, sigma=1.0, axis=0)
    
    # 3. 空间细化循环
    refined_masks = []
    H, W = results_p1.depth.shape[1], results_p1.depth.shape[2]
    
    print("  -> Applying Spatial Guided Filter & Hysteresis...")
    for t in tqdm(range(num_frames)):
        rgb = cv2.imread(img_files[t])
        rgb = cv2.resize(rgb, (W, H))
        
        # 准备 Guide: RGB + Depth 融合
        rgb_norm = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        depth = results_p1.depth[t]
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        guide_img = rgb_norm * 0.7 + depth_norm * 0.3
        
        # 取时序平滑后的概率图
        p = mask_sequence_smooth[t]
        
        # Guided Filter (平滑保边)
        # r=W*0.02 (约10-20像素)，eps=1e-3 (稍大一点以抑制背景噪声)
        refined = guided_filter(guide_img, p, r=int(W*0.02), eps=1e-3)
        refined = np.clip(refined, 0, 1)
        
        # Hysteresis Thresholding (双阈值保持完整性)
        # 强阈值 0.6，弱阈值 0.3
        final_mask = apply_hysteresis_threshold(refined, low=0.3, high=0.6)
        
        refined_masks.append(final_mask)
        
    refined_masks_np = np.stack(refined_masks)
    
    # --- Pass 2: Inference ---
    print("[4/4] Pass 2: Masked Inference...")
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
    
    os.makedirs(args.output_dir, exist_ok=True)
    for t in range(num_frames):
        cv2.imwrite(os.path.join(args.output_dir, f"mask_{t:03d}.png"), (refined_masks_np[t]*255).astype(np.uint8))
        d1 = normalize_map(results_p1.depth[t])
        d2 = normalize_map(results_p2.depth[t])
        vis = np.concatenate([d1, d2], axis=1)
        cv2.imwrite(os.path.join(args.output_dir, f"depth_compare_{t:03d}.png"), (vis*255).astype(np.uint8))
        
    print(f"Done. Output: {args.output_dir}")

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