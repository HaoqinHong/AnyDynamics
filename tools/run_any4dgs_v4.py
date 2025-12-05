# tools/run_any4dgs_v4.py
"""
Usage:
python tools/run_any4dgs_v4.py \
    --image-dir ./demo/kling \
    --output-dir ./analysis/vis_results_global_smooth \
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

from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.dinov2.layers.attention import Attention

# =========================================================================
# Part 1: 高级图像处理 (时空平滑 + 引导滤波)
# =========================================================================
def temporal_smooth_3d(volume, sigma_t=1.5, sigma_s=1.0):
    """
    3D 时空平滑: 同时在时间轴和空间轴去噪
    volume: (T, H, W) numpy array
    """
    # 1. 时间轴平滑 (消除闪烁)
    T = volume.shape[0]
    radius = int(3 * sigma_t + 0.5)
    t_kernel = np.exp(-np.arange(-radius, radius+1)**2 / (2 * sigma_t**2))
    t_kernel /= t_kernel.sum()
    
    vol_smooth_t = np.zeros_like(volume)
    for t in range(T):
        val = 0
        w_sum = 0
        for k in range(-radius, radius+1):
            idx = np.clip(t + k, 0, T - 1)
            val += volume[idx] * t_kernel[k + radius]
            w_sum += t_kernel[k + radius]
        vol_smooth_t[t] = val / w_sum
        
    # 2. 空间轴平滑 (消除斑点)
    vol_final = np.zeros_like(vol_smooth_t)
    k_size = int(3 * sigma_s) * 2 + 1
    for t in range(T):
        vol_final[t] = cv2.GaussianBlur(vol_smooth_t[t], (k_size, k_size), sigma_s)
        
    return vol_final

def guided_filter(I, p, r, eps):
    # 引导滤波保边
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

# =========================================================================
# Part 2: 核心 Patch (Gram + QK 统计)
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
            
            # 适配 Local/Global
            if B == num_frames and B > 1:
                q_in = q.permute(1, 0, 2, 3).reshape(1, self.num_heads, -1, C // self.num_heads)
                k_in = k.permute(1, 0, 2, 3).reshape(1, self.num_heads, -1, C // self.num_heads)
                total_tokens = B * N
            else:
                q_in, k_in = q, k
                total_tokens = N
            
            tokens_per_frame = total_tokens // num_frames
            
            # 使用 float32 避免溢出
            stats = {
                "qk_mean": torch.zeros(1, self.num_heads, total_tokens, device=q.device, dtype=torch.float32),
                "qk_var":  torch.zeros(1, self.num_heads, total_tokens, device=q.device, dtype=torch.float32),
                "qq_mean": torch.zeros(1, self.num_heads, total_tokens, device=q.device, dtype=torch.float32),
                "qq_var":  torch.zeros(1, self.num_heads, total_tokens, device=q.device, dtype=torch.float32),
                "kk_mean": torch.zeros(1, self.num_heads, total_tokens, device=q.device, dtype=torch.float32),
            }
            
            q_t = q_in.transpose(-2, -1)
            k_t = k_in.transpose(-2, -1)

            # 滑动窗口
            for t in range(num_frames):
                t_s, t_e = t * tokens_per_frame, (t+1) * tokens_per_frame
                w_s = max(0, t - window) * tokens_per_frame
                w_e = min(num_frames, t + window + 1) * tokens_per_frame
                
                q_curr = q_in[:, :, t_s:t_e]
                k_curr = k_in[:, :, t_s:t_e]
                q_win = q_t[:, :, :, w_s:w_e]
                k_win = k_t[:, :, :, w_s:w_e]

                attn_qk = q_curr @ k_win * scale
                stats["qk_mean"][:, :, t_s:t_e] = attn_qk.mean(dim=-1)
                stats["qk_var"][:, :, t_s:t_e]  = attn_qk.var(dim=-1)
                
                gram_qq = q_curr @ q_win * scale
                stats["qq_mean"][:, :, t_s:t_e] = gram_qq.mean(dim=-1)
                stats["qq_var"][:, :, t_s:t_e]  = gram_qq.var(dim=-1)
                
                gram_kk = k_curr @ k_win * scale
                stats["kk_mean"][:, :, t_s:t_e] = gram_kk.mean(dim=-1)

            # 转 CPU 存入 buffer
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
    
    # --- Pass 1: 提取 ---
    print("[2/4] Pass 1: Extracting Cues...")
    for m in model.model.modules():
        if isinstance(m, Attention): m.collect_stats = True
            
    results_p1 = model.inference(pil_imgs, infer_gs=False)
    
    # --- 全局特征处理 ---
    print("[3/4] Processing Heatmaps (Global Norm + 3D Smooth)...")
    if hasattr(model.model, 'da3'): vit = model.model.da3.backbone.pretrained 
    else: vit = model.model.backbone.pretrained
    
    H_feat, W_feat = results_p1.depth.shape[1] // 14, results_p1.depth.shape[2] // 14
    
    # 收集 Raw Logits
    accum_gram = torch.zeros(num_frames, H_feat, W_feat, device=args.device)
    accum_qk = torch.zeros(num_frames, H_feat, W_feat, device=args.device)
    
    for idx, block in enumerate(vit.blocks):
        if not block.attn._captured_stats: continue
        stats = block.attn._captured_stats
        
        # [FIXED] 增加 .to(args.device) 以匹配累加器设备
        def get_raw(name):
            d = stats[name].view(num_frames, -1)
            if d.shape[1] == (H_feat * W_feat + 1): d = d[:, 1:]
            return d.view(num_frames, H_feat, W_feat).to(args.device)

        if idx in [1, 9, 21]: 
            accum_qk += get_raw("qk_mean")
        if idx == 1: 
            accum_gram += (1 - get_raw("kk_mean")) * get_raw("qk_var")
        if 6 <= idx <= 13: 
            accum_gram += (1 - get_raw("qq_mean"))
        if 30 <= idx <= 37: 
            static_score = (1 - get_raw("qq_var")) * get_raw("qq_mean")
            accum_gram += (1 - static_score)
            
        block.attn._captured_stats = {}

    # 转到 CPU numpy
    gram_np = accum_gram.cpu().numpy()
    qk_np = accum_qk.cpu().numpy()
    
    # 1. 全局归一化
    def global_norm(vol):
        v_min, v_max = np.percentile(vol, 2), np.percentile(vol, 98)
        return np.clip((vol - v_min) / (v_max - v_min + 1e-6), 0, 1)
        
    mask_gram = global_norm(gram_np)
    mask_std = 1.0 - global_norm(qk_np)
    mask_coarse_vol = (mask_gram + mask_std) / 2.0
    
    # 2. 3D 时空平滑
    print("  -> Applying 3D Spatio-Temporal Smoothing...")
    mask_smooth_vol = temporal_smooth_3d(mask_coarse_vol, sigma_t=1.5, sigma_s=0.5)
    
    # Upsample & Guided Filter Loop
    refined_masks = []
    H, W = results_p1.depth.shape[1], results_p1.depth.shape[2]
    
    print("  -> Applying Guided Filter per frame...")
    for t in tqdm(range(num_frames)):
        rgb = cv2.imread(img_files[t])
        rgb = cv2.resize(rgb, (W, H))
        I = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        # Upsample coarse mask
        p_small = mask_smooth_vol[t]
        p_big = cv2.resize(p_small, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # Guided Filter
        refined = guided_filter(I, p_big, r=int(W*0.02), eps=1e-4)
        refined = np.clip(refined, 0, 1)
        
        # 3. 动态阈值二值化 (Otsu)
        if refined.max() < 0.3:
            final_mask = np.zeros_like(refined)
        else:
            valid_pix = refined[refined > 0.1]
            if len(valid_pix) == 0:
                thresh = 0.5
            else:
                thresh = valid_pix.mean() 
            final_mask = (refined > thresh).astype(np.float32)
            
        refined_masks.append(final_mask)
        
    refined_masks_np = np.stack(refined_masks)
    
    # --- Pass 2 ---
    print("[4/4] Pass 2: Masked Inference...")
    mask_tensor = torch.from_numpy(refined_masks_np).to(args.device)
    mask_small = F.interpolate(mask_tensor.unsqueeze(1), size=(H_feat, W_feat), mode='area').view(num_frames, -1)
    mask_tokens_binary = (mask_small > 0.1).float()
    
    cls_mask = torch.zeros(num_frames, 1, device=args.device)
    mask_tokens = torch.cat([cls_mask, mask_tokens_binary], dim=1)
    
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
        d1 = F.interpolate(torch.from_numpy(results_p1.depth[t]).unsqueeze(0).unsqueeze(0), size=(H,W)).squeeze().numpy()
        d2 = F.interpolate(torch.from_numpy(results_p2.depth[t]).unsqueeze(0).unsqueeze(0), size=(H,W)).squeeze().numpy()
        
        def norm_disp(d):
            return np.clip((d - d.min()) / (d.max() - d.min() + 1e-6), 0, 1)
            
        vis = np.concatenate([norm_disp(d1), norm_disp(d2)], axis=1)
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