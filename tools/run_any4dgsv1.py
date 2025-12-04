# tools/run_vggt4d_refined.py
"""
Usage:
python tools/run_vggt4d_refined.py \
    --image-dir ./demo/kling \
    --output-dir ./analysis/vis_results_refined \
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
# Part 1: 增强版图像处理工具 (RGB-D Guided Filter)
# =========================================================================
def guided_filter_multichannel(I, p, r, eps):
    """
    多通道引导滤波: 支持 RGB 或 RGB-D 作为引导图 I
    I: (H, W, C) Guide Image
    p: (H, W) Input Mask
    """
    # 简单的独立通道近似实现 (Independent Channel Guided Filter)
    # 这种方式比完整的协方差矩阵求逆快得多，且效果在 Mask Refinement 上足够好
    
    # 确保 p 也是多通道广播
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    
    # I 的每个通道分别计算
    # a_k = cov_Ip_k / (var_I_k + eps)
    # b_k = mean_p - a_k * mean_I_k
    # q = mean(a_k) * I + mean(b_k)
    
    C = I.shape[2]
    a_sum = np.zeros_like(p, dtype=np.float64)
    b_sum = np.zeros_like(p, dtype=np.float64)
    
    for c in range(C):
        I_c = I[:, :, c]
        mean_I_c = cv2.boxFilter(I_c, cv2.CV_64F, (r, r))
        mean_Ip_c = cv2.boxFilter(I_c * p, cv2.CV_64F, (r, r))
        mean_II_c = cv2.boxFilter(I_c * I_c, cv2.CV_64F, (r, r))
        
        var_I_c = mean_II_c - mean_I_c * mean_I_c
        cov_Ip_c = mean_Ip_c - mean_I_c * mean_p
        
        a_c = cov_Ip_c / (var_I_c + eps)
        b_c = mean_p - a_c * mean_I_c
        
        a_sum += a_c
        b_sum += b_c
        
    a_avg = a_sum / C
    b_avg = b_sum / C
    
    mean_a = cv2.boxFilter(a_avg, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b_avg, cv2.CV_64F, (r, r))
    
    # 这里有点 trick: 我们希望 Guide 作为一个整体去引导
    # 简单平均参数是一种近似。更严谨的是用 Color Guided Filter。
    # 但为了代码独立性，这里使用 "灰度化引导" 的变体：
    # 如果 I 是多通道，先 PCA 或 Mean 降维成 1 通道再滤波可能更好？
    # 不，我们直接用 I 的平均特征来引导。
    
    # 为了保持简单且利用深度信息，我们采用一种更直观的策略：
    # 将 RGB 和 Depth 拼接，计算一个“联合边缘强度”，然后作为单通道引导图。
    # 或者：分别对 RGB 和 Depth 做 Guided Filter，然后取交集？
    
    # 修正策略：仅使用单通道 Guided Filter，但 Guide Image 是 RGB 和 Depth 的 PCA 主成分或平均值。
    # 这里我们简化：I_guide = (R+G+B)/3 * 0.7 + Depth * 0.3
    pass 

def simple_guided_filter(I_guide, p, r, eps):
    """标准单通道引导滤波"""
    mean_I = cv2.boxFilter(I_guide, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(I_guide * p, cv2.CV_64F, (r, r))
    mean_II = cv2.boxFilter(I_guide * I_guide, cv2.CV_64F, (r, r))
    
    var_I = mean_II - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    
    q = mean_a * I_guide + mean_b
    return q

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
    
    # --- Pass 1 ---
    print("[2/4] Pass 1: Extracting Cues...")
    for m in model.model.modules():
        if isinstance(m, Attention): m.collect_stats = True
            
    results_p1 = model.inference(pil_imgs, infer_gs=False)
    
    # --- Mask Generation ---
    print("[3/4] High-Precision Mask Refinement (RGB-D)...")
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

    # Initial Coarse Mask
    w_s = normalize_map(accum_shallow.cpu().numpy())
    w_m = normalize_map(accum_middle.cpu().numpy())
    w_d = normalize_map(accum_deep.cpu().numpy())
    mask_gram = w_s * w_m * w_d
    
    qk_val = normalize_map(accum_qk_mean.cpu().numpy())
    mask_std = 1.0 - qk_val
    
    mask_coarse = (mask_gram + mask_std) / 2.0
    
    refined_masks = []
    H, W = results_p1.depth.shape[1], results_p1.depth.shape[2]
    
    # --- Refinement Loop ---
    for t in tqdm(range(num_frames)):
        # 1. 准备引导图 Guide Image
        # 读取 RGB
        rgb = cv2.imread(img_files[t])
        rgb = cv2.resize(rgb, (W, H))
        rgb_norm = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        # 读取 Depth (作为额外的几何引导)
        depth = results_p1.depth[t]
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        
        # 融合 RGB 和 Depth 边缘
        # 简单策略: Guide = RGB * 0.7 + Depth * 0.3
        # 这样 Guide Image 既有纹理边缘，也有深度几何边缘
        guide_img = rgb_norm * 0.7 + depth_norm * 0.3
        
        # 2. 形态学清理 (去除孤立噪点)
        p = mask_coarse[t]
        p_uint8 = (p * 255).astype(np.uint8)
        # 开运算: 腐蚀->膨胀 (去白噪)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        p_cleaned = cv2.morphologyEx(p_uint8, cv2.MORPH_OPEN, kernel)
        p_cleaned = p_cleaned.astype(np.float32) / 255.0
        
        # 3. RGB-D Guided Filter
        # 半径 r 可以稍微小一点以保持细节
        refined = simple_guided_filter(guide_img, p_cleaned, r=int(W*0.01), eps=1e-4)
        refined = np.clip(refined, 0, 1)
        
        # 4. 自适应阈值 (Otsu)
        # 将 float 转回 uint8 进行 Otsu 计算
        refined_u8 = (refined * 255).astype(np.uint8)
        # 只有当画面中有明显动态物体时 Otsu 才有效，否则容易把背景也切出来
        # 加入一个全局判断: 如果 refined 整体很暗，说明这帧是静态的
        if refined.mean() < 0.05:
            final_mask = np.zeros_like(refined)
        else:
            thresh_val, final_mask_u8 = cv2.threshold(refined_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            final_mask = final_mask_u8.astype(np.float32) / 255.0
            
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