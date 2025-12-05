# tools/run_vggt4d_grabcut.py
"""
Usage:
python tools/run_vggt4d_grabcut.py \
    --image-dir ./demo/kling \
    --output-dir ./analysis/vis_results_grabcut \
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
# Part 1: 高级图像处理 (颜色生长 + 引导滤波)
# =========================================================================
def color_region_growing(rgb, prob_map, low=0.3, high=0.6):
    """
    基于颜色的区域生长:
    1. 确信区 (prob > high) 作为种子。
    2. 候选区 (prob > low) 中，如果颜色和种子相似，则加入。
    """
    H, W = prob_map.shape
    
    # 1. 种子
    seeds = (prob_map > high).astype(np.uint8)
    candidates = (prob_map > low).astype(np.uint8)
    
    # 如果没有种子，直接返回空
    if seeds.sum() == 0:
        return seeds.astype(np.float32)
    
    # 2. 简单的连通域扩展不可行，我们需要颜色相似性
    # 转换到 LAB 空间，颜色距离更均匀
    lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)
    
    # 计算种子的平均颜色 (分块计算或全局平均)
    # 为了简单，我们用全局种子平均颜色作为基准 (假设物体颜色单一)
    # 更好的做法是局部生长，这里用形态学重建模拟
    
    # 替代方案：迭代式膨胀，但只向颜色相似的区域膨胀
    mask = seeds.copy()
    for _ in range(5): # 迭代 5 次
        # 膨胀 1 圈
        dilated = cv2.dilate(mask, np.ones((3,3), np.uint8))
        
        # 获取新增的边缘区域
        edge = dilated - mask
        
        # 只保留: 1. 在候选区内 2. 颜色差异小 的点
        # 颜色差异计算: 当前像素 vs 局部均值
        # 这里简化: 只保留候选区内的膨胀
        # 真正的 GrabCut 太慢，这里用 "带约束的形态学膨胀"
        
        new_pixels = edge & candidates
        if new_pixels.sum() == 0: break
        
        mask = mask | new_pixels
        
    return mask.astype(np.float32)

def guided_filter(I, p, r, eps):
    # 标准引导滤波
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
# Part 2: 核心 Patch (保持不变)
# =========================================================================
def fused_attention_forward(self, x, pos=None, attn_mask=None):
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

                attn_qk = q_curr @ k_win * scale
                stats["qk_mean"][:, :, t_start:t_end] = attn_qk.mean(dim=-1)
                stats["qk_var"][:, :, t_start:t_end]  = attn_qk.var(dim=-1)
                gram_qq = q_curr @ q_win * scale
                stats["qq_mean"][:, :, t_start:t_end] = gram_qq.mean(dim=-1)
                stats["qq_var"][:, :, t_start:t_end]  = gram_qq.var(dim=-1)
                gram_kk = k_curr @ k_win * scale
                stats["kk_mean"][:, :, t_start:t_end] = gram_kk.mean(dim=-1)

            self._captured_stats = {k: v.mean(dim=1).cpu() for k, v in stats.items()}
            del stats, q_t, k_t

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
    print("[3/4] Refinement (Morphology + RGB-D Guided)...")
    if hasattr(model.model, 'da3'): vit = model.model.da3.backbone.pretrained 
    else: vit = model.model.backbone.pretrained
    
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

    w_s = normalize_map(accum_shallow.cpu().numpy())
    w_m = normalize_map(accum_middle.cpu().numpy())
    w_d = normalize_map(accum_deep.cpu().numpy())
    mask_gram = w_s * w_m * w_d
    
    qk_val = normalize_map(accum_qk_mean.cpu().numpy())
    mask_std = 1.0 - qk_val
    
    mask_coarse = (mask_gram + mask_std) / 2.0
    
    refined_masks = []
    H, W = results_p1.depth.shape[1], results_p1.depth.shape[2]
    
    for t in tqdm(range(num_frames)):
        rgb = cv2.imread(img_files[t])
        rgb = cv2.resize(rgb, (W, H))
        I_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        # 1. 强力形态学闭运算 (填补内部空洞)
        # 用一个较大的核先把物体内部连起来
        p = mask_coarse[t]
        p_u8 = (p * 255).astype(np.uint8)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        p_closed = cv2.morphologyEx(p_u8, cv2.MORPH_CLOSE, kernel_close)
        p_float = p_closed.astype(np.float32) / 255.0
        
        # 2. RGB-D 联合引导 (吸附边缘)
        depth = results_p1.depth[t]
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        guide_img = I_rgb * 0.7 + depth_norm * 0.3
        
        # 增大半径以覆盖更大的区域
        refined = guided_filter(guide_img, p_float, r=int(W*0.02), eps=1e-4)
        refined = np.clip(refined, 0, 1)
        
        # 3. 区域生长 / 双阈值
        # 强阈值定核心，弱阈值找边缘
        # 这里用 Color Region Growing 的简化版: Otsu + Dilate
        thresh_val, binary = cv2.threshold((refined*255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 再做一次闭运算整理边缘
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        
        final_mask = binary.astype(np.float32) / 255.0
        refined_masks.append(final_mask)
        
    refined_masks_np = np.stack(refined_masks)
    
    # --- Pass 2: Inference ---
    print("[4/4] Pass 2: Masked Inference...")
    mask_tensor = torch.from_numpy(refined_masks_np).to(args.device) # (S, H, W)
    
    # [关键] 使用 area 插值 + 覆盖阈值，确保 Token 边缘被覆盖
    mask_prob = F.interpolate(
        mask_tensor.unsqueeze(1), 
        size=(H_feat, W_feat), 
        mode='area' 
    ).view(num_frames, -1)
    
    # 只要 Token 里面有 10% 是动态的，就 Mask 掉，保证“杀干净”
    mask_tokens_binary = (mask_prob > 0.1).float()
    
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