# tools/collect_spatial_features.py (Ultimate Hybrid Version)
"""
Usage:
python tools/collect_spatial_features.py \
    --model-name da3nested-giant-large \
    --image-dir ./demo/kling \
    --output-dir ./analysis/vis_results_kling
"""
import argparse
import types
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import gc
from pathlib import Path
from PIL import Image

from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.dinov2.layers.attention import Attention

# ==========================================
# PART 1: 全能特征提取 (计算 QK, QQ, KK)
# ==========================================
def custom_attention_forward_hybrid(self, x, pos=None, attn_mask=None):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q, k = self.q_norm(q), self.k_norm(k)

    # 1. 应用 RoPE
    if self.rope is not None and pos is not None:
        q = self.rope(q, pos)
        k = self.rope(k, pos)

    # 2. 计算特征
    if getattr(self, "collect_stats", False):
        with torch.no_grad():
            scale = self.scale
            q_t = q.transpose(-2, -1)
            k_t = k.transpose(-2, -1)
            
            # 准备累加器 (全部存下来)
            # Method 1 (QK): 标准 Attention 统计量
            acc_m_qk = torch.zeros(B, self.num_heads, N, device=q.device, dtype=torch.float32)
            acc_v_qk = torch.zeros(B, self.num_heads, N, device=q.device, dtype=torch.float32)
            
            # Method 2 (Gram): VGGT4D 专用统计量
            acc_m_qq = torch.zeros(B, self.num_heads, N, device=q.device, dtype=torch.float32)
            acc_v_qq = torch.zeros(B, self.num_heads, N, device=q.device, dtype=torch.float32)
            acc_m_kk = torch.zeros(B, self.num_heads, N, device=q.device, dtype=torch.float32)
            
            chunk_size = 1024 
            for i in range(0, N, chunk_size):
                end = min(i + chunk_size, N)
                q_chunk = q[:, :, i:end, :]
                k_chunk = k[:, :, i:end, :]
                
                # --- A. 计算 QK (Standard Cross-Attention) ---
                # 对应 Method 1: S = Mean(QK), V = Var(QK)
                attn_qk = torch.matmul(q_chunk, k_t) * scale
                acc_m_qk[:, :, i:end] = attn_qk.mean(dim=-1)
                acc_v_qk[:, :, i:end] = attn_qk.var(dim=-1)
                del attn_qk
                
                # --- B. 计算 QQ (Gram Self-Similarity) ---
                # 对应 Method 2: S^QQ, V^QQ
                gram_qq = torch.matmul(q_chunk, q_t) * scale
                acc_m_qq[:, :, i:end] = gram_qq.mean(dim=-1)
                acc_v_qq[:, :, i:end] = gram_qq.var(dim=-1)
                del gram_qq
                
                # --- C. 计算 KK (Gram Key-Similarity) ---
                # 对应 Method 2: S^KK
                gram_kk = torch.matmul(k_chunk, k_t) * scale
                acc_m_kk[:, :, i:end] = gram_kk.mean(dim=-1)
                del gram_kk

            # 保存所有结果
            self._captured_stats = {
                # Method 1: QK Stats
                "qk_mean": acc_m_qk.mean(dim=1).cpu(),
                "qk_var":  acc_v_qk.mean(dim=1).cpu(),
                
                # Method 2: Gram Stats
                "qq_mean": acc_m_qq.mean(dim=1).cpu(),
                "qq_var":  acc_v_qq.mean(dim=1).cpu(),
                "kk_mean": acc_m_kk.mean(dim=1).cpu(),
            }

    # 3. 原有 Forward
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
    print("Applying Ultimate Hybrid Monkey Patch...")
    for module in model.modules():
        if isinstance(module, Attention):
            module.forward = types.MethodType(custom_attention_forward_hybrid, module)
            module.collect_stats = False
            module._captured_stats = {}

# ==========================================
# PART 2: 通用可视化辅助函数
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

def save_heatmap(data, path, W_raw, H_raw):
    # 对比度增强 + 平滑
    data = np.power(data, 1.5)
    resized = cv2.resize(data, (W_raw, H_raw), interpolation=cv2.INTER_CUBIC)
    k = int(W_raw * 0.02) | 1
    blurred = cv2.GaussianBlur(resized, (k, k), 0)
    color = cv2.applyColorMap((blurred * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(path, color)

def load_images(image_dir):
    files = sorted(list(Path(image_dir).glob("*.png")) + list(Path(image_dir).glob("*.jpg")))
    if not files: raise ValueError("No images found")
    return [Image.open(p).convert("RGB") for p in files]

# ==========================================
# PART 3: 主程序
# ==========================================
@torch.no_grad()
def main(args):
    print(f"Loading model: {args.model_name}...")
    torch.cuda.empty_cache()
    gc.collect()
    
    model = DepthAnything3.from_pretrained(f"depth-anything/{args.model_name.upper()}").to(args.device)
    model.eval()
    apply_monkey_patch(model.model)
    
    imgs = load_images(args.image_dir)
    print(f"Loaded {len(imgs)} images. Running Inference...")
    
    for module in model.model.modules():
        if isinstance(module, Attention): module.collect_stats = True
        
    _ = model.inference(imgs, infer_gs=False)
    
    # 拆分并保存数据
    print("Splitting features...")
    stats_qk, stats_gram = {}, {}
    
    if hasattr(model.model, 'da3'): vit = model.model.da3.backbone.pretrained
    else: vit = model.model.backbone.pretrained

    for idx, block in enumerate(vit.blocks):
        if hasattr(block.attn, "_captured_stats") and block.attn._captured_stats:
            captured = block.attn._captured_stats
            key = f"layer_{idx:02d}"
            
            # Method 1: QK (Standard Attention)
            if "qk_mean" in captured:
                stats_qk[key] = {
                    "qk_mean": captured["qk_mean"],
                    "qk_var": captured["qk_var"]
                }
            
            # Method 2: Gram (VGGT4D)
            # 注意: VGGT4D 的 v_qk 其实就是 qk_var
            if "qq_mean" in captured:
                stats_gram[key] = {
                    "s_qq": captured["qq_mean"],
                    "v_qq": captured["qq_var"],
                    "s_kk": captured["kk_mean"],
                    "v_qk": captured["qk_var"] # 复用 QK var
                }
            
            del block.attn._captured_stats
            block.attn._captured_stats = {}

    gc.collect()

    # 目录结构
    root_dir = Path(args.output_dir)
    dir_qk = root_dir / "method_1_QK_Attention"
    dir_gram = root_dir / "method_2_VGGT4D_Gram"
    
    dir_qk.mkdir(parents=True, exist_ok=True)
    dir_gram.mkdir(parents=True, exist_ok=True)
    (dir_qk / "images").mkdir(exist_ok=True)
    (dir_gram / "images").mkdir(exist_ok=True)

    meta = {"H": imgs[0].height, "W": imgs[0].width, "num_frames": len(imgs)}
    aspect_ratio = meta["W"] / meta["H"]

    print(f"Saving .pt files to {root_dir}...")
    torch.save({"stats": stats_qk, "meta": meta}, dir_qk / "qk_features.pt")
    torch.save({"stats": stats_gram, "meta": meta}, dir_gram / "gram_features.pt")

    # ==========================
    # 可视化 Method 1: Standard QK
    # ==========================
    print("Visualizing Method 1 (Standard QK)...")
    selected_layers = [1, 9, 21]
    for layer_idx in selected_layers:
        key = f"layer_{layer_idx:02d}"
        if key not in stats_qk: continue
        
        # 1. 提取数据
        qk_mean = stats_qk[key]["qk_mean"].float().numpy().flatten()
        qk_var = stats_qk[key]["qk_var"].float().numpy().flatten()
        
        grid, offset = infer_grid(qk_mean.shape[0], meta["num_frames"], aspect_ratio)
        if not grid: continue
        
        map_mean = normalize_and_reshape(qk_mean, meta["num_frames"], grid, offset)
        map_var = normalize_and_reshape(qk_var, meta["num_frames"], grid, offset)
        
        for t in range(min(10, meta["num_frames"])):
            base = str(dir_qk / "images" / f"{key}_frame{t:03d}")
            
            # --- 你要求的 4 种输出 ---
            # 1. _similarity.png (原始均值): S
            save_heatmap(map_mean[t], base + "_similarity.png", meta["W"], meta["H"])
            
            # 2. _inv_similarity.png (反转均值): 1 - S  [重点: 浅层动态]
            save_heatmap(1.0 - map_mean[t], base + "_inv_similarity.png", meta["W"], meta["H"])
            
            # 3. _variance.png (方差): V  [重点: 深层动态]
            save_heatmap(map_var[t], base + "_variance.png", meta["W"], meta["H"])
            
            # 4. _stability.png (稳定性): 1 - V
            save_heatmap(1.0 - map_var[t], base + "_stability.png", meta["W"], meta["H"])

    # ==========================
    # 可视化 Method 2: VGGT4D Gram
    # ==========================
    print("Visualizing Method 2 (VGGT4D Gram)...")
    for layer_idx in selected_layers:
        key = f"layer_{layer_idx:02d}"
        if key not in stats_gram: continue
        
        s_qq = stats_gram[key]["s_qq"].float().numpy().flatten()
        v_qq = stats_gram[key]["v_qq"].float().numpy().flatten()
        s_kk = stats_gram[key]["s_kk"].float().numpy().flatten()
        v_qk = stats_gram[key]["v_qk"].float().numpy().flatten()
        
        grid, offset = infer_grid(s_qq.shape[0], meta["num_frames"], aspect_ratio)
        if not grid: continue
        
        map_s_qq = normalize_and_reshape(s_qq, meta["num_frames"], grid, offset)
        map_v_qq = normalize_and_reshape(v_qq, meta["num_frames"], grid, offset)
        map_s_kk = normalize_and_reshape(s_kk, meta["num_frames"], grid, offset)
        map_v_qk = normalize_and_reshape(v_qk, meta["num_frames"], grid, offset)
        
        for t in range(min(10, meta["num_frames"])):
            w_shallow = (1.0 - map_s_kk[t]) * map_v_qk[t]
            w_middle = 1.0 - map_s_qq[t]
            w_deep_bg = map_s_qq[t] * (1.0 - map_v_qq[t]) # 静态背景
            w_deep_dyn = 1.0 - w_deep_bg                  # 动态物体
            final = w_shallow * w_middle * w_deep_dyn
            
            base = str(dir_gram / "images" / f"{key}_frame{t:03d}")
            save_heatmap(w_shallow, base + "_shallow.png", meta["W"], meta["H"])
            save_heatmap(w_deep_dyn, base + "_deep_var.png", meta["W"], meta["H"])
            save_heatmap(final, base + "_final.png", meta["W"], meta["H"])

    print(f"\nSuccess! Check results in: {root_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="da3nested-giant-large")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", default="./analysis/vis_results_kling")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args)