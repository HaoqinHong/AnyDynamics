# tools/run_vggt4d_full.py
"""
Usage:
python tools/run_vggt4d_full.py \
    --image-dir ./demo/kling \
    --output-dir ./analysis/vggt4d_strict_result \
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
import open3d as o3d
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import DBSCAN

from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.dinov2.layers.attention import Attention

# =========================================================================
# Part 0: Pose 处理工具
# =========================================================================
def process_poses(extrinsics_3x4):
    N = extrinsics_3x4.shape[0]
    w2c_homo = np.concatenate([extrinsics_3x4, np.zeros((N, 1, 4))], axis=1) 
    w2c_homo[:, 3, 3] = 1.0
    c2w_homo = np.linalg.inv(w2c_homo)
    return c2w_homo

# =========================================================================
# Part 1: 通用几何工具
# =========================================================================
def unproject_depth(depth, intrinsics, device):
    B, H, W = depth.shape
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    x = x.reshape(1, -1).repeat(B, 1).float() + 0.5
    y = y.reshape(1, -1).repeat(B, 1).float() + 0.5
    z = depth.reshape(B, -1)

    cx = intrinsics[:, 0, 2].unsqueeze(1)
    cy = intrinsics[:, 1, 2].unsqueeze(1)
    fx = intrinsics[:, 0, 0].unsqueeze(1)
    fy = intrinsics[:, 1, 1].unsqueeze(1)

    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    pts_cam = torch.stack([X, Y, z], dim=-1) # (B, N, 3)
    return pts_cam

# =========================================================================
# Part 2: VGGT4D 核心 Monkey Patch
# =========================================================================
def vggt4d_attention_forward(self, x, pos=None, attn_mask=None):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q, k = self.q_norm(q), self.k_norm(k)

    if self.rope is not None and pos is not None:
        q = self.rope(q, pos)
        k = self.rope(k, pos)

    # --- Mode A: 收集动态线索 ---
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
                "qk_var": torch.zeros(1, self.num_heads, total_tokens, device=q.device),
                "qq_mean": torch.zeros(1, self.num_heads, total_tokens, device=q.device),
                "qq_var": torch.zeros(1, self.num_heads, total_tokens, device=q.device),
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

                stats["qk_var"][:, :, t_start:t_end] = (q_curr @ k_win * scale).var(dim=-1)
                
                gram_qq = q_curr @ q_win * scale
                stats["qq_mean"][:, :, t_start:t_end] = gram_qq.mean(dim=-1)
                stats["qq_var"][:, :, t_start:t_end] = gram_qq.var(dim=-1)
                
                gram_kk = k_curr @ k_win * scale
                stats["kk_mean"][:, :, t_start:t_end] = gram_kk.mean(dim=-1)

            self._captured_stats = {k: v.mean(dim=1).cpu() for k, v in stats.items()}
            del stats, q_t, k_t, q_in, k_in

    # --- Mode B: 注入动态掩码 ---
    is_mask_injected = False
    if getattr(self, "apply_masking", False) and hasattr(self, "dynamic_mask"):
        mask_map = self.dynamic_mask.to(x.device)
        S, N_p = mask_map.shape
        
        if B == S: # Local
            bias = torch.zeros(S, 1, 1, N_p, device=x.device)
            bias.masked_fill_(mask_map.view(S, 1, 1, N_p).bool(), float("-inf"))
        else: # Global
            bias = torch.zeros(1, 1, 1, S*N_p, device=x.device)
            bias.masked_fill_(mask_map.view(1, 1, 1, S*N_p).bool(), float("-inf"))
            
        if attn_mask is None:
            attn_mask = bias
        else:
            attn_mask = attn_mask + bias
        is_mask_injected = True

    if self.fused_attn:
        if not is_mask_injected and attn_mask is not None:
             attn_mask = attn_mask[:, None].repeat(1, self.num_heads, 1, 1)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=attn_mask)
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

def apply_patch(model, num_frames, window_size=2):
    for module in model.modules():
        if isinstance(module, Attention):
            module.forward = types.MethodType(vggt4d_attention_forward, module)
            module.collect_stats = False
            module.apply_masking = False
            module.num_frames = num_frames
            module.window_size = window_size
            module._captured_stats = {}

# =========================================================================
# Part 3: 投影梯度细化 (Strict: Sum of Norms + Photometric)
# =========================================================================
def compute_projection_gradient_strict(pts_world, depths, poses, intrinsics, coarse_masks, images, H, W, window_size, t_curr, num_frames):
    device = pts_world.device
    pts_world = pts_world.detach().requires_grad_(True)
    
    total_score = torch.zeros(pts_world.shape[0], device=device)
    valid_count = torch.zeros(pts_world.shape[0], device=device)
    
    w_start = max(0, t_curr - window_size)
    w_end = min(num_frames, t_curr + window_size + 1)
    sources = [i for i in range(w_start, w_end) if i != t_curr]
    
    if not sources: return total_score

    # [FIX] 关键修复: Resize 图片到 Depth 分辨率以匹配点数
    img_curr = images[0, t_curr] # (3, H_raw, W_raw)
    if img_curr.shape[1] != H or img_curr.shape[2] != W:
        img_curr_resized = F.interpolate(img_curr.unsqueeze(0), size=(H, W), mode='area').squeeze(0)
    else:
        img_curr_resized = img_curr
        
    c_ref = img_curr_resized.permute(1, 2, 0).reshape(-1, 3) # (N, 3)

    for i in sources:
        pose = poses[0, i]
        K = intrinsics[0, i]
        obs_depth_map = depths[0, i]
        obs_img = images[0, i]
        mask_map = coarse_masks[0, i]
        
        # 1. 投影
        w2c = torch.inverse(pose)
        R, T = w2c[:3, :3], w2c[:3, 3]
        pts_cam = (pts_world @ R.T) + T
        z_proj = pts_cam[:, 2]
        
        u = (pts_cam[:, 0] * K[0,0] / z_proj) + K[0,2]
        v = (pts_cam[:, 1] * K[1,1] / z_proj) + K[1,2]
        
        u_norm = 2 * (u / (W - 1)) - 1
        v_norm = 2 * (v / (H - 1)) - 1
        grid = torch.stack([u_norm, v_norm], dim=-1).view(1, 1, -1, 2)
        
        valid_uv = (u_norm > -1) & (u_norm < 1) & (v_norm > -1) & (v_norm < 1) & (z_proj > 0.1)
        
        # 2. 采样
        z_obs = F.grid_sample(obs_depth_map.view(1,1,H,W), grid, align_corners=True, padding_mode='border').view(-1)
        
        # [FIX] 使用 permute 正确采样颜色
        c_obs_raw = F.grid_sample(obs_img.unsqueeze(0), grid, align_corners=True) # (1, 3, 1, N)
        c_obs = c_obs_raw.squeeze(0).squeeze(1).permute(1, 0) # (N, 3)
        
        m_obs = F.grid_sample(mask_map.view(1,1,H,W).float(), grid, align_corners=True).view(-1)
        
        is_static = m_obs < 0.5
        valid = valid_uv & is_static
        
        if valid.sum() > 0:
            # A. 几何梯度 Norm (Eq. 10)
            res_geom = 0.5 * ((z_proj - z_obs) ** 2)
            grad_i = torch.autograd.grad(
                (res_geom * valid.float()).sum(), 
                pts_world, 
                retain_graph=True
            )[0]
            geom_score = grad_i.norm(dim=1)
            
            # B. 光度误差 (Eq. 11)
            photo_err = (c_ref - c_obs).norm(dim=1)
            
            # C. 组合 (Geom + Photo)
            score_i = geom_score * 10.0 + photo_err * 1.0
            
            total_score += score_i.detach() * valid.float()
            valid_count += valid.float()
            
    final_score = total_score / (valid_count + 1e-6)
    return final_score

# =========================================================================
# Part 4: 官方 Mask Refinement 策略 (3D SOR + Clustering)
# =========================================================================
def refine_mask_3d_official(coarse_mask_2d, grad_map_2d, pts_world, H, W):
    grad_vals = grad_map_2d.flatten()
    coarse_vals = coarse_mask_2d.flatten()
    
    # 候选点: Coarse激活 或 梯度高
    candidate_mask = (coarse_vals > 0.5) | (grad_vals > 0.3)
    candidate_indices = np.where(candidate_mask)[0]
    
    if len(candidate_indices) < 10:
        return np.zeros((H, W), dtype=np.float32)
        
    pts_cand = pts_world[candidate_indices]
    
    # --- A. SOR (Statistical Outlier Removal) ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_cand)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    valid_indices = candidate_indices[ind]
    
    if len(valid_indices) < 10:
        return np.zeros((H, W), dtype=np.float32)

    # --- B. Clustering (聚类) ---
    pts_valid = pts_world[valid_indices]
    clustering = DBSCAN(eps=0.5, min_samples=10).fit(pts_valid)
    labels = clustering.labels_
    grad_valid = grad_vals[valid_indices]
    
    final_indices = []
    unique_labels = set(labels)
    if -1 in unique_labels: unique_labels.remove(-1) 
    
    for label in unique_labels:
        cluster_mask = (labels == label)
        avg_grad = np.mean(grad_valid[cluster_mask])
        if avg_grad > 0.15: # 阈值
            final_indices.extend(valid_indices[cluster_mask])
            
    final_mask_flat = np.zeros(H * W, dtype=np.float32)
    final_mask_flat[final_indices] = 1.0
    
    final_mask_2d = final_mask_flat.reshape(H, W)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    final_mask_2d = cv2.morphologyEx(final_mask_2d, cv2.MORPH_CLOSE, kernel)
    
    return final_mask_2d

# =========================================================================
# Part 5: 主流程
# =========================================================================
def normalize_map(data):
    flat = data.flatten()
    lower, upper = np.percentile(flat, 5), np.percentile(flat, 95)
    return np.clip((data - lower) / (upper - lower + 1e-6), 0, 1)

@torch.no_grad()
def main(args):
    print(f"[1/5] Loading Model: {args.model_name}")
    load_source = args.model_path if args.model_path else f"depth-anything/{args.model_name.upper()}"
    model = DepthAnything3.from_pretrained(load_source).to(args.device)
    model.eval()
    
    img_files = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")) + glob.glob(os.path.join(args.image_dir, "*.png")))
    pil_imgs = [Image.open(p).convert("RGB") for p in img_files]
    num_frames = len(pil_imgs)
    print(f"Processing {num_frames} frames from {args.image_dir}")
    
    # 传入原始图片 Tensor
    imgs_tensor = torch.stack([
        torch.from_numpy(np.array(img)).permute(2,0,1).float()/255.0 
        for img in pil_imgs
    ]).to(args.device).unsqueeze(0)
    
    apply_patch(model.model, num_frames, window_size=args.window_size)
    
    # --- Pass 1 ---
    print("[2/5] Pass 1: Extracting Motion Cues...")
    for m in model.model.modules():
        if isinstance(m, Attention): m.collect_stats = True
            
    results_p1 = model.inference(pil_imgs, infer_gs=False)
    
    c2w_poses_np = process_poses(results_p1.extrinsics)
    depths = torch.from_numpy(results_p1.depth).to(args.device).unsqueeze(0)
    poses = torch.from_numpy(c2w_poses_np).float().to(args.device).unsqueeze(0)
    intrinsics = torch.from_numpy(results_p1.intrinsics).to(args.device).unsqueeze(0)
    
    # --- Gram Matrix ---
    print("[3/5] Generating Coarse Masks...")
    vit = model.model.da3.backbone.pretrained if hasattr(model.model, 'da3') else model.model.backbone.pretrained
    H_feat, W_feat = results_p1.depth.shape[1] // 14, results_p1.depth.shape[2] // 14
    
    accum_shallow, accum_middle, accum_deep = 0, 0, 0
    for idx, block in enumerate(vit.blocks):
        if not block.attn._captured_stats: continue
        stats = block.attn._captured_stats
        def get_map(name):
            d = stats[name].view(num_frames, -1)
            if d.shape[1] == (H_feat * W_feat + 1): d = d[:, 1:]
            d = d.view(num_frames, H_feat, W_feat)
            return F.interpolate(d.unsqueeze(1), size=results_p1.depth.shape[1:], mode='bilinear').squeeze(1).to(args.device)

        if idx == 1: accum_shallow += (1 - get_map("kk_mean")) * get_map("qk_var")
        if 6 <= idx <= 13: accum_middle += (1 - get_map("qq_mean"))
        if 30 <= idx <= 37: accum_deep += (1 - (1 - get_map("qq_var")) * get_map("qq_mean"))
        block.attn._captured_stats = {}

    w_s = normalize_map(accum_shallow.cpu().numpy())
    w_m = normalize_map(accum_middle.cpu().numpy())
    w_d = normalize_map(accum_deep.cpu().numpy())
    coarse_masks = (w_s * w_m * w_d > 0.5).astype(np.float32)
    
    # --- Refinement ---
    print("[4/5] Refining Masks (Strict 3D)...")
    refined_masks = []
    coarse_masks_t = torch.from_numpy(coarse_masks).to(args.device).unsqueeze(0)
    H, W = results_p1.depth.shape[1], results_p1.depth.shape[2]
    
    for t in tqdm(range(num_frames)):
        pts_cam = unproject_depth(depths[:, t], intrinsics[:, t], args.device).reshape(-1, 3)
        pose = poses[0, t]
        pts_world = (pts_cam @ pose[:3, :3].T) + pose[:3, 3] 
        
        with torch.enable_grad():
            grad_score = compute_projection_gradient_strict(
                pts_world, depths, poses, intrinsics, coarse_masks_t, imgs_tensor,
                H, W, args.window_size, t, num_frames
            )
        
        pts_world_np = pts_world.detach().cpu().numpy()
        grad_map_np = normalize_map(grad_score.view(H, W).cpu().numpy())
        coarse_mask_np = coarse_masks[t]
        
        final_mask = refine_mask_3d_official(coarse_mask_np, grad_map_np, pts_world_np, H, W)
        refined_masks.append(final_mask)
        
    refined_masks_np = np.stack(refined_masks).astype(np.float32)
    
    # --- Pass 2 ---
    print("[5/5] Pass 2: Masked Inference...")
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