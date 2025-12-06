import argparse
import glob
import os
import shutil
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from depth_anything_3.api import DepthAnything3

# ==========================================
# 1. Utils: Saving & Helper Functions
# ==========================================
def as_homogeneous_numpy(poses):
    """Convert [N, 3, 4] to [N, 4, 4]"""
    if poses.shape[-2:] == (4, 4): return poses
    N = poses.shape[0]
    bottom = np.array([0, 0, 0, 1]).reshape(1, 1, 4).repeat(N, axis=0)
    return np.concatenate([poses, bottom], axis=1)

def save_tum_poses(output_dir, poses, filenames=None):
    output_path = os.path.join(output_dir, "poses.txt")
    # Fix: Ensure 4x4 matrix for inversion
    poses_4x4 = as_homogeneous_numpy(poses)
    
    with open(output_path, "w") as f:
        for i, pose in enumerate(poses_4x4):
            # World-to-Camera (DA3) -> Camera-to-World (TUM)
            try:
                c2w = np.linalg.inv(pose)
            except np.linalg.LinAlgError:
                print(f"Warning: Singular matrix at frame {i}, skipping inversion.")
                c2w = np.eye(4)

            t = c2w[:3, 3]
            r = Rotation.from_matrix(c2w[:3, :3])
            q = r.as_quat() # x, y, z, w
            timestamp = i if filenames is None else os.path.splitext(os.path.basename(filenames[i]))[0]
            f.write(f"{timestamp} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")

def save_intrinsic_txt(output_dir, intrinsics):
    with open(os.path.join(output_dir, "intrinsic.txt"), "w") as f:
        K = intrinsics[0]
        f.write(f"{K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]}\n")

def save_results(output_dir, image_paths, pred_depth, pred_conf, pred_poses, refined_mask=None):
    for sub in ["depth", "depth_conf", "dynamic_mask", "rgb"]:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)
    
    save_intrinsic_txt(output_dir, pred_poses.intrinsics)
    save_tum_poses(output_dir, pred_poses.extrinsics, image_paths)
    
    for i, path in enumerate(image_paths):
        name = os.path.splitext(os.path.basename(path))[0]
        shutil.copy(path, os.path.join(output_dir, "rgb", os.path.basename(path)))
        cv2.imwrite(os.path.join(output_dir, "depth", f"{name}.png"), (pred_depth[i] * 1000).astype(np.uint16))
        cv2.imwrite(os.path.join(output_dir, "depth_conf", f"{name}.png"), (pred_conf[i] * 255).astype(np.uint8))
        if refined_mask is not None:
            cv2.imwrite(os.path.join(output_dir, "dynamic_mask", f"{name}.png"), (refined_mask[i].astype(np.uint8) * 255))

def adaptive_multiotsu_threshold(img_batch):
    """[S, H, W] -> [S, H, W] binary mask"""
    if isinstance(img_batch, torch.Tensor): img_batch = img_batch.cpu().numpy()
    masks = []
    for img in img_batch:
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if img_norm.max() == img_norm.min():
            masks.append(np.zeros_like(img_norm))
        else:
            try:
                _, mask = cv2.threshold(img_norm, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                masks.append(mask)
            except:
                masks.append(np.zeros_like(img_norm))
    return np.array(masks)

# ==========================================
# 2. VGGT4D Plugin (Stage 1 & 2)
# ==========================================
class VGGT4DPlugin:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.motion_cues = {"qs": []}
        self.dynamic_mask = None
        self.patch_size = 14

    def _clear_hooks(self):
        for h in self.hooks: h.remove()
        self.hooks = []

    def _mining_hook(self, module, args, output):
        x = args[0]
        B, N, C = x.shape
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
        q = module.q_norm(qkv[0])
        self.motion_cues["qs"].append(q.detach())

    def _masking_hook(self, module, args, kwargs):
        if self.dynamic_mask is not None:
            # Injecting mask into Local Attention layers
            kwargs['attn_mask'] = self.dynamic_mask
        return args, kwargs

    def _compute_gram_mask(self, H, W):
        qs = self.motion_cues['qs']
        if not qs: return None, None
        
        q_example = qs[0]
        B_batch, n_heads, n_tokens, dim = q_example.shape
        
        n_patches = (H // self.patch_size) * (W // self.patch_size)
        S = int(round(n_tokens / n_patches))
        
        if S <= 1: return None, None

        n_tokens_per_frame = n_tokens // S
        start_idx = 1 if (n_tokens_per_frame > n_patches) else 0
        
        agg_score = 0
        for q in qs:
            try:
                q = rearrange(q, 'b h (s n_full) d -> b h s n_full d', s=S, n_full=n_tokens_per_frame)
            except: return None, None

            # 1. Strip & Keep patches
            q_patches = q[:, :, :, start_idx:, :] # [B, H, S, Np, D]
            if q_patches.shape[3] != n_patches: q_patches = q_patches[:, :, :, :n_patches, :]

            # 2. Gram Matrix
            q_patches = F.normalize(q_patches, p=2, dim=-1)
            q_patches = rearrange(q_patches, 'b h s n d -> b h n s d')
            gram_qq = torch.matmul(q_patches, q_patches.transpose(-1, -2)) # [B, H, Np, S, S]
            
            # 3. Temporal Consistency
            sum_sim = gram_qq.sum(dim=-1) - 1.0 
            s_qq = sum_sim / (S - 1) # [B, H, Np, S]
            w_middle = 1.0 - s_qq 
            agg_score += w_middle.mean(dim=1) # [B, Np, S]

        agg_score /= len(qs)
        
        # 4. Process Spatial Map
        agg_score = rearrange(agg_score, 'b n s -> b s n')
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        saliency_map = agg_score.reshape(B_batch, S, patch_h, patch_w)
        
        full_res_saliency = F.interpolate(
            rearrange(saliency_map, 'b s h w -> (b s) 1 h w'), 
            size=(H, W), mode='bilinear', align_corners=False
        ).squeeze(1) # [B*S, H, W]
        
        # 5. Thresholding
        binary_mask_np = adaptive_multiotsu_threshold(full_res_saliency)
        binary_mask = torch.from_numpy(binary_mask_np).to(q.device).bool() # [B*S, H, W]
        
        # 6. Construct Attention Mask
        # We need [B*S, 1, Nf] for Local Attention
        mask_patches = F.interpolate(binary_mask.float().unsqueeze(1), size=(patch_h, patch_w), mode='nearest')
        mask_patches = rearrange(mask_patches, '(b s) 1 h w -> b s (h w)', b=B_batch) # [B, S, Np]
        
        # Add Extra Token (Unmasked)
        if start_idx > 0:
            extra = torch.zeros((B_batch, S, start_idx), device=q.device).bool()
            mask_flat = torch.cat([extra, mask_patches > 0.5], dim=2)
        else:
            mask_flat = (mask_patches > 0.5)
            
        # [CRITICAL FIX]: Reshape to [B*S, 1, Total_Tokens]
        # Attention.py expects: [Batch, 1, Seq_Len]
        # Then it does [:, None] -> [Batch, 1, 1, Seq_Len] -> Repeat heads
        mask_final = rearrange(mask_flat, 'b s n -> (b s) n') # [50, 1297]
        attn_mask = mask_final.unsqueeze(1) # [50, 1, 1297]
        
        # Return binary_mask in [S, H, W] for refinement
        return attn_mask, binary_mask.view(B_batch, S, H, W)[0]

    @torch.no_grad()
    def mining_stage(self, images, extrinsics, intrinsics):
        H, W = images.shape[-2:]
        self.motion_cues = {"qs": []}
        self.dynamic_mask = None
        self._clear_hooks()
        
        if hasattr(self.model.backbone, 'pretrained'): transformer = self.model.backbone.pretrained
        else: transformer = self.model.backbone
        alt_start = getattr(self.model.backbone, 'alt_start', getattr(transformer, 'alt_start', -1))
        
        if alt_start != -1:
            for i, blk in enumerate(transformer.blocks):
                if i >= alt_start and i % 2 == 1: 
                    handle = blk.attn.register_forward_hook(self._mining_hook)
                    self.hooks.append(handle)
        
        out1 = self.model(images, extrinsics, intrinsics)
        attn_mask, binary_mask = self._compute_gram_mask(H, W)
        self._clear_hooks()
        return out1, attn_mask, binary_mask

    @torch.no_grad()
    def masking_stage(self, images, extrinsics, intrinsics, attn_mask):
        self._clear_hooks()
        self.dynamic_mask = attn_mask
        
        if hasattr(self.model.backbone, 'pretrained'): transformer = self.model.backbone.pretrained
        else: transformer = self.model.backbone
        
        for i in range(5):
            blk = transformer.blocks[i]
            handle = blk.attn.register_forward_pre_hook(self._masking_hook, with_kwargs=True)
            self.hooks.append(handle)
            
        out2 = self.model(images, extrinsics, intrinsics)
        self._clear_hooks()
        return out2

# ==========================================
# 3. Geometry Refiner (Stage 3)
# ==========================================
class RefineDynMask:
    def __init__(self, depths, initial_masks, extrinsics, intrinsics, device):
        self.depths = depths.squeeze(0) # [S, H, W]
        self.initial_masks = initial_masks # [S, H, W]
        
        # Ensure extrinsics are 4x4
        if extrinsics.shape[-2:] == (3, 4):
            filler = torch.tensor([0,0,0,1], device=device).reshape(1,1,4).repeat(extrinsics.shape[1],1,1)
            self.extrinsics = torch.cat([extrinsics, filler], dim=2).squeeze(0)
        else:
            self.extrinsics = extrinsics.squeeze(0)
            
        self.intrinsics = intrinsics.squeeze(0)
        self.device = device
        self.S, self.H, self.W = self.depths.shape
        
        y, x = torch.meshgrid(torch.arange(self.H, device=device), torch.arange(self.W, device=device), indexing='ij')
        self.coords = torch.stack([x, y, torch.ones_like(x)], dim=0).reshape(3, -1).float()

    def refine_masks(self, depth_thres=0.1):
        refined_masks = self.initial_masks.clone().float()
        
        for t in tqdm(range(self.S), desc="Refining"):
            try:
                K_inv = torch.linalg.inv(self.intrinsics[t])
                E_inv = torch.linalg.inv(self.extrinsics[t]) # Cam -> World
            except: continue # Skip failed inversions
            
            d_t = self.depths[t].reshape(-1)
            P_cam = (K_inv @ self.coords) * d_t
            P_world = E_inv[:3, :3] @ P_cam + E_inv[:3, 3:4]
            
            error_acc = torch.zeros_like(d_t)
            count = torch.zeros_like(d_t)
            
            for tn in [t-1, t+1]:
                if tn < 0 or tn >= self.S: continue
                
                En = self.extrinsics[tn] # World -> Neighbor
                Kn = self.intrinsics[tn]
                Dn = self.depths[tn]
                
                P_cam_n = En[:3, :3] @ P_world + En[:3, 3:4]
                z_n = P_cam_n[2, :]
                uv_n = Kn @ P_cam_n
                u, v = uv_n[0] / (z_n + 1e-6), uv_n[1] / (z_n + 1e-6)
                
                u_norm = 2 * u / (self.W - 1) - 1
                v_norm = 2 * v / (self.H - 1) - 1
                grid = torch.stack([u_norm, v_norm], dim=-1).view(1, 1, -1, 2)
                
                d_sampled = F.grid_sample(Dn.view(1, 1, self.H, self.W), grid, align_corners=True).reshape(-1)
                
                mask = (z_n > 0.1) & (u_norm.abs() <= 1) & (v_norm.abs() <= 1)
                err = torch.abs(z_n - d_sampled) / (d_sampled + 1e-4)
                
                error_acc[mask] += err[mask]
                count[mask] += 1
                
            avg_err = error_acc / (count + 1e-6)
            geo_mask = (avg_err > depth_thres).view(self.H, self.W).float()
            refined_masks[t] = torch.max(refined_masks[t], geo_mask)
            
        return refined_masks.bool().cpu().numpy()

# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if not os.path.exists(args.image_dir): raise FileNotFoundError("Image dir not found")
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading model: {args.model_path}")
    model_wrapper = DepthAnything3.from_pretrained(args.model_path).to(device)
    vggt4d = VGGT4DPlugin(model_wrapper.model)

    image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*")))
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_paths: return
    
    print(f"Processing {len(image_paths)} images")
    imgs_cpu, exts, ints = model_wrapper._preprocess_inputs(image_paths)
    imgs_tensor, ex_t, in_t = model_wrapper._prepare_model_inputs(imgs_cpu, exts, ints)
    ex_t_norm = model_wrapper._normalize_extrinsics(ex_t)

    # Stage 1
    print("\nStage 1: Mining Motion Cues...")
    out1, attn_mask, init_mask = vggt4d.mining_stage(imgs_tensor, ex_t_norm, in_t)
    if attn_mask is None:
        print("Failed to mine cues. Exiting.")
        return
    pred1 = model_wrapper._convert_to_prediction(out1)
    pred1 = model_wrapper._align_to_input_extrinsics_intrinsics(exts, ints, pred1)

    # Stage 2
    print("\nStage 2: Masked Inference...")
    out2 = vggt4d.masking_stage(imgs_tensor, ex_t_norm, in_t, attn_mask)
    pred2 = model_wrapper._convert_to_prediction(out2)
    pred2 = model_wrapper._align_to_input_extrinsics_intrinsics(exts, ints, pred2)

    # Stage 3
    print("\nStage 3: Refinement...")
    try:
        refiner = RefineDynMask(
            torch.from_numpy(pred1.depth).unsqueeze(0).to(device),
            init_mask,
            torch.from_numpy(pred2.extrinsics).unsqueeze(0).float().to(device),
            torch.from_numpy(pred2.intrinsics).unsqueeze(0).float().to(device),
            device
        )
        final_masks = refiner.refine_masks()
    except Exception as e:
        print(f"Refinement error: {e}")
        final_masks = init_mask.cpu().numpy()

    # Save
    print(f"Saving to {args.output_dir}")
    save_results(args.output_dir, image_paths, pred1.depth, pred1.conf, pred2, final_masks)
    print("Done!")

if __name__ == "__main__":
    main()