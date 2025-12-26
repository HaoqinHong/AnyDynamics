import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import glob
import os
import gc
from tqdm import tqdm

# 确保项目根目录在 PYTHONPATH 中，以便引用 src
from depth_anything_3.api import DepthAnything3
import submodules.Concerto.concerto as concerto
from submodules.Concerto.concerto.transform import Compose

class IntegratedVideoDataset(Dataset):
    def __init__(self, 
                 video_dir, 
                 da3_model_path, 
                 concerto_model_path, 
                 dino_model_path, # 传入 'facebook/dinov2-base' 或 本地路径
                 voxel_size=0.02, 
                 device='cuda'):
        
        self.device = device
        self.image_paths = sorted(glob.glob(os.path.join(video_dir, "*.jpg")) + 
                                  glob.glob(os.path.join(video_dir, "*.png")))
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {video_dir}")
        self.num_frames = len(self.image_paths)
        print(f"[Dataset] Found {self.num_frames} frames. Starting Pipeline...")

        # 1. 运行 DA3 (提取几何 + 相机)
        self._run_da3_geometry(da3_model_path)
        
        # 2. 运行 Concerto (提取 Token)
        self._run_concerto(concerto_model_path, voxel_size)
        
        # 3. 运行 DINOv2 (预提取 GT Feature)
        self._extract_gt_features_dinov2(dino_model_path)

        print("[Dataset] Pipeline Done. Training Data Ready.")

    def _run_da3_geometry(self, model_path):
            print(f">> [1/3] Loading DA3 ({os.path.basename(model_path)})...")
            da3_model = DepthAnything3.from_pretrained(model_path, dynamic=True).to(self.device)
            da3_model.eval()

            print("   Generating Point Cloud...")
            with torch.no_grad():
                prediction = da3_model.inference(
                    self.image_paths, infer_gs=True, process_res=518, export_format="mini_npz"
                )
            
            # 1. 获取原始数据
            depths = torch.from_numpy(prediction.depth).to(self.device)
            intrinsics = torch.from_numpy(prediction.intrinsics).to(self.device)
            extrinsics_np = prediction.extrinsics
            
            # 补齐 3x4 -> 4x4
            if extrinsics_np.ndim == 3 and extrinsics_np.shape[1] == 3 and extrinsics_np.shape[2] == 4:
                N = extrinsics_np.shape[0]
                bottom_row = np.array([[[0, 0, 0, 1]]], dtype=extrinsics_np.dtype).repeat(N, axis=0)
                extrinsics_np = np.concatenate([extrinsics_np, bottom_row], axis=1)
            
            w2c_raw = torch.from_numpy(extrinsics_np).to(self.device).float()
            c2w_raw = torch.linalg.inv(w2c_raw)

            # 2. 生成原始世界坐标点云
            all_pts = []
            all_colors = []
            
            print("   Accumulating points & Calculating Center...")
            for i in range(self.num_frames):
                d = depths[i]
                K = intrinsics[i]
                c2w = c2w_raw[i] # 原始相机位姿
                
                img_raw = cv2.imread(self.image_paths[i])
                img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img_raw).to(self.device).float() / 255.0
                
                H, W = d.shape
                y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
                x, y, z = x.to(self.device).flatten(), y.to(self.device).flatten(), d.flatten()
                
                valid = (z > 0)
                # 降采样
                if valid.sum() > 40000:
                    indices = torch.nonzero(valid).squeeze()
                    idx = indices[torch.randperm(len(indices))[:40000]]
                else:
                    idx = torch.nonzero(valid).squeeze()
                
                if idx.numel() == 0: continue

                # Back-project
                x_s, y_s, z_s = x[idx], y[idx], z[idx]
                fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
                X_c = (x_s - cx) * z_s / fx
                Y_c = (y_s - cy) * z_s / fy
                Z_c = z_s
                pts_c = torch.stack([X_c, Y_c, Z_c], dim=-1)
                pts_w = (c2w[:3,:3] @ pts_c.T).T + c2w[:3,3]
                
                all_pts.append(pts_w)
                all_colors.append(img_tensor.flatten(0,1)[idx])
                
            raw_coords = torch.cat(all_pts, dim=0)
            self.big_colors = torch.cat(all_colors, dim=0)
            
            # =========================================================
            # 3. 核心修复：坐标系对齐 (Center Shift)
            # =========================================================
            # 计算整个场景的中心
            scene_center = raw_coords.mean(dim=0) # [3]
            print(f"   [Coordinate Fix] Shifting scene center from {scene_center.cpu().numpy()} to (0,0,0)")
            
            # A. 移动点云
            self.big_coords = raw_coords - scene_center
            
            # B. 移动相机 (修改 c2w 的平移部分)
            # c2w 矩阵的前3行第4列是平移向量 T
            c2w_fixed = c2w_raw.clone()
            c2w_fixed[:, :3, 3] -= scene_center
            
            # 重新计算 w2c (extrinsics)
            w2c_fixed = torch.linalg.inv(c2w_fixed)
            
            self.extrinsics = w2c_fixed.cpu() # 保存修正后的相机
            self.intrinsics = intrinsics.cpu()

            del da3_model
            torch.cuda.empty_cache()
            gc.collect()

    def _run_concerto(self, model_path, voxel_size):
        print(f">> [2/3] Loading Concerto ({os.path.basename(model_path)})...")
        concerto_model = concerto.model.load(model_path).to(self.device)
        concerto_model.eval()

        transform = Compose([
                    # dict(type="CenterShift", apply_z=True),  <-- 注释掉这一行！我们已经手动对齐了
                    dict(type="GridSample", grid_size=voxel_size, hash_type="fnv", mode="train",
                        return_grid_coord=True, return_inverse=True),
                    dict(type="ToTensor"),
                    dict(type="Collect", keys=("coord", "grid_coord", "inverse"), feat_keys=("coord", "color"))
                ])

        input_dict = {"coord": self.big_coords.cpu().numpy(), "color": self.big_colors.cpu().numpy()}
        input_dict = transform(input_dict)
        
        feat = input_dict["feat"]
        if feat.shape[1] == 6:
            input_dict["feat"] = torch.cat([feat, torch.zeros((feat.shape[0], 3), dtype=feat.dtype)], dim=1)

        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor): input_dict[k] = v.to(self.device)
        input_dict["offset"] = torch.tensor([input_dict["coord"].shape[0]], device=self.device)

        with torch.no_grad():
            output_point = concerto_model(input_dict)
            point = output_point
            while "pooling_parent" in point.keys():
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            
            self.scene_tokens = point.feat.cpu()
            self.scene_coords = input_dict["coord"].cpu()

        del concerto_model
        torch.cuda.empty_cache()
        gc.collect()

    def _extract_gt_features_dinov2(self, model_path):
        print(f">> [3/3] Extracting DINOv2 features for Loss...")
        
        # 临时导入 Loss 类来构建模型 (复用代码)
        from training.loss import DINOMetricLoss
        extractor = DINOMetricLoss(model_path=model_path, device=self.device)
        
        self.gt_feats_list = []
        
        for i in range(self.num_frames):
            img = cv2.imread(self.image_paths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img).permute(2,0,1).float().to(self.device) / 255.0
            
            with torch.no_grad():
                # preprocess 内部包含了 Resize 和 Normalize
                img_in = extractor.preprocess(img_tensor.unsqueeze(0))
                # HF transformers 输出
                outputs = extractor.dino(pixel_values=img_in)
                feat = outputs.last_hidden_state[:, 0, :] # CLS Token [1, 768]
                self.gt_feats_list.append(feat.cpu())
        
        del extractor
        torch.cuda.empty_cache()
        gc.collect()

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        t = torch.tensor([idx / (self.num_frames - 1)]).float()
        
        return {
            "tokens": self.scene_tokens,
            "coords": self.scene_coords,
            "t": t,
            "gt_image": img,
            "gt_feat": self.gt_feats_list[idx], # 预存的 CLS Token
            "c2w": torch.linalg.inv(self.extrinsics[idx]),
            "K": self.intrinsics[idx]
        }