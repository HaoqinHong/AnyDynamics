# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from einops import rearrange # 新增引用
import torch.nn.functional as F

from __future__ import annotations

import torch
import torch.nn as nn
from addict import Dict
from omegaconf import DictConfig, OmegaConf

from depth_anything_3.cfg import create_object
from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
from depth_anything_3.utils.alignment import (
    apply_metric_scaling,
    compute_alignment_mask,
    compute_sky_mask,
    least_squares_scale_scalar,
    sample_tensor_for_quantile,
    set_sky_regions_to_max_depth,
)
from depth_anything_3.utils.geometry import affine_inverse, as_homogeneous, map_pdf_to_opacity


def _wrap_cfg(cfg_obj):
    return OmegaConf.create(cfg_obj)


class DepthAnything3Net(nn.Module):
    """
    Depth Anything 3 network for depth estimation and camera pose estimation.

    This network consists of:
    - Backbone: DinoV2 feature extractor
    - Head: DPT or DualDPT for depth prediction
    - Optional camera decoders for pose estimation
    - Optional GSDPT for 3DGS prediction

    Args:
        preset: Configuration preset containing network dimensions and settings

    Returns:
        Dictionary containing:
        - depth: Predicted depth map (B, H, W)
        - depth_conf: Depth confidence map (B, H, W)
        - extrinsics: Camera extrinsics (B, N, 4, 4)
        - intrinsics: Camera intrinsics (B, N, 3, 3)
        - gaussians: 3D Gaussian Splats (world space), type: model.gs_adapter.Gaussians
        - aux: Auxiliary features for specified layers
    """

    # Patch size for feature extraction
    PATCH_SIZE = 14

    def __init__(self, net, head, cam_dec=None, cam_enc=None, gs_head=None, gs_adapter=None):
        """
        Initialize DepthAnything3Net with given yaml-initialized configuration.
        """
        super().__init__()
        self.backbone = net if isinstance(net, nn.Module) else create_object(_wrap_cfg(net))
        self.head = head if isinstance(head, nn.Module) else create_object(_wrap_cfg(head))
        self.cam_dec, self.cam_enc = None, None
        if cam_dec is not None:
            self.cam_dec = (
                cam_dec if isinstance(cam_dec, nn.Module) else create_object(_wrap_cfg(cam_dec))
            )
            self.cam_enc = (
                cam_dec if isinstance(cam_enc, nn.Module) else create_object(_wrap_cfg(cam_enc))
            )
        self.gs_adapter, self.gs_head = None, None
        if gs_head is not None and gs_adapter is not None:
            self.gs_adapter = (
                gs_adapter
                if isinstance(gs_adapter, nn.Module)
                else create_object(_wrap_cfg(gs_adapter))
            )
            gs_out_dim = self.gs_adapter.d_in + 1
            if isinstance(gs_head, nn.Module):
                assert (
                    gs_head.out_dim == gs_out_dim
                ), f"gs_head.out_dim should be {gs_out_dim}, got {gs_head.out_dim}"
                self.gs_head = gs_head
            else:
                assert (
                    gs_head["output_dim"] == gs_out_dim
                ), f"gs_head output_dim should set to {gs_out_dim}, got {gs_head['output_dim']}"
                self.gs_head = create_object(_wrap_cfg(gs_head))

    def forward(
        self,
        x: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = [],
        infer_gs: bool = False,
        # --- [New] VGGT4D 开关 ---
        enable_vggt4d: bool = False, 
        mask_layer_indices: list[int] = [0, 1, 2, 3, 4], # 默认在浅层 Mask
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional VGGT4D dynamic masking.
        """
        # 准备 Camera Token (逻辑不变)
        if extrinsics is not None:
            with torch.autocast(device_type=x.device.type, enabled=False):
                cam_token = self.cam_enc(extrinsics, intrinsics, x.shape[-2:])
        else:
            cam_token = None

        H, W = x.shape[-2], x.shape[-1]
        dynamic_mask = None

        # === [Stage 1] VGGT4D 挖掘阶段 ===
        if enable_vggt4d:
            # 这一步不需要梯度，只为了求 Mask
            with torch.no_grad():
                # 调用 backbone，请求 extract_motion_cues (需配合上一轮修改的 vision_transformer.py)
                # 注意：这里我们利用 DA3 的 Global Layers (L9, L11...) 来挖掘运动线索
                _, _, motion_cues = self.backbone(
                    x, 
                    cam_token=cam_token, 
                    extract_motion_cues=True 
                )
                
                # 计算掩码 (Gram Matrix -> Mean/Var -> Threshold)
                # motion_cues['qs'] 包含了 Global Layers 的 Query
                dynamic_mask = self._compute_vggt4d_mask(motion_cues, H, W)
                
                # [可选] 可视化或调试 Mask
                # import matplotlib.pyplot as plt; plt.imshow(dynamic_mask[0,0].cpu()); plt.show()

        # === [Stage 2] 正式推理阶段 ===
        # 如果 enable_vggt4d 为 True，这里会将计算好的 mask 传入
        feats, aux_feats = self.backbone(
            x, 
            cam_token=cam_token, 
            export_feat_layers=export_feat_layers,
            # 传入 Mask 参数 (需配合上一轮修改的 vision_transformer.py)
            dynamic_mask=dynamic_mask,
            mask_layers=mask_layer_indices
        )

        # feats = [[item for item in feat] for feat in feats]
        H, W = x.shape[-2], x.shape[-1]

        # Process features through depth head
        with torch.autocast(device_type=x.device.type, enabled=False):
            output = self._process_depth_head(feats, H, W)
            output = self._process_camera_estimation(feats, H, W, output)
            if infer_gs:
                output = self._process_gs_head(feats, H, W, output, x, extrinsics, intrinsics)

        output.aux = self._extract_auxiliary_features(aux_feats, export_feat_layers, H, W)
        
        # 将 Mask 也存入输出，方便调试
        if dynamic_mask is not None:
            output['dynamic_mask'] = dynamic_mask

        return output

    def _process_depth_head(
        self, feats: list[torch.Tensor], H: int, W: int
    ) -> Dict[str, torch.Tensor]:
        """Process features through the depth prediction head."""
        return self.head(feats, H, W, patch_start_idx=0)

    def _process_camera_estimation(
        self, feats: list[torch.Tensor], H: int, W: int, output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Process camera pose estimation if camera decoder is available."""
        if self.cam_dec is not None:
            pose_enc = self.cam_dec(feats[-1][1])
            # Remove ray information as it's not needed for pose estimation
            if "ray" in output:
                del output.ray
            if "ray_conf" in output:
                del output.ray_conf

            # Convert pose encoding to extrinsics and intrinsics
            c2w, ixt = pose_encoding_to_extri_intri(pose_enc, (H, W))
            output.extrinsics = affine_inverse(c2w)
            output.intrinsics = ixt

        return output

    def _process_gs_head(
        self,
        feats: list[torch.Tensor],
        H: int,
        W: int,
        output: Dict[str, torch.Tensor],
        in_images: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Process 3DGS parameters estimation if 3DGS head is available."""
        if self.gs_head is None or self.gs_adapter is None:
            return output
        assert output.get("depth", None) is not None, "must provide MV depth for the GS head."

        # The depth is defined in the DA3 model's camera space,
        # so even with provided GT camera poses,
        # we instead use the predicted camera poses for better alignment.
        ctx_extr = output.get("extrinsics", None)
        ctx_intr = output.get("intrinsics", None)
        assert (
            ctx_extr is not None and ctx_intr is not None
        ), "must process camera info first if GT is not available"

        gt_extr = extrinsics
        # homo the extr if needed
        ctx_extr = as_homogeneous(ctx_extr)
        if gt_extr is not None:
            gt_extr = as_homogeneous(gt_extr)

        # forward through the gs_dpt head to get 'camera space' parameters
        gs_outs = self.gs_head(
            feats=feats,
            H=H,
            W=W,
            patch_start_idx=0,
            images=in_images,
        )
        raw_gaussians = gs_outs.raw_gs
        densities = gs_outs.raw_gs_conf

        # convert to 'world space' 3DGS parameters; ready to export and render
        # gt_extr could be None, and will be used to align the pose scale if available
        gs_world = self.gs_adapter(
            extrinsics=ctx_extr,
            intrinsics=ctx_intr,
            depths=output.depth,
            opacities=map_pdf_to_opacity(densities),
            raw_gaussians=raw_gaussians,
            image_shape=(H, W),
            gt_extrinsics=gt_extr,
        )
        output.gaussians = gs_world

        return output

    def _extract_auxiliary_features(
        self, feats: list[torch.Tensor], feat_layers: list[int], H: int, W: int
    ) -> Dict[str, torch.Tensor]:
        """Extract auxiliary features from specified layers."""
        aux_features = Dict()
        assert len(feats) == len(feat_layers)
        for feat, feat_layer in zip(feats, feat_layers):
            # Reshape features to spatial dimensions
            feat_reshaped = feat.reshape(
                [
                    feat.shape[0],
                    feat.shape[1],
                    H // self.PATCH_SIZE,
                    W // self.PATCH_SIZE,
                    feat.shape[-1],
                ]
            )
            aux_features[f"feat_layer_{feat_layer}"] = feat_reshaped

        return aux_features

    def _compute_vggt4d_mask(self, motion_cues, H, W, threshold_ratio=0.6):
            """
            根据 VGGT4D 论文公式 (3)-(8) 计算动态掩码。
            这里主要利用 Global Layers 的 Q/K 计算跨帧一致性。
            """
            qs = motion_cues['qs'] # List of [B, H, S, N, D] from Global Layers
            ks = motion_cues['ks'] 
            
            B, num_heads, S, N, D = qs[0].shape
            device = qs[0].device
            
            # 初始化聚合的 S (Mean) 和 V (Variance)
            # 对应论文 Eq(3) Eq(4)
            agg_score = 0
            
            # 遍历捕获到的 Global Layers (对应 VGGT4D 的 Middle/Deep Layers)
            for q, k in zip(qs, ks):
                # 1. 计算 Gram 矩阵 A_QQ = Q * Q^T
                # 我们只关心当前帧 t 和其他帧 s 的相似度
                # shape: [B, H, S(Ref), S(Src), N(Ref), N(Src)] -> 太大了
                # 优化：VGGT4D 实际上是计算 Token-wise 的跨帧相关性
                
                # 简化版实现：计算每个 Token 在时间轴上的特征方差
                # 如果一个 Token 是静态背景，它在 Global Layer 的 Feature (Q) 应该在所有帧间保持稳定
                # 如果是动态物体，它的 Q 会剧烈变化
                
                # 论文使用的是 Gram Matrix 的统计量。
                # 这里我们计算 Query 向量在时间窗口内的自相似度 (Self-Similarity over Time)
                
                # [B, H, S, N, D] -> [B, H, N, S, D]
                q_per_token = rearrange(q, 'b h s n d -> (b h n) s d')
                k_per_token = rearrange(k, 'b h s n d -> (b h n) s d')
                
                # 计算 Gram Matrix: [Tokens, S, S]
                # 这一步衡量了该 Token 在不同帧之间的相似性
                gram_qq = torch.bmm(q_per_token, q_per_token.transpose(1, 2)) / (D ** 0.5)
                
                # 提取非对角线元素 (即跨帧相似度)
                # 静态物体：跨帧相似度高 -> Mean 高, Var 低
                # 动态物体：跨帧相似度低 -> Mean 低, Var 高 (或不稳定)
                
                # 论文 Eq 7: w_middle = 1 - S_middle_QQ (S 是 Mean)
                # 意味着：相似度均值越低，越可能是动态物体
                
                mask = ~torch.eye(S, device=device).bool() # 排除自身帧
                gram_qq_off_diag = gram_qq[:, mask].reshape(gram_qq.shape[0], -1)
                
                s_qq = gram_qq_off_diag.mean(dim=-1) # Mean over time window
                
                # 归一化到 0-1
                s_qq = (s_qq - s_qq.min()) / (s_qq.max() - s_qq.min() + 1e-6)
                
                # 动态得分 = 1 - 静态得分
                layer_dynamic_score = 1.0 - s_qq
                
                # 累加多层的得分
                agg_score += layer_dynamic_score.view(B, num_heads, N).mean(dim=1) # Average over heads

            # 平均所有层的得分
            agg_score /= len(qs)
            
            # [B, S, N] -> [B, S, H, W] (Reshape back to image)
            patch_h, patch_w = H // self.PATCH_SIZE, W // self.PATCH_SIZE
            saliency_map = agg_score.reshape(B, S, patch_h, patch_w)
            
            # 上采样回原图尺寸
            saliency_map = F.interpolate(
                saliency_map.reshape(B * S, 1, patch_h, patch_w), 
                size=(H, W), 
                mode='bilinear'
            ).reshape(B, S, H, W)
            
            # 简单的阈值处理 (论文用了 Otsu，这里用分位数简化)
            # 大于阈值的被认为是动态物体
            threshold = torch.quantile(saliency_map.flatten(), threshold_ratio)
            binary_mask = (saliency_map > threshold).float()
            
            # 生成 attention mask
            # 如果 binary_mask为1 (动态)，我们需要屏蔽它，所以在 Attention Mask 中设为 -inf 或 True
            # 具体取决于 vision_transformer.py 如何处理 mask。
            # 通常 attn_mask: 0 for keep, -inf for mask out. 或者 boolean mask.
            # 这里假设传入的是 Boolean Mask，True 表示要屏蔽 (Is Dynamic)
            return binary_mask.bool()


class NestedDepthAnything3Net(nn.Module):
    """
    Nested Depth Anything 3 network with metric scaling capabilities.

    This network combines two DepthAnything3Net branches:
    - Main branch: Standard depth estimation
    - Metric branch: Metric depth estimation for scaling alignment

    The network performs depth alignment using least squares scaling
    and handles sky region masking for improved depth estimation.

    Args:
        preset: Configuration for the main depth estimation branch
        second_preset: Configuration for the metric depth branch
    """

    def __init__(self, anyview: DictConfig, metric: DictConfig):
        """
        Initialize NestedDepthAnything3Net with two branches.

        Args:
            preset: Configuration for main depth estimation branch
            second_preset: Configuration for metric depth branch
        """
        super().__init__()
        self.da3 = create_object(anyview)
        self.da3_metric = create_object(metric)

    def forward(
        self,
        x: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = [],
        infer_gs: bool = False,
        # --- [New] 透传参数 ---
        enable_vggt4d: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through both branches with metric scaling alignment.

        Args:
            x: Input images (B, N, 3, H, W)
            extrinsics: Camera extrinsics (B, N, 4, 4) - unused
            intrinsics: Camera intrinsics (B, N, 3, 3) - unused
            feat_layers: List of layer indices to extract features from
            metric_feat: Whether to use metric features (unused)

        Returns:
            Dictionary containing aligned depth predictions and camera parameters
        """
        # Get predictions from both branches
        output = self.da3(
            x, extrinsics, intrinsics, 
            export_feat_layers=export_feat_layers, 
            infer_gs=infer_gs,
            # 透传
            enable_vggt4d=enable_vggt4d 
        )
        
        # metric_output 通常不需要 VGGT4D Mask，因为它只负责 Scale
        metric_output = self.da3_metric(x, infer_gs=infer_gs)

        # Apply metric scaling and alignment
        output = self._apply_metric_scaling(output, metric_output)
        output = self._apply_depth_alignment(output, metric_output)
        output = self._handle_sky_regions(output, metric_output)

        return output

    def _apply_metric_scaling(
        self, output: Dict[str, torch.Tensor], metric_output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply metric scaling to the metric depth output."""
        # Scale metric depth based on camera intrinsics
        metric_output.depth = apply_metric_scaling(
            metric_output.depth,
            output.intrinsics,
        )
        return output

    def _apply_depth_alignment(
        self, output: Dict[str, torch.Tensor], metric_output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply depth alignment using least squares scaling."""
        # Compute non-sky mask
        non_sky_mask = compute_sky_mask(metric_output.sky, threshold=0.3)

        # Ensure we have enough non-sky pixels
        assert non_sky_mask.sum() > 10, "Insufficient non-sky pixels for alignment"

        # Sample depth confidence for quantile computation
        depth_conf_ns = output.depth_conf[non_sky_mask]
        depth_conf_sampled = sample_tensor_for_quantile(depth_conf_ns, max_samples=100000)
        median_conf = torch.quantile(depth_conf_sampled, 0.5)

        # Compute alignment mask
        align_mask = compute_alignment_mask(
            output.depth_conf, non_sky_mask, output.depth, metric_output.depth, median_conf
        )

        # Compute scale factor using least squares
        valid_depth = output.depth[align_mask]
        valid_metric_depth = metric_output.depth[align_mask]
        scale_factor = least_squares_scale_scalar(valid_metric_depth, valid_depth)

        # Apply scaling to depth and extrinsics
        output.depth *= scale_factor
        output.extrinsics[:, :, :3, 3] *= scale_factor
        output.is_metric = 1
        output.scale_factor = scale_factor.item()

        return output

    def _handle_sky_regions(
        self,
        output: Dict[str, torch.Tensor],
        metric_output: Dict[str, torch.Tensor],
        sky_depth_def: float = 200.0,
    ) -> Dict[str, torch.Tensor]:
        """Handle sky regions by setting them to maximum depth."""
        non_sky_mask = compute_sky_mask(metric_output.sky, threshold=0.3)

        # Compute maximum depth for non-sky regions
        # Use sampling to safely compute quantile on large tensors
        non_sky_depth = output.depth[non_sky_mask]
        if non_sky_depth.numel() > 100000:
            idx = torch.randint(0, non_sky_depth.numel(), (100000,), device=non_sky_depth.device)
            sampled_depth = non_sky_depth[idx]
        else:
            sampled_depth = non_sky_depth
        non_sky_max = min(torch.quantile(sampled_depth, 0.99), sky_depth_def)

        # Set sky regions to maximum depth and high confidence
        output.depth, output.depth_conf = set_sky_regions_to_max_depth(
            output.depth, output.depth_conf, non_sky_mask, max_depth=non_sky_max
        )

        return output
