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

from typing import Dict as TyDict
from typing import List, Sequence
import torch
import torch.nn as nn

from depth_anything_3.model.dpt import DPT
from depth_anything_3.model.utils.head_utils import activate_head_gs, custom_interpolate


class GSDPT(DPT):
# 输入维度 dim_in，patch 大小 patch_size，输出通道 output_dim（默认 4），主头激活 activation，置信度激活 conf_activation，
# 特征宽度 features，每级输出通道 out_channels，是否加位置编码 pos_embed，是否只返回特征 feature_only，
# 下采样倍率 down_ratio，置信度维度 conf_dim，规范化类型 norm_type，以及融合块是否 inplace。
    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 4,
        activation: str = "linear",
        conf_activation: str = "sigmoid",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        pos_embed: bool = True,
        feature_only: bool = False,
        down_ratio: int = 1,
        conf_dim: int = 1,
        norm_type: str = "idt",  # use to match legacy GS-DPT head, "idt" / "layer"
        fusion_block_inplace: bool = False,
    ) -> None:
        # 把这些配置传给基类 DPT，并指定 head_name="raw_gs"、不使用天空分支。
        super().__init__(
            dim_in=dim_in,
            patch_size=patch_size,
            output_dim=output_dim,
            activation=activation,
            conf_activation=conf_activation,
            features=features,
            out_channels=out_channels,
            pos_embed=pos_embed,
            down_ratio=down_ratio,
            head_name="raw_gs",
            use_sky_head=False,
            norm_type=norm_type,
            fusion_block_inplace=fusion_block_inplace,
        )
        self.conf_dim = conf_dim # 记录置信度维度。
        # 断言 conf_activation 必须是线性，因为多视角透明度需要线性输出。
        if conf_dim and conf_dim > 1:
            assert (
                conf_activation == "linear"
            ), "use linear prediction when using view-dependent opacity"

        # 决定后续图像融合分支的输出通道数。
        merger_out_dim = features if feature_only else features // 2

        # 三层 3x3 Conv+GELU，将原始 RGB 图像编码到与主分支同尺度的特征，通道逐步增加（1/4 → 1/2 → full）。
        self.images_merger = nn.Sequential(
            nn.Conv2d(3, merger_out_dim // 4, 3, 1, 1),  # fewer channels first
            nn.GELU(),
            nn.Conv2d(merger_out_dim // 4, merger_out_dim // 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(merger_out_dim // 2, merger_out_dim, 3, 1, 1),
            nn.GELU(),
        )

    # -------------------------------------------------------------------------
    # Internal forward (single chunk)
    # -------------------------------------------------------------------------
    # 处理单个 chunk 的内部前向，返回字典。
    # 输入：多尺度 feats 列表、原图高宽 H/W、当前 patch 起始索引 patch_start_idx、原始图像 images。
    def _forward_impl(
        self,
        feats: List[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
        images: torch.Tensor,
    ) -> TyDict[str, torch.Tensor]:
        B, _, C = feats[0].shape
        ph, pw = H // self.patch_size, W // self.patch_size
        resized_feats = []
        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            x = feats[take_idx][:, patch_start_idx:]  # [B*S, N_patch, C]
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape(B, C, ph, pw)  # [B*S, C, ph, pw]

            x = self.projects[stage_idx](x)
            if self.pos_embed:
                x = self._add_pos_embed(x, W, H)
            x = self.resize_layers[stage_idx](x)  # Align scale
            resized_feats.append(x)

        # 2) Fusion pyramid (main branch only)
        fused = self._fuse(resized_feats)
        fused = self.scratch.output_conv1(fused)

        # 3) Upsample to target resolution, optionally add position encoding again
        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)

        fused = custom_interpolate(fused, (h_out, w_out), mode="bilinear", align_corners=True)

        # inject the image information here
        fused = fused + self.images_merger(images)

        if self.pos_embed:
            fused = self._add_pos_embed(fused, W, H)

        # 4) Shared neck1
        # feat = self.scratch.output_conv1(fused)
        feat = fused

        # 5) Main head: logits -> activate_head or single channel activation
        main_logits = self.scratch.output_conv2(feat)
        outs: TyDict[str, torch.Tensor] = {}
        if self.has_conf:
            pred, conf = activate_head_gs(
                main_logits,
                activation=self.activation,
                conf_activation=self.conf_activation,
                conf_dim=self.conf_dim,
            )
            outs[self.head_main] = pred.squeeze(1)
            outs[f"{self.head_main}_conf"] = conf.squeeze(1)
        else:
            outs[self.head_main] = self._apply_activation_single(main_logits).squeeze(1)

        return outs
