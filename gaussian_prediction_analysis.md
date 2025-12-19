结合 GSDPT (解码头) 和 GaussianAdapter (适配器) 的代码，Depth Anything 3 (DA3) 生成 3DGS 的流程可以分为两个主要阶段：图像空间参数预测 和 世界空间参数转换。


1. 整体流程概览
DA3 不直接预测世界坐标下的 3D 高斯球，而是先在相机/图像空间预测相对于像素和深度的局部属性（如相对位移、局部旋转、相对于深度的尺度），然后结合预测的深度图 (Depth) 和相机位姿 (Pose) 将其 "反投影" 并转换到世界坐标系。

流程如下：
- Backbone: 提取多尺度特征。
- Depth Head: 先预测出一张深度图 depth。
- GSDPT: 利用特征和原始 RGB 图像，预测出每个像素点的 "原始高斯参数" (raw_gaussians) 和 "不透明度/密度" (conf)。
- GaussianAdapter: 接收 raw_gaussians、depth、相机内参 (intrinsics) 和外参 (extrinsics)，通过几何计算将参数提升到 3D 世界空间。

-------

2. GSDPT：从特征到原始参数 (Image Space)
代码文件: src/depth_anything_3/model/gsdpt.py

它的核心任务是像分割或深度估计一样，输出一张多通道的特征图，其中每个通道代表高斯球的一个属性。

输入融合:
    继承自 DPT，融合 Backbone 的多尺度特征 (feats)。
    图像注入: 通过 self.images_merger(images) 将原始 RGB 图像编码后加到融合特征中，以保留高频纹理细节。
输出层:
    通过 self.scratch.output_conv2 输出 main_logits。
    维度: 输出通道数 output_dim 等于 GaussianAdapter.d_in + 1 (额外的 1 是不透明度 conf)。

-------

3. 各个输出层的维度 (Channel Layout)代码文件: src/depth_anything_3/model/gs_adapter.py (d_in 属性)GSDPT 输出的张量形状为 (B, Total_Channels, H, W)。这些通道在 GaussianAdapter 中被切分和解析。假设所有选项开启 (pred_offset_xy=True, pred_offset_depth=False 等)，通道排列顺序如下：

- XY 偏移 (Offset XY) [可选]:
    维度: 2 通道 (x, y)。
    作用: 允许高斯中心在像素平面上微调，不局限于像素中心。
    代码对应: raw_gaussians[..., :2] (如果 pred_offset_xy 为 True)。

- 尺度 (Scales):
    维度: 3 通道 (sx, sy, sz)。
    作用: 预测高斯球在三个轴向上的大小（在对数域或 Sigmoid 前的域）。
    代码对应: scales, ... = raw_gaussians.split(...)。

- 旋转 (Rotation/Quaternion):
    维度: 4 通道 (x, y, z, w)。
    作用: 预测四元数，表示高斯球的旋转姿态。
    代码对应: ..., rotations, ... = raw_gaussians.split(...)。

- 颜色/球谐系数 (Color/SH):
    维度: 3 × d_sh 通道。如果是纯颜色 (sh_degree=0 或 pred_color=True)，则是 3 通道 (RGB)。如果是高阶 SH，则是 $3 \times (sh\_degree + 1)^2$。
    作用: 3DGS 的外观颜色。
    代码对应: ..., sh = raw_gaussians.split(...)。

- 深度偏移 (Offset Depth) [可选]:
    维度: 1 通道。
    作用: 对 Depth Head 预测的主深度进行微调。
    代码对应: raw_gaussians[..., -1] (如果 pred_offset_depth 为 True)。

- 不透明度 (Opacity/Conf):
    维度: 1 通道。
    这是由 GSDPT 的 activate_head_gs 分离出的另一个输出 raw_gs_conf，不包含在 raw_gaussians 那个大张量里。

-------

4. GaussianAdapter：转换到世界坐标 (World Space)代码文件: src/depth_anything_3/model/gs_adapter.py (forward 方法)这一步是将上述“图像空间”的参数结合相机参数转换到“世界空间”。

A. 中心坐标 (Means) 的转换
    准备深度: 取 depths (来自深度头) + offset_depth (来自 GSDPT，可选)。
    准备像素坐标: 生成标准化的像素网格 xy_ray，并加上 offset_xy (来自 GSDPT) 乘以像素尺寸。
    反投影 (Unproject):
        利用内参 (intrinsics) 和像素坐标，计算出从相机光心发出的射线方向 directions 和起点 
        origins。origins 和 directions 是通过 extrinsics (转为 cam2worlds) 变换到世界坐标系的。
        公式: $P_{world} = Origin_{world} + Direction_{world} \times Depth$。
        代码: gs_means_world = origins + directions * gs_depths[..., None]。

B. 尺度 (Scales) 的转换
    激活: 对网络输出的 scales 做 Sigmoid，映射到 [min, max] 范围。
    物理缩放:
        网络预测的尺度是相对于像素的。需要乘以 深度 (物体越远，投影越小，实际物理尺寸越大) 和 内参系数 (考虑 FOV)。
        代码: gs_scales = scales * gs_depths[...] * multiplier[...]。

C. 旋转 (Rotations) 的转换
    归一化: 归一化四元数以保证合法性。
    空间旋转:
        网络预测的旋转是相对于相机坐标系的。
        需要左乘相机到世界的旋转矩阵 ($R_{c2w}$)。
        代码: world_quat_wxyz = cam_quat_xyzw_to_world_quat_wxyz(cam_quat_xyzw, c2w_mat)。注意代码中还处理了四元数顺序 (xyzw vs wxyz) 的转换。

D. 球谐系数 (SH/Color) 的转换
    高阶抑制: 如果不直接预测颜色，初始化时会用 sh_mask 抑制高阶 SH 系数，使训练初期更稳定。
    旋转 SH:
        球谐函数具有旋转性。如果预测的是高阶 SH，当观察角度变化（即转动世界坐标系下的相机）时，SH 系数也需要根据相机的旋转进行变换。
        代码: gs_sh_world = rotate_sh(sh, cam2worlds[...])。

E. 不透明度 (Opacities)
直接利用 GSDPT 输出的密度/置信度，通过映射函数 (map_pdf_to_opacity 在 da3.py 中调用) 转换为 0~1 的不透明度。