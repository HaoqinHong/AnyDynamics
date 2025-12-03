# 重建点云的坐标系问题
## DA3 的对齐方式
- 深度/尺度对齐：NestedDepthAnything3Net 同时跑主干 (anyview) 和 metric 分支，metric 分支先根据相机内参把自己的深度拉到米制 (apply_metric_scaling)，然后用主干输出的置信度、sky mask 共同挑出非天空且置信度够高的像素，做最小二乘求一个全局 scale_factor，并把这个比例一次性乘到整批深度和所有视角的平移向量上 (src/depth_anything_3/model/da3.py (lines 309-349))。所以尺度是按照整组影像共享的，而不是逐帧各算一个。

- 相机外参对齐：推理 API 在模型输出后调用 _align_to_input_extrinsics_intrinsics，用预测外参和输入外参做一次 Umeyama Sim(3)（必要时带 RANSAC），求到的尺度 scale 用来回调深度，同时把外参替换成输入外参的尺度或返回对齐后的外参 (src/depth_anything_3/api.py (lines 312-336)，具体求解在 src/depth_anything_3/utils/pose_align.py (lines 158-194))。如果只提供预测相机，GS 分支还会把预测相机和可选的 GT 外参再跑一遍 batch 版 Umeyama，得到 pose_scales 后整体缩放 cam2world 和深度 (src/depth_anything_3/model/gs_adapter.py (lines 58-102))。

模型的输入 x 本身就是 (B, N, 3, H, W) 的多视角批次，backbone 加上 camera encoder/decoder 一次同时看完所有视角才产出深度与相机 (src/depth_anything_3/model/da3.py (lines 99-168))。后续的尺度对齐是基于所有视角的有效像素一起算的单一比例 (src/depth_anything_3/model/da3.py (lines 320-349))，相机对齐也是针对整条轨迹同时求一个 Sim(3) 变换 (src/depth_anything_3/api.py (lines 312-334))。如果开启 3DGS，GaussianAdapter 会把所有视角的深度、射线一起投到世界坐标并堆成一个 (v*h*w) 级别的高斯集合 (src/depth_anything_3/model/gs_adapter.py (lines 75-169))，根本不是逐帧独立生成。因此 DA3 是 "整组帧联动建模 + 一次性对齐" 的设计，而非 "每帧先独立重建再拼起来"。


## 点云和 3DGS 的世界坐标是怎么确定的
DA3 模型在相机分支里把预测到的相机姿态 c2w 取逆，存成 w2c extrinsics（output.extrinsics = affine_inverse(c2w)），所以所有后处理都把它再反过来得到世界系  (src/depth_anything_3/model/da3.py (lines 150-165))。

- 点云：GLB 导出时 _depths_to_world_points_with_colors 先取每帧内参 K 和 w2c，把像素坐标反投影成摄像机坐标，再乘深度，然后用 c2w = (w2c)^{-1} 送到统一的世界系 (src/depth_anything_3/utils/export/glb.py (lines 110-205))；得到的 world points 再通过 _compute_alignment_transform_first_cam_glTF_center_by_points 旋转/翻转到 glTF 轴并以点云中位值作平移，使导出的 glb 既和第一台相机对齐又满足 glTF 右手坐标 (src/depth_anything_3/utils/export/glb.py (lines 230-284))。

- 3DGS：GaussianAdapter 先把每帧 extrinsics 取逆拿到 cam2worlds，把内参归一化，然后：
    - 可选地对深度做 offset/尺度；若提供 GT extrinsics，再调用 batch_align_poses_umeyama 求批量 Sim(3) 缩放，使预测相机和平移对齐 (src/depth_anything_3/model/gs_adapter.py (lines 58-102))。
    - 用采样的像素坐标 xy_ray 和 get_world_rays 生成每个像素的世界光线，origins + directions * depth 得到世界坐标的高斯中心 (means) (src/depth_anything_3/model/gs_adapter.py (lines 103-119))。
    - 其余属性（scale、四元数、SH）同样在世界坐标下表示或旋转 (src/depth_anything_3/model/gs_adapter.py (lines 120-169))。
因此 GS 的世界系与深度点云一致：都基于预测相机 cam2worlds 构建，再按需要对齐到 GT 或 glTF。

## Depth Anything 3 是否依赖于参考帧呢？
DA3 的网络输入就是整批 (B,N,3,H,W) 视角，backbone+camera encoder/decoder在一次 forward 里同时看所有帧、解出深度和相机 (src/depth_anything_3/model/da3.py (lines 99-188))；模型内部没有把第 1 帧当作显式参考来约束其他帧，所有相机和深度都是联合预测的。

推理 API 在预处理时仅把若有的输入外参做归一化，取第一帧外参 ex_t[:, :1] 反转为参考位姿，右乘到整批外参上，再用全体平移的中位数归一化尺度 (src/depth_anything_3/api.py (lines 299-310))。这只是为了让网络看到的相机轨迹在训练时的数值范围，不会固化 "第一帧坐标系" 到输出里。预测完成后，若提供原始外参，API 会调用 Umeyama Sim(3) 去对齐整条轨迹并恢复输入尺度，可以选择直接替换成输入外参 (src/depth_anything_3/api.py (lines 312-336))。因此最终导出的相机/点云也不强行依赖第一帧，只有在导出 GLB 时，为了符合 glTF 约定，会用第一帧的姿态来定义 glTF 坐标的方向，然后居中整个点云，这一步仅限可视化 (src/depth_anything_3/utils/export/glb.py (lines 110-284))。

综上，模型推理/训练阶段不依赖某个参考帧；只有在输入归一化和某些导出步骤里，第一帧被用作临时参考来定义坐标尺度或视图方向，对预测内容没有硬性约束。

## 静态多视角如何重建？
- 输入准备：把全部视角整理成 (N,3,H,W) Tensor；若有标定，传 extrinsics/intrinsics 以便模型在 _process_camera_estimation 中直接利用 (src/depth_anything_3/model/da3.py (lines 99-220))。如果没有就让 DA3 自估相机。
- 推理/导出：跑 DepthAnything3.inference(..., infer_gs=True, export_format="gs_video" 或 "gs_ply")。深度 + 相机姿态会送进 GaussianAdapter，它把所有视角的光线统一投射到一个世界系、输出 prediction.gaussians (src/depth_anything_3/model/gs_adapter.py (lines 58-169))。然后用 utils/export/gs_*.py 导出 PLY 或视频即可。
- 后处理：可选地用 _align_to_input_extrinsics_intrinsics 把预测姿态对齐到你的输入标定，再导出，以保证 3DGS 与真实世界尺度一致 (src/depth_anything_3/api.py (lines 312-336))。

DA3 的多视角重建不是“每帧独立生成点云再对齐”。模型一次 forward 就把整组帧（(B,N,3,H,W)）送进 backbone，联合推理深度、相机姿态、置信度 (src/depth_anything_3/model/da3.py (lines 99-220))。所有视角共享同一个特征上下文与姿态编码，因此输出深度本身已互相约束。在 DepthAnything3.inference 里，如果我们提供真实外参，预测会用 Umeyama Sim(3) 对齐到输入尺度；没有的话就直接使用模型估计的相机 (src/depth_anything_3/api.py (lines 312-336))。

- 点云：utils/export/glb.py 中 _depths_to_world_points_with_colors 会遍历所有帧，把像素 (u,v) 用各自内外参反投影到世界系，拼成一个统一的点集，再做采样/滤波 (src/depth_anything_3/utils/export/glb.py (lines 150-210))；也就是说点云是在世界坐标里一步构建的，而不是先 per-frame 再拼。

- 3D Gaussian Splat：开启 infer_gs=True 时，GaussianAdapter 把深度、相机和像素射线结合，直接输出世界坐标的高斯均值/尺度/旋转 (src/depth_anything_3/model/gs_adapter.py (lines 58-169))。如果有 GT 相机还能在内部按批次做 Sim(3) 校正，确保所有视角落在同一世界系。

因此“最后的结果”就是由多帧深度+姿态一次性反投影到统一世界坐标得到：点云通过 back-projection 得到稠密点集，3DGS 则得到稀疏可渲染的高斯集合。无需 per-frame 点云再对齐，所有视角在模型推理和世界系反投影时就已经对齐完毕。

# Depth Anything 3 的 Backbone 
DepthAnything3Net 在类注释里就写明“Backbone: DinoV2 feature extractor”，并且构造函数里直接通过配置实例化 net，也就是 DINOv2 编码器 (src/depth_anything_3/model/da3.py (lines 70-118))。具体实现位于 src/depth_anything_3/model/dinov2/dinov2.py：DepthAnything3Net 的 self.backbone 初始化为 depth_anything_3.model.dinov2.dinov2.DinoVisionTransformer（或在 YAML preset 中选择的变体），这就是该项目的 backbone。