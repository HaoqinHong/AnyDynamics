### DA3 复现：用 DA3 重建视频
```
mkdir -p frames
ffmpeg -i your_video.mp4 -vf "fps=10" frames/%05d.jpg # 或用其他帧率/区间截取。
```

跑推理并导出，示例（点云 + 深度导出）：
```
python demo.py \
  --model-name da3nested-giant-large \
  --image-dir frames \
  --export-dir out \
  --export-format mini_npz-glb \
  --process-res 504
```

如果要直接出 3DGS 或 3DGS 视频：
```
python demo.py \
  --model-name da3nested-giant-large \
  --image-dir frames \
  --export-dir out \
  --export-format gs_ply-gs_video \
  --infer-gs \
  --process-res 504
```

- 有真实相机外参/内参时，传入 --extrinsics path.npy --intrinsics path.npy（或在脚本中传入 numpy）。align_to_input_ext_scale=True 会把深度按输入外参尺度对齐。
- 无外参时，用模型估计的相机。GLB 导出会自动归一化场景并对齐到 glTF 坐标，便于查看。

### DA3 基于预测出的深度图（Depth Map）和相机内参（Intrinsics），利用梯度信息来计算高质量的法面法线
反投影（Back-projection）：利用深度图和内参，将每个像素 $(u, v)$ 重建为相机坐标系下的 3D 点 $\mathbf{P}(u, v) = [X, Y, Z]^T$。
计算梯度（Gradients）：利用 Sobel 算子或简单的差分，计算 3D 点在图像 $u$ 方向和 $v$ 方向的偏导数（切向量）：$\frac{\partial \mathbf{P}}{\partial u}$ 和 $\frac{\partial \mathbf{P}}{\partial v}$。
叉积（Cross Product）：法线即为两个切向量的叉积：$\mathbf{n} = \frac{\partial \mathbf{P}}{\partial u} \times \frac{\partial \mathbf{P}}{\partial v}$。
归一化：将 $\mathbf{n}$ 归一化为单位向量。

```
# 虽然对于某些纯坐标数据（如 LiDAR）法线是可选的，但在大多数感知任务中，拥有法线信息通常能提升效果
# 读取视频帧文件夹 -> 批量推理深度 -> 计算法线 -> 自动旋转坐标系（适配 Sonata/ScanNet 标准） -> 导出 PLY
python video_to_sonata_ply.py
```

### 将 DA3 的点云预测结合通过 sonata 进行编码
如果想把点云 Token 放入一个 Transformer（尤其是像 LLM 或 DiT 这样的标准 Transformer），Transformer 本质是处理 1D 序列的模型。Voxel 产生的是 3D 稀疏张量，通常需要 Flatten。而 Sonata (PTv3) 的核心贡献就是高效的序列化（Serialization），它天然地把 3D 点云变成了 Transformer 最喜欢的 1D Token 序列，且保持了空间邻近性（通过空间填充曲线）。

Sonata 和 Point Transformer V3 (PTv3) 的关系可以总结为：Sonata 是基于 PTv3 架构的一种特定的自监督预训练模型/方法。

标准 PTv3（用于分割任务时）通常是一个完整的 Encoder-Decoder（编码器-解码器） 结构（U-Net 形状）。它不仅提取特征（Encoder），还通过上采样将特征恢复到原始点数以进行逐点预测。Sonata 在预训练时去掉了 Decoder，只专注于 Encoder 部分。这是为了避免模型利用几何捷径（Geometric Shortcuts），强制 Encoder 学习更深层的语义信息。Sonata 由于侧重于自监督学习，它特别强调利用多模态信息（如颜色、法线）来辅助几何理解。Sonata 的预训练模型通常是在大规模数据集（如 SA-1B 投影出的 3D 数据）上训练的，期望输入包含丰富的语义线索。

demo/0_pca.py: 特征空间可视化 (Feature PCA Visualization)
核心目的：展示 Sonata 提取的特征（Features）是否具备语义一致性。看特征质量（颜色是否这有一块那有一块，还是语义连贯）。

demo/1_similarity.py: 特征相似度匹配 (Feature Similarity / Correspondence)
核心目的：展示模型在不同视图（View）之间寻找对应点的能力。看匹配能力（能不能找到同一个点）。

demo/2_sem_seg.py: 语义分割 (Semantic Segmentation - Linear Probing)
核心目的：展示 Sonata 预训练特征在下游任务（分类/分割）中的性能。骨干网络不微调，只训练最后一层分类器。看应用效果（能不能正确分类物体）。

```
run_sonata_with_DA3.py
```

### sonata 区分动静态粒子并分别编码
利用 Sonata (Pointcept) 框架中的 GridSample (网格采样) 机制和 PTv3 的高效长序列处理能力。

对于静态背景（大部分粒子）：当您将多帧点云通过相机外参（Extrinsics）对齐到同一个世界坐标系后，静态物体（如墙壁、桌子）在不同帧中的点会落在空间中的几乎同一个位置。Sonata 的 GridSample 功能 会将空间划分为细小的体素（Voxel，例如 2cm）。落入同一个体素内的多个点（来自不同帧）会被合并（通常保留一个或做平均）。效果：这实际上起到了**多帧去噪（Denoising）和点云致密化（Densification）**的作用。多帧的观测让静态背景的几何结构更清晰、更完整。

对于动态物体（少部分粒子）：移动物体在不同帧中的空间位置不同，因此它们会落在不同的体素中。效果：这些点会被保留下来，形成物体运动的 轨迹 或 残影。Sonata 会为这些轨迹上的每个点提取特征。这保留了运动信息，而不会像体素融合（TSDF）那样产生严重的鬼影干扰。

核心功能更新
- 帧索引追踪：在融合循环中，记录每个点到底来自哪一帧。
- 时空一致性过滤：将空间划分为网格（如 5cm），统计每个网格内包含的独立帧数量。
    - 静态判定：如果一个网格内的点来自超过 threshold（例如 5）个不同帧，视为静态。
    - 动态判定：否则视为动态（或噪声）。
- 分离导出：除了总的点云，还会分别保存 static_scene.ply 和 dynamic_objects.ply。
```
python run_sonata_fused_separation.py
```