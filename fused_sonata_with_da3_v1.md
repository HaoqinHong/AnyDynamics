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

### 区分动静态粒子并分别编码
#### 1. 体素计数方法（失败）
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

#### 2. 基于重投影一致性的动静分离（失败，应该采用 metric depth）
我们不再等到生成 3D 点云后再去分，而是在 2D 像素阶段 就通过多帧比对把动态物体“扣”出来。
核心逻辑：对于第 $i$ 帧的某个像素，如果它是静止的墙壁，那么根据相机位姿把它投影到第 $j$ 帧，其深度应该和第 $j$ 帧预测的深度严丝合缝。如果误差很大，说明它是动的（或者是被遮挡的）。

- 引入 CONF_THRESH：利用 DA3 的置信度图，直接剔除那些不可信的预测（通常是也是噪声源），提高点云质量。
- 重投影一致性 (Geometric Consistency)：这是利用深度和位姿先验的核心。只要一个点在相邻帧（CHECK_STRIDE=3）能找到对应的位置且深度一致，它就是静态背景。找不到对应（误差大）的点，就是动态物体（或新出现的区域）。
```
python run_sonata_reprojection.py
```

#### 3. 特征流一致性 (Feature-Metric Consistency)
出发点：Sonata 绝对可以做匹配，但是 完全匹配上就当作静态 这个逻辑需要加一个空间约束才成立。
正确的利用方式：特征流一致性 (Feature-Metric Consistency)，要实现 “匹配即静态”，您需要同时检查特征和位置。
我们可以设计这样一个更高级的判别器：$$IsStatic = (FeatureMatch > \text{Thresh}_1) \quad \mathbf{AND} \quad (WorldDist < \text{Thresh}_2)$$
步骤 1：找匹配利用 Sonata 提取第 $i$ 帧和第 $j$ 帧的特征，计算相似度矩阵，找到最佳匹配点对 $(P_i, P_j)$。这一步利用了 Sonata 强大的抗噪和语义理解能力，比光流法（Optical Flow）更稳，不会因为光照变化跟丢。
步骤 2：测距离将匹配点对 $(P_i, P_j)$ 都转到世界坐标系（利用 DA3 的 Pose）。如果 $||P_i^{world} - P_j^{world}|| \approx 0$：是静态背景（特征匹配，且位置没动）。如果 $||P_i^{world} - P_j^{world}|| \gg 0$：是动态物体（特征匹配，但位置变了）。如果找不到匹配：是噪声或遮挡。

利用 Sonata 的 特征相似度（Feature Similarity） 结合 几何距离（Geometric Distance），我们可以构建一个更加鲁棒的动静判别器。
这种方法的核心优势在于：它能识别出“移动的物体”。传统的距离法只能告诉你“这里变了”，但不知道是物体走了，还是新物体来了。特征法可以告诉你：“这只熊（特征A） 从 A点 移动到了 B点。”

对于每一帧 $t$ 的每一个点 $P_i$：在相邻帧 $t+k$ 中搜索与其 Sonata 特征最相似 的点 $P_{match}$。
静态判定：如果特征相似度很高（$>0.8$），且空间距离很近（$<0.1m$） $\rightarrow$ 背景。
动态判定：如果特征相似度很高（$>0.8$），但空间距离很远（$>0.1m$） $\rightarrow$ 移动物体。
噪声判定：如果找不到相似特征 $\rightarrow$ 遮挡或噪声。

**体素化后 Token 数量锐减**，罪魁祸首：DA3 的相对深度 (Relative Depth)。Depth Anything 3 (尤其是 GIANT 版本) 输出的深度通常是归一化的或者相对的。它输出的数值范围可能在 0.0 ~ 1.0 之间。如果不乘以 SCALE_FACTOR，或者乘得不够大，整个 3D 场景就会被压缩在一个极小的盒子里（例如 $1 \times 1 \times 1$）。所以采用 depth-anything/DA3NESTED-GIANT-LARGE，直接信任模型输出的 Metric Depth。

```
python run_sonata_feature_matching.py
```

#### 4. 利用 Inverse Mapping 实现无损还原
Sonata (PointTransformerV3) 的核心优势之一就是它在进行 GridSample（体素化）时，会生成一个 inverse 索引。这个索引是连接“微观像素”和“宏观体素”的高速公路，我们完全不需要 KNN，就能把体素的判断结果原封不动地“广播”回原始点云。

- DA3 生成 (Dense)：利用 DA3 预测出高精度的密集点云 $P_{dense}$ (例如 100万点)。
- Sonata 特征化 (Sparse)：将 $P_{dense}$ 放入 GridSample。得到 $P_{token}$ (例如 1万个 Token) 用于计算。关键点：同时得到一个 $I_{map}$ (Inverse Index)，它记录了“第 $i$ 个原始点属于第 $j$ 个 Token”。
- 时序匹配 (On Tokens)：在 Token 层面计算特征相似度，判断哪些 Token 是静态的，哪些是动态的。得到 $Mask_{token}$。
- 无损还原 (Broadcast)：利用 $I_{map}$ 直接查表：$Mask_{dense} = Mask_{token}[I_{map}]$。这就相当于：如果“体素A”是静态的，那么“体素A”里包含的 50 个原始点全都是静态的。

```
python run_sonata_inverse_projection.py
```

### 基于网格的稀疏序列化

要把 DA3 重建的点云通过 Sonata 编码为 Token，并融入 Transformer 来解码出 Free-time 3DGS（任意时刻的动态 3D 高斯），同时最大化动静分离效果，真正应该利用的是 Sonata 的 “基于网格的稀疏序列化（Grid-based Sparse Serialization）” 能力，并结合 “双流 Token 化（Dual-Stream Tokenization）” 策略。

- 不要试图让一个 Sonata 模型同时处理混乱的动静混合点云。
- 要利用 DA3 提供的多帧几何一致性，先在物理层面把点云拆开，然后分别用 Sonata 编码成 Token，最后在 Transformer 里做时空融合。这是通往高质量 Free-time 3DGS 的捷径。