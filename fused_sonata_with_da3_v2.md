# DA3 的 3DGS 是 Pixel-wise 的吗？和 PLY 无关吗？
DA3 的 3DGS 是 Pixel-wise 的，且与 PLY 高度相关（是一体的）。
Pixel-wise 机制：DA3 的架构（基于 DPT/ViT）是一个密集预测网络 (Dense Prediction Network)。它对输入图像的每一个像素都预测了一组属性：Depth（深度）、Opacity（不透明度）、Scale（缩放）、Rotation（旋转）。这意味着，如果你有一张 $512 \times 512$ 的图，DA3 理论上就会生成 $512 \times 512$ 个高斯球。
与 PLY 的关系：它们是同源不同身。
PLY = Pixel + Depth $\rightarrow$ Unproject $\rightarrow$ XYZ 坐标 + RGB 颜色。
3DGS = PLY 的点 + 额外属性 (协方差/透明度/球谐系数)。

# DA3 的 3DGS 预测逻辑：Pixel-wise 回归，而非 Structure-aware
DA3 预测 3DGS 的过程没有使用整体点云结构 (PLY)作为先验信息。它的工作流是完全 Pixel-wise（逐像素） 和 Local（局部） 的：
  位置 (Means)：直接由 Depth Head 预测的深度图决定。$Position_{(u,v)} = Depth_{(u,v)} \times Ray_{(u,v)}$它只看当前像素的深度，完全不考虑“这个点和旁边的点是不是组成了一辆车”。
  属性 (Opacity, Scale, Rotation, SH)：由 3DGS Head (DPT) 直接从当前像素的图像特征回归得到。

# 利用 Sonata 将 PLY (无序点云) 编码为 Structured Tokens (结构化特征)，然后用 Transformer 进行时空推理
Step 1 (DA3 清洗):

- 利用 conf > 0.7 剔除边缘噪点。
- 利用 depth < 50m (Force Static Dist) 剔除天空和远景。
- 耦合点：直接输出 Tensor 格式的点云 (N, 3) 和颜色 (N, 3)，不做繁琐的 Numpy 转换。

Step 2 (Sonata 编码):

- 体素化 (GridSample)：将非结构化的清洗点云转化为规则的稀疏体素 (Token)。
- 特征提取：利用预训练的 Sonata 提取高维语义特征 (Features)。
- 输出：保存为 .pt 文件，包含 token_feat (特征), token_coord (坐标), grid_coord (索引)。这是 Transformer 最喜欢的格式。

# Canonical Scene Representation 规范化场景表征
**一次性输入所有帧，利用 Camera/Time Token 进行编码，生成全局场景 Token。**
- Per-Frame Processing:
  - DA3 $\rightarrow$ World Points ($P_t$) & Colors ($C_t$).
  - 计算 Camera Embedding ($E_{cam}$) 和 Time Embedding ($E_{time}$).
  - 注入：每个点的初始特征 $F_{pt} = \text{Concat}(C_t, E_{cam}, E_{time})$。

- Global Aggregation: 把所有帧的 $P_t$ 和 $F_{pt}$ 拼成一个巨大的 超级点云

- Sonata Processing:输入超级点云 $\rightarrow$ GridSample (物理合并发生在这里) $\rightarrow$ Encoder。

- Output:
  - 得到一组 Scene Tokens。
  - Transformer 只需要看着这组 Token，就能解码出任意时刻的 3DGS。

```
python run_preprocess_4d_tokens_v1.py
```

核心逻辑是：利用 Depth Anything 3 (DA3) 提供的强几何先验，将视频中的每一帧提升为 3D 点云，然后通过 Sonata 将这些庞大的点云在时空上压缩成紧凑的 体素 Token，并保留时间及视角信息。

在 sonata/model.py 中，PointTransformerV3 的 __init__ 定义了 enc_depths 和 stride：stride=(2, 2, 2, 2)：共 4 个下采样阶段，每次网格步长扩大 2 倍。总步长扩大了 $2^4 = 16$ 倍。但由于是稀疏点云，GridPooling 会合并同一个大格子里的所有点。对于拥挤的场景，点数的减少往往远超 16 倍，可能达到 100 倍甚至更多（取决于场景的空间分布）。

引入 get_ray_embedding：替代了之前的 Camera Pose Flattening。现在会计算每个点相对于相机的归一化视线向量 $(P - C)$，更适合 3DGS 渲染。在 Step 3 中，同样对 Ray Embedding 进行了平均池化。这意味着每个体素 Token 获得的是“平均观测方向”。

保存原始点云 (pts_w_original)：在 Step 1 中，将未经过体素化压缩的、每帧几万个点的原始点云收集起来，并在 Step 4 保存为 vis_original_da3.ply。这可以作为 Ground Truth 用来评估体素化后的质量损失。


特征上采样 (Upcasting)：加入了 while 循环逻辑（参考官方 Demo），将模型输出的 100 个稀疏特征逐层还原回 ~4万个稠密体素特征。这确保了生成的 Token 数量足以覆盖整个场景，且包含丰富的语义和几何细节。

Sonata 模型有 5 个 Stage（层级），根据 sonata/model.py 的配置，它们的通道数（例如：Stage 0 关注纹理边缘，Stage 4 关注物体语义）。
  Stage 0 (最浅层/细节): 48 维
  Stage 1: 96 维
  Stage 2: 192 维
  Stage 3: 384 维
  Stage 4 (最深层/语义): 512 维
我们执行的 Upcasting 操作是 Concat (拼接)：$$48 + 96 + 192 + 384 + 512 = \mathbf{1232}$$

体素化过程本身是会损失微观信息的，但 Sonata 的架构设计通过 “逆向映射索引” 保留了原始信息的恢复路径，使得我们可以在最后无损地找回原始点云的坐标和数量。

```
python run_preprocess_4d_tokens_v2.py
```