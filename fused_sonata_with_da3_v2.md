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

```
python run_preprocess_4d_tokens.py
```