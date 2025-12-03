### 复现 VGGT4D
DepthAnything3 的 backbone 就是 DINOv2 变体；在 src/depth_anything_3/model/dinov2/vision_transformer.py (lines 382-425) 可以看到预设：vit_small 和 vit_base 都是 12 层 Transformer block，vit_large 是 24 层，vit_giant2 是 40 层。你的具体模型（例如 da3-giant）引用哪个变体，就对应那条深度配置。

使用 tools/collect_vggt4d_features.py

脚本会给 DepthAnything3 模型挂钩，收集每个 DINO block 的输入/输出 token，以及注意力里的 Q/K，然后计算 Gram 统计。Hook 注册/卸载和 Gram 计算在 tools/collect_vggt4d_features.py (lines 20-82)，主流程在 tools/collect_vggt4d_features.py (lines 92-120)。
准备一组同分辨率的多视角 JPG（脚本默认按文件名排序，load_images 在 tools/collect_vggt4d_features.py。然后执行示例命令：
```
mkdir -p frames
ffmpeg -i demo/4D-Video.mp4 -vf "fps=10" frames/%05d.jpg

python tools/collect_spatial_features.py \
    --model-name da3nested-giant-large \
    --image-dir ./demo/kling \
    --output-dir ./analysis/vis_results_kling

python tools/make_all_videos.py \
    --root-dir ./analysis/vis_results_kling \
    --fps 10
```
运行后会输出浅/中/深层 block 列表，并把 prediction（深度/相机）、block_states（每层 tokens、Q/K）以及 gram_stats 统统保存到 --out-path 指定的文件里。

### 实现细节
DA3 使用了旋转位置编码 (RoPE)，这是判断空间几何关系的关键。普通的 Hook 方法（register_forward_hook）只能拿到计算完的结果，拿不到 RoPE 注入过程中的 Q 和 K。我们在运行时，强行把模型里所有 Attention 层的 forward 函数替换成了我们自己写的 custom_attention_forward_full。我们得以在计算 Gram 矩阵之前，手动插入了 q = self.rope(q, pos)，从而捕获了包含空间信息的特征。

利用统计学原理。我们不需要存下巨大的矩阵，只需要它的均值 (Mean) 和 方差 (Var)。

- 把 Token 切成 1024 个一组的小块。
- 算出这一小块与全图的关系，累加 Sum 和 Sum_Square。
- 算完一块，立刻扔掉 (del)，只保留累加器。

DA3 在输入序列中塞入了特殊的 Token（CLS/Camera Token），导致特征数量（比如 1297）无法被 reshape 成正方形图片。我们的脚本内置了一个暴力搜索算法，尝试剔除前 0~16 个 Token，直到剩下的数量能完美开方成一个 Grid。脚本自动识别出 Offset: 1，成功剔除了那个捣乱的 Camera/CLS Token，让图像恢复正常。

### 标准空间显著性的 Cross-Attention 输出图像的逻辑
_similarity.png (原始均值)
数学含义：S = Mean(QK^T)，表示该像素与全视频其他像素的平均相似度。
物理含义：高亮 (红) 表示该物体在所有帧里都长得一样，且位置关系稳定（通常是静态背景）。暗淡 (蓝) 表示该物体格格不入。

_inv_similarity.png (反转均值) [浅层/中层重点看这个]
数学含义：S_inv = 1 - Mean(QK^T)，表示该像素与全视频其他像素的平均不相似度。
物理含义：高亮 (红) 动态物体，在浅层（语义层），动态物体（人、车）是场景中的Outlier，它们与全局背景的相似度低。反转后，差异越大越红，从而高亮出动态物体。

_variance.png (方差) [深层重点看这个]
数学含义：V = Var(QK^T)，表示该像素与全视频其他像素相似度的变化程度。
物理含义：高亮 (红)：动态物体，在深层（几何层），模型强行寻找几何一致性。静态背景几何关系稳定，方差小（蓝）。动态物体在动，违反了极线约束，导致模型一会能匹配上，一会匹配不上，特征波动极大，方差大（红）。

_stability.png (稳定性 / 1-Var)
数学含义：Stability = 1 - Var(QK^T)，表示该像素与全视频其他像素相似度的稳定程度。
物理含义：高亮 (红)：极其稳定的静态背景，这是用来做反向验证的，用于确认哪些区域是绝对静止的。

### 基于 Gram 矩阵的 Cross-Attention 空间特征解释
VGGT4D 的核心假设是：动态物体 = 异常值 (Outliers)。
利用 Cross-Attention 计算的 Gram 矩阵，我们可以量化每个像素在时间维度上的一致性和稳定性，从而区分动态物体和静态背景。
A^{QQ} = Q Q^T：用于计算 S^{QQ} 和 V^{QQ}，衡量像素在同一帧内的自相似性。（中层和深层 Mask 的核心）
A^{KK} = K K^T：用于计算 S^{KK} 和 V^{KK}，衡量像素在全视频范围内的全局相似性。（浅层 Mask 的核心）
A^{QK} = Q K^T：用于计算 S^{QK} 和 V^{QK}，衡量像素在不同帧之间的交互相似性。（浅层 Mask 的辅助）

_w_shallow.png (对应 $1-S^{KK} \cdot V^{QK}$)
预期：利用语义和纹理差异。动态物体轮廓清晰，呈现红色。
适用层级：Layer 01 - 05。

_w_middle.png (对应 $1-S^{QQ}$)
预期：利用 Query 的自相似性。动态物体因与背景不同而呈现深红色。
适用层级：Layer 06 - 15。

_w_deep.png (对应 $S^{QQ} \cdot (1-V^{QQ})$)
预期：这是静态背景的 Mask，静态背景因几何一致性强，呈现红色；而动态物体违反几何约束，呈现蓝色/黑色。
适用层级：Layer 18 - 21。注意：之前的脚本这里是反的，现在这个逻辑完全符合论文公式。

_final_mask.png
预期：将上述三者融合。理想情况下，这应该是一张非常干净的动态物体分割图（如果不同层级互补得好）。


### 问题分析与改进
为什么 Eeasi3R 的标准 Cross-Attention 和 VGGT4D 的 Gram Attention 会有一层 红色的轮廓？
这其实是 Patch-based 模型（如 ViT/DINO）的固有特性 加上 插值算法 共同导致的：边缘响应强（Edge Attention）：DINOv2 的特征提取器非常擅长捕捉物体的 边界。对于动态物体，其边缘（与背景接触的地方）通常是语义和几何冲突最剧烈的地方，因此特征值的响应最高（红色）。而物体内部如果纹理比较均一（比如纯色衣服），特征响应反而会变弱（变绿/蓝），形成了 空心。

模型的原生分辨率只有 36x36 (Patch Size 14)。
当我们把它强制放大到 504x504 时，边缘的高响应区被拉伸模糊，形成了一个发光的 光环（Halo）。在 Jet 颜色映射下，0.5 左右的模糊边界会显示为黄色/绿色，而 1.0 的强边界显示为红色，视觉上就像一个红圈套着内部。

在动态物体分割（Dynamic Object Segmentation）这个任务上，Easi3R 和 VGGT4D 代表了两种截然不同的切入点，但它们各自都有明显的缺陷。









```
python tools/run_vggt4d_fused.py \
    --image-dir ./demo/kling \
    --output-dir ./analysis/fused_result \
    --model-path /opt/data/private/models/depthanything3/DA3NESTED-GIANT-LARGE
```