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
collect_vggt4d_features.py/原版 VGGT4D 流程的短板：需要多次 hook/拆分，缓存 token/QK/Gram，显存占用大；RoPE 前后的 Q/K 捕获易丢失空间位置信息；相机/CLS token 导致栅格重排不稳定，offset 搜索只能近似；分开导出浅/中/深图再后处理，插值/对齐误差会叠加；可视化阈值、层级组合需要手动调，动态区域容易碎或漏。

单看 Cross-Attn (QK) 的缺点：浅层对语义/纹理敏感，动态物体边缘容易 "光环"，深层 QK 方差对几何失配敏感但噪声高，平均相似度受全局纹理主导，动态分割不够稳。

单看 Gram (QQ/KK) 的缺点：只看自相似/全局相似，缺少跨帧方向性；浅层 KK 对亮度/纹理变化很敏感，容易把局部纹理变化当动态；深层 QQ 强调几何一致性，遇到低纹理或尺度变化会误判。

当前 fused 脚本的优势 (tools/run_vggt4d_fused.py): 在 forward 里同时抓 QK 与 QQ/KK，RoPE 注入后直接计算，避免位置信息丢失。chunk 化只存 mean/var，显存友好。自动 offset 搜索剔除 camera/CLS token，栅格还原稳定。用浅/中/深互补公式组合 (mask_std + w_shalloww_middlew_deep_dyn)/2，动态/静态分离更干净。末端 guided filter 用 RGB 边缘细化，减少光环与锯齿。

```
python tools/collect_vggt4d_fused.py \
    --image-dir ./demo/kling \
    --output-dir ./analysis/fused_result \
    --model-path /opt/data/private/models/depthanything3/DA3NESTED-GIANT-LARGE
```

### 时序聚合和掩码细化
#### 时间聚合
这里存在一个细微但重要的区别：滑动窗口 vs. 全局平均。
论文明确提到使用滑动窗口 (Sliding Window) $\mathcal{W}(t) = \{t-n, ..., t+n\}$ 来聚合统计量。目的：聚焦于局部的时序变化，避免长序列中场景结构变化过大导致的干扰。代码实现:

在 custom_attention_forward_hybrid 中：acc_m_qk[:, :, i:end] = attn_qk.mean(dim=-1) # dim=-1 是所有 tokens
分析: 这里的 dim=-1 是对 Attention 矩阵的 Key 维度求均值。如果你的输入 x 包含了所有帧的 Tokens（即 N 是 Total Tokens），那么这里的操作等价于全局窗口 (Global Window)，即每一帧都与序列中的所有其他帧计算统计量，而不是仅与相邻帧。影响: 对于较短的视频（如 Demo），这没有问题，甚至可能更稳健。但对于论文中提到的 500+ 帧长视频，全局平均可能会引入过多的噪声或导致内存溢出（尽管我们分块计算了，但逻辑上是全局的）。

#### 掩码细化
这是代码与论文差异最大的地方。我采用了一种更轻量级、纯 2D 的替代方案。
论文方法 (Sec 3.4 - Projection Gradients) 原理: 基于 3D 几何。利用初步的深度和位姿，将点投影到其他视图。判据: 动态点在投影后会有较大的几何重投影误差 ($\mathcal{L}_{proj}$) 和光度误差。方法: 计算投影梯度的聚合 ($agg^{proj}$)。这是一套涉及 3D 反投影和重投影的复杂流程。

根据论文 Supplementary Material Sec 7.3 ，官方的 Mask Refinement 包含两个核心步骤：
SOR (Statistical Outlier Removal)：统计滤波，在 3D 空间去除离群噪点。
Clustering (聚类)：对剩余的 3D 点进行聚类，并在簇内平均梯度分数，以消除局部噪声。

完全复现 VGGT4D 的方案在 DepthAnything3 代码库中 (tools/run_vggt4d_full.py):
```
python tools/run_vggt4d_full.py \
    --image-dir ./demo/kling \
    --output-dir ./analysis/vggt4d_full_result \
    --model-path /opt/data/private/models/depthanything3/DA3NESTED-GIANT-LARGE
```
 
#### VGGT4D 复现结果分析
mask_{t:03d}.png：动态物体掩码 (Dynamic Mask)
白色区域 (255)：被判定为动态 (Dynamic) 的区域。例如走动的人、行驶的车等。
黑色区域 (0)：被判定为静态 (Static) 的背景区域。

depth_compare_{t:03d}.png：深度图对比 (Depth Comparison)
这是一张左右拼接的图像，用于直观展示 VGGT4D 方法的效果。
左半部分：Pass 1 (原始推理解)
来源：results_p1。这是直接将图片喂给 Depth Anything 3 (DA3) 得到的深度图。
特征：包含动态物体。在多视角重建中，移动物体通常会导致几何冲突（因为它们违反了静态场景假设），你可能会在这一侧看到物体周围有伪影、深度不连续或者拖影。

右半部分：Pass 2 (掩码抑制后推理)
来源：results_p2。这是应用了 Early-Stage Masking 后的结果。
原理：在这次推理中，脚本将上面生成的 mask 注入到了 DA3 的前 8 层 Attention 中，强制模型 忽略 动态区域的 Key Token。

论文在 Limitations (局限性) 章节明确指出了这一点 ：
"Second, our mask refinement depends on the quality of the initial depth estimates from VGGT. If the backbone misestimates depth (e.g., blending foreground and background), the projection gradients become unreliable." (第二，我们的掩码细化依赖于 VGGT 的初始深度估计质量。如果骨干网络错误估计了深度（例如混合了前景和背景），投影梯度将变得不可靠。) 

在 Pass 1（初始推理）中，模型还没有被 Mask 抑制，因此它看到的动态物体会干扰深度估计。这可能导致物体边缘的深度不准确（比如产生“拖影”或与背景粘连）。当我们用这些不完美的深度去计算投影梯度（$\nabla r$）时，错误的深度会导致错误的几何不一致性判断。比如，一个静止的背景点如果深度估错了，投影到另一帧时也会产生很大的误差，从而被误判为“动态”，形成伪影。即使深度大体准确，投影梯度图（Gradient Map） 本身也不是一张干净的二值掩码，而是一张充满高频噪声的灰度图。

VGGT4D 的方法对于 Depth Anything 3 (DA3) 同样适用，甚至因为 DA3 拥有更强的骨干网络（ViT-Giant），效果理论上可能更好。VGGT4D 的核心假设是：动态物体在 Attention 层的特征分布（Gram Matrix）与静态背景不同。

“投影梯度细化”确实是导致“越往后伪影越多”的直接元凶，而根本原因在于相机位姿（Pose）的累积误差（Drift）。这是所有基于几何一致性（Geometric Consistency）的方法在长序列视频处理中面临的共同难题。

VGGT4D 的核心假设是：如果一个点是静止的，且我们的**深度（Depth）和位姿（Pose）**是完美的，那么这个点投影到相邻帧时，应该会有很小的重投影误差。反之，误差大就是动态物体。

### 我们的改进方案  
既然热力图（Gram Feature）效果很好，说明DA3 提取的语义动态线索非常准确。我们应该利用它的优势，屏蔽梯度的劣势。完全弃用投影梯度：既然它是伪影之源，且在长视频中受 Pose 漂移影响大，直接去掉。仅依赖 Gram 热力图 + Guided Filter：用 表现好 的热力图作为主体。用 RGB 原图的 Guided Filter 来把模糊的热力图“吸附”到清晰的物体边缘（解决分辨率低的问题），而不是用梯度去修边。

全局视野 (No Sliding Window)：在计算 Gram Matrix 和 QK 统计量时，强制让每一帧都与整个视频的所有帧进行交互。这意味着模型能从全局角度判断“谁是动的”（动态物体在全局范围内特征分布差异大）。
强制全局化 (Force Global Stats)：即使是在 DA3 的浅层（Local Attention 层），我们也手动拼接所有帧的 Key，强制计算全局统计量，确保浅层也能捕捉到运动信息。
引导滤波 (Guided Filter)：继续使用您认可的 RGB 引导滤波进行边缘吸附。
分块计算 (Chunking)：为了防止全局计算爆显存（OOM），内置了分块计算逻辑。


RGB-D 联合引导滤波 (RGB-D Guided Filter)：
原理：之前的 Guided Filter 只用了 RGB 颜色边缘。但在很多场景下（如人穿着和背景颜色相近的衣服），仅靠颜色无法区分。改进：利用 DA3 强大的深度估计能力，将 Depth（深度图） 作为第四个通道加入引导图（Guide Image）。这样，掩码不仅会吸附颜色边缘，还会强力吸附深度不连续的几何边缘（通常就是物体轮廓）。


```
python tools/run_any4dgsv1.py \
    --image-dir ./demo/kling \
    --output-dir ./analysis/any4dgs \
    --model-path /opt/data/private/models/depthanything3/DA3NESTED-GIANT-LARGE
```

时序平滑 (Temporal Smoothing)：不再是算出一帧处理一帧，而是先计算出全视频的 Coarse Mask 序列。在时间维度上应用 高斯平滑 (Gaussian Smoothing)。比如，第 $t$ 帧的掩码会参考 $t-2, t-1, t+1, t+2$ 帧的信息。这能极大消除“上一帧有、下一帧没了”的闪烁问题。双阈值滞后 (Hysteresis)：高阈值 (Strong)：确信是动态的核心区域（如 >0.6）。低阈值 (Weak)：可能是动态的边缘区域（如 >0.3）。
策略：只有当弱区域与强区域相连时，才保留它。这样既能保留锐利的边缘细节，又能过滤掉背景中孤立的低置信度噪点。参数微调：略微增大了 Guided Filter 的半径，增强空间平滑性。

```
python tools/run_any4dgs.py \
    --image-dir ./demo/kling \
    --output-dir ./analysis/any4dgs_v2 \
    --model-path /opt/data/private/models/depthanything3/DA3NESTED-GIANT-LARGE
```