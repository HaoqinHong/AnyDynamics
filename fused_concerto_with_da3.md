Sonata 的局限：Sonata 主要是一个 纯 3D 的预训练模型（Masked Scene Contrast），它侧重于几何结构（Geometry）的理解。
Concerto 的优势：Concerto 的核心卖点是 "Joint 2D-3D Self-Supervised Learning" (联合 2D-3D 自监督学习)。
    - 管线匹配度极高：我们的输入数据源头是 DA3（从 2D 图像生成的 3D 点云），输出目标是 FreeTimeGS（用于渲染 2D 图像的 3D/4D 表示）。这意味着我们的任务天然包含强烈的 2D 纹理与 3D 几何的对应关系。
    - 特征质量：Concerto 在预训练时就已经强制模型学习 2D 图像特征（如 DINOv2）和 3D 点云特征的对齐。这意味着 Concerto 提取的 Token 不仅仅包含“这里有个角”的几何信息，还隐含了“这里看起来像什么”的语义/纹理信息。这对于后续解码成带有颜色/SH 系数的 3D 高斯（3DGS）至关重要。
Concerto (特别是 concerto_large) 甚至可能支持直接处理颜色信息（因为它关注 2D-3D）。如果加载的模型支持颜色输入，我们可以直接传入 DA3 预测的高质量颜色，而不是仅依赖坐标，这将大大增强 Token 的表现力。

```
python run_preprocess_4d_token_with_concerto.py 
```

实现类: GridSample 类（在文件中定义）。
原理: 该类通过将点云坐标除以 grid_size 并向下取整来计算网格坐标（即体素坐标），然后使用哈希（Hash）方法处理落入同一个体素内的点（例如去重或随机采样）。理论上，静态背景的 Token 数量不会随着帧数增加而增加，只有移动的物体（熊）因为每一帧位置不同，才会产生新的 Token（形成一条运动轨迹）。

scene_latent.pt 是一个字典，包含以下关键数据（假设有 $N$ 个 Token）：
    - geo_feat ($N \times C_{concerto}$):含义：这是最有价值的部分。它包含了 Concerto 提取的 语义+几何 特征。作用：告诉 Transformer  这个位置是什么 。例如，它不只是记录这里有个点，而是编码了 这是一个椅子腿的边缘 或 这是一只熊的毛发 这样的高层语义。
    - coord ($N \times 3$):含义：体素网格的整数坐标 $(x, y, z)$。作用：提供 空间位置编码 (Positional Encoding)。Transformer 需要知道每个 Token 在空间中的相对位置。
    - color ($N \times 3$):含义：该体素内所有原始点的平均颜色。作用：提供基础的 外观信息。虽然 Concerto 特征里也隐含了纹理，但显式的 RGB 颜色对渲染任务（如 3DGS）非常有帮助。
    - time_emb ($N \times D_{time}$):含义：该体素内所有原始点的时间编码的平均值。作用：提供 时间上下文。这用于区分 这是一个只在第 10 帧出现的动态物体 还是 这是一个一直存在的静态背景。
    - cam_emb ($N \times D_{cam}$):含义：该体素被观测到时的相机视角的平均值。作用：告诉模型这个特征是从哪个角度看到的（有助于处理高光/反射）。

DA3 的反投影公式是这样的：$$P_{world} = \text{CameraOrigin} + \text{RayDirection} \times \text{Depth}$$
在这个公式里，DA3 已经把 Ray（光线方向） 和 Depth（深度） 这两个信息，完美地烘焙进了 $P_{world}$ (点云坐标) 里。
核心矛盾：光线是 像素级（Pixel-wise） 的概念，而 Token 是 体素级（Voxel-wise） 的概念。在 Concerto 阶段 GridSample 会把这 100 个来自不同角度的点，合并成 1 个 Token。

    - 如果想编码光线：这时候该填什么？是正面的光线？侧面的？还是求平均？如果求平均：$\text{平均光线} \approx 0$（互相抵消了）。这不仅没用，反而给了模型错误的信号。
    - 如果编码相机：虽然相机位置也会被平均（变成“平均观测中心”），但 cam_emb 更多是作为一个 Condition（条件/上下文） 存在的。它告诉模型：“这个特征是基于这一组相机参数提取出来的”。而在 Transformer 解码阶段，重要的是 当前查询的相机 (Query Camera)，而不是当初生成的相机。

# 利用 Concerto 编码的 3D 点云 Token，赋予 DA3 预测动态 Free-time 3DGS 的能力
DPT（如 DA3 中使用的解码器）是为 稠密 2D 预测 设计的。它的核心逻辑是：
    - 输入：来自 ViT 的 Patch Token（本质上是 2D 网格切片）。
    - 操作：Reassemble 操作将 Token 重新拼回 2D 特征图，进行上采样和卷积融合。
    - 输出：一张和原图分辨率对齐的 2D 图像（如深度图、分割图）。

Concerto (基于 PTv3) 的最大优势就是序列化（Serialization）。它把 3D 数据变成了 1D 序列，这正是标准 Transformer 最喜欢的输入格式，可以构建一个轻量级的 Time-Conditioned Transformer Decoder。

