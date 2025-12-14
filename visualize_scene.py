"""可视化 scene2.glb 中的点云和路径数据（可能是相机位姿）"""

import sys
import numpy as np
import rerun as rr

try:
    import trimesh
except ImportError:
    print("错误: 需要安装 trimesh 库")
    print("安装命令: pip install trimesh")
    sys.exit(1)

# 获取文件路径
if len(sys.argv) < 2:
    asset_path = "scene2.glb"
    print(f"使用默认路径: {asset_path}")
else:
    asset_path = sys.argv[1]

print(f"正在加载 {asset_path}...")

try:
    scene = trimesh.load(asset_path)
except Exception as e:
    print(f"加载文件失败: {e}")
    sys.exit(1)

# 初始化 Rerun
rr.init("scene2_visualization", spawn=True)

# 设置坐标系
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

print("\n=== 数据统计 ===")
print(f"几何体数量: {len(scene.geometry)}")

# 1. 可视化点云数据 (geometry_0)
if "geometry_0" in scene.geometry:
    pointcloud = scene.geometry["geometry_0"]
    if isinstance(pointcloud, trimesh.PointCloud):
        print(f"\n点云数据 (geometry_0):")
        print(f"  顶点数: {len(pointcloud.vertices)}")
        
        # 采样点云（如果点太多，只显示一部分）
        max_points = 1e8  # 最多显示1e8个点
        if len(pointcloud.vertices) > max_points:
            indices = np.random.choice(len(pointcloud.vertices), max_points, replace=False)
            points = pointcloud.vertices[indices]
            print(f"  [采样显示 {max_points} 个点，总共 {len(pointcloud.vertices)} 个点]")
        else:
            points = pointcloud.vertices
            indices = None
        
        # 使用点云本身的颜色（如果存在）
        colors = None
        if hasattr(pointcloud, 'colors') and pointcloud.colors is not None:
            # 点云有颜色属性
            if indices is not None:
                # 如果采样了，也要采样对应的颜色
                colors = pointcloud.colors[indices]
            else:
                colors = pointcloud.colors
            
            # 转换为RGB格式（如果是RGBA，只取前3个通道）
            if colors.shape[1] == 4:
                colors = colors[:, :3]  # 取RGB，忽略Alpha
            elif colors.shape[1] == 3:
                colors = colors  # 已经是RGB
            else:
                colors = None  # 格式不支持，使用默认颜色
        
        # 如果点云没有颜色，根据Z坐标生成颜色作为后备方案
        if colors is None:
            print("  [警告] 点云没有颜色信息，使用Z坐标生成颜色")
            colors = np.zeros((len(points), 3), dtype=np.uint8)
            if len(points) > 0:
                z_min, z_max = points[:, 2].min(), points[:, 2].max()
                if z_max > z_min:
                    z_normalized = (points[:, 2] - z_min) / (z_max - z_min)
                    colors[:, 0] = (z_normalized * 255).astype(np.uint8)  # 红色通道
                    colors[:, 1] = ((1 - z_normalized) * 255).astype(np.uint8)  # 绿色通道
                    colors[:, 2] = 128  # 蓝色通道固定
                else:
                    colors[:] = [128, 128, 255]  # 默认蓝色
        else:
            print(f"  [使用原始颜色] 颜色范围: RGB({colors.min()}-{colors.max()})")
        
        rr.log("world/pointcloud", rr.Points3D(points, colors=colors, radii=0.01))
        print(f"  ✓ 点云已可视化")

# 2. 可视化路径数据 (geometry_1 到 geometry_60)
paths_logged = 0
path_positions = []

for i in range(1, 61):
    geom_name = f"geometry_{i}"
    if geom_name in scene.geometry:
        path = scene.geometry[geom_name]
        if isinstance(path, trimesh.path.Path3D):
            if len(path.vertices) > 0:
                # 将路径作为线条可视化
                # Path3D 可能包含多个实体，我们需要提取所有线条
                if hasattr(path, 'entities') and len(path.entities) > 0:
                    # 如果有实体，提取线条
                    for entity in path.entities:
                        if hasattr(entity, 'points'):
                            line_points = path.vertices[entity.points]
                            if len(line_points) > 1:
                                rr.log(f"world/paths/path_{i}", rr.LineStrips3D([line_points]))
                                paths_logged += 1
                                # 收集路径的起点位置（可能是相机位置）
                                path_positions.append(line_points[0])
                else:
                    # 如果没有实体，直接使用所有顶点作为一条线
                    if len(path.vertices) > 1:
                        rr.log(f"world/paths/path_{i}", rr.LineStrips3D([path.vertices]))
                        paths_logged += 1
                        path_positions.append(path.vertices[0])

print(f"\n路径数据:")
print(f"  路径数量: {paths_logged}")
if path_positions:
    path_positions = np.array(path_positions)
    print(f"  路径起点位置数量: {len(path_positions)}")
    
    # 如果路径起点看起来像是相机位姿，用箭头显示
    # 检查路径起点是否形成轨迹
    if len(path_positions) > 1:
        # 计算相邻点之间的方向向量
        directions = np.diff(path_positions, axis=0)
        # 只显示部分箭头（避免太密集）
        step = max(1, len(directions) // 50)  # 最多显示50个箭头
        origins = path_positions[::step][:-1]
        vectors = directions[::step]
        
        # 归一化向量用于显示方向
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # 避免除零
        normalized_vectors = vectors / norms * 0.5  # 箭头长度0.5
        
        if len(origins) > 0:
            rr.log("world/camera_trajectory", rr.Arrows3D(
                origins=origins,
                vectors=normalized_vectors,
                colors=[[255, 0, 0, 255]] * len(origins)  # 红色箭头
            ))
            print(f"  ✓ 相机轨迹箭头已可视化 ({len(origins)} 个箭头)")
        
        # 将路径起点作为点显示
        rr.log("world/camera_positions", rr.Points3D(
            path_positions,
            colors=[[255, 255, 0, 255]] * len(path_positions),  # 黄色点
            radii=0.05
        ))
        print(f"  ✓ 相机位置点已可视化 ({len(path_positions)} 个点)")

print("\n=== 可视化完成 ===")
print("在 Rerun Viewer 中查看结果")
print("\n提示:")
print("  - 点云: world/pointcloud (蓝色到红色渐变)")
print("  - 路径线条: world/paths/path_*")
print("  - 相机轨迹箭头: world/camera_trajectory (红色)")
print("  - 相机位置: world/camera_positions (黄色点)")

