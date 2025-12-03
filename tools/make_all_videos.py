# tools/make_all_videos.py
"""
Usage:
python tools/make_all_videos.py --root-dir ./analysis/vis_results_kling --fps 10
"""
import cv2
import os
import argparse
from pathlib import Path
from collections import defaultdict

def make_video(image_paths, output_path, fps=10):
    if not image_paths: return
    
    # 获取尺寸
    frame = cv2.imread(str(image_paths[0]))
    if frame is None: return
    height, width, _ = frame.shape
    
    # 写入视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for p in image_paths:
        frame = cv2.imread(str(p))
        if frame is not None:
            video.write(frame)
            
    video.release()
    print(f"  [Video] Generated: {output_path}")

def main(args):
    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        print(f"Error: Directory {root_dir} not found.")
        return

    print(f"Scanning {root_dir} for image sequences...")
    
    # 1. 寻找所有包含图片的子目录
    image_dirs = set()
    for p in root_dir.rglob("*.png"):
        image_dirs.add(p.parent)
    
    if not image_dirs:
        print("No images found.")
        return

    # 2. 对每个目录进行处理
    for img_dir in image_dirs:
        print(f"\nProcessing directory: {img_dir}")
        
        # 归类图片： { (layer, suffix): [path1, path2, ...] }
        sequences = defaultdict(list)
        
        for img_path in sorted(img_dir.glob("*.png")):
            name = img_path.stem # e.g. layer_09_frame000_variance
            
            # 解析文件名结构
            # 假设格式: layer_XX_frameXXX_SUFFIX
            parts = name.split('_')
            if len(parts) < 4: continue # 格式不对跳过
            
            # 提取 layer (layer_09)
            layer_key = f"{parts[0]}_{parts[1]}"
            
            # 提取 frame (frame000)
            frame_idx = int(parts[2].replace("frame", ""))
            
            # 提取 suffix (variance, final, etc.)
            suffix = "_".join(parts[3:])
            
            group_key = (layer_key, suffix)
            sequences[group_key].append((frame_idx, img_path))
            
        # 3. 生成视频
        video_output_dir = img_dir.parent / "videos"
        video_output_dir.mkdir(exist_ok=True)
        
        for (layer, suffix), file_list in sequences.items():
            # 按帧号排序
            file_list.sort(key=lambda x: x[0])
            sorted_paths = [x[1] for x in file_list]
            
            vid_name = f"{layer}_{suffix}.mp4"
            vid_path = video_output_dir / vid_name
            
            make_video(sorted_paths, vid_path, args.fps)

    print("\nAll videos generated successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True, help="Root directory containing result images")
    parser.add_argument("--fps", type=int, default=10, help="Frame rate")
    args = parser.parse_args()
    main(args)