import os
import numpy as np
import matplotlib.pyplot as plt
from diffusers.utils import load_video 
import torch
from src.depth_anything_3.api import DepthAnything3
# from dynamic.depth_anything_3.api import DepthAnything3
# from src.depth_anything_3.utils.visualize import visualize_depth
import os

# Set args
start_idx = 10
max_frames = 10
step = 1
# read_dir = './verification/scan24/images'
# save_dir = f"./verification/scan24/results/start-{start_idx}_max-{max_frames}_step-{step}"
read_dir = './demo/bear'
save_dir = f"./demo/bear_test/results/start-{start_idx}_max-{max_frames}_step-{step}"

def main():
    # Load frame paths
    frames = []
    for file in os.listdir(read_dir):
        if file.endswith((".jpg", ".png")):
            frames.append(os.path.join(read_dir, file))

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
    model = DepthAnything3.from_pretrained("depth-anything/DA3-GIANT", dynamic=True)
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # Set export_kwargs
    export_kwargs={
        "gs_ply": {
            "gs_views_interval": 1,
            },
        "gs_video": {
            "trj_mode": "interpolate_smooth",
            "chunk_size": 1,
            "vis_depth": "hcat",
            "enable_tqdm": True,
            },
        }

    # Inference
    prediction = model.inference(
        image=frames,                  # list of "image paths", "PIL images", or "numpy arrays"
        # extrinsics=extrinsics_array,      # Optional
        # intrinsics=intrinsics_array,      # Optional
        align_to_input_ext_scale=True,   # Whether to align predicted poses to input scale
        infer_gs=True,                   # Enable Gaussian branch for gs exports
        # use_ray_pose=False,              # Use ray-based pose estimation instead of camera decoder
        # ref_view_strategy="saddle_balanced",  # Reference view selection strategy
        # render_exts=render_extrinsics,    # Optional renders for gs_video
        # render_ixts=render_intrinsics,    # Optional renders for gs_video
        # render_hw=(height, width),        # Optional renders for gs_video
        process_res=504,
        process_res_method="upper_bound_resize",
        export_dir=save_dir,    # Optional
        export_format="mini_npz-glb-gs_ply",         # Optional, see "https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/API.md#export_format-default-mini_npz"
        export_feat_layers=[],            # List of layer indices to export features from
        conf_thresh_percentile=40.0,      # Confidence threshold percentile for depth map in GLB export
        num_max_points=1_000_000,         # Maximum number of points to export in GLB export
        show_cameras=True,                # Whether to show cameras in GLB export
        feat_vis_fps=15,                  # Frames per second for feature visualization in feat_vis export
        export_kwargs=export_kwargs       # Optional, additional arguments to export functions. export_format:key:val, see 'Parameters/Export Parameters' for details
    )


    # Access depth maps
    if hasattr(prediction, 'depth'):
        depth_maps = prediction.depth  # shape: (2, H, W)
        print("Shape of depth maps:\n", depth_maps.shape)
        print("Depth maps:\n", depth_maps)

    # Access confidence
    if hasattr(prediction, 'conf'):
        confidence = prediction.conf
        print("Shape of confidence:\n", confidence.shape)
        print("Confidence:\n", confidence)

    # Access camera parameters (if available)
    if hasattr(prediction, 'extrinsics'):
        camera_poses = prediction.extrinsics  # shape: (2, 4, 4)
        print("Shape of camera poses:\n", camera_poses.shape)
        print("Camera poses:\n", camera_poses)

    if hasattr(prediction, 'intrinsics'):
        camera_intrinsics = prediction.intrinsics  # shape: (2, 3, 3)
        print("Shape of camera intrinsics:\n", camera_intrinsics.shape)
        print("Camera intrinsics:\n", camera_intrinsics)

    # Access intermediate features (if export_feat_layers was set)
    if hasattr(prediction, 'aux') and 'feat_layer_0' in prediction.aux:
        features = prediction.aux['feat_layer_0']
        print("Shape of features:\n", features.shape)
        print("Features:\n", features)

    # Access 3D Gaussian Splats data
    if hasattr(prediction, 'aux') and 'gaussians' in prediction.aux:
        gs = prediction.aux['gaussians']
        print("Shape of gs:\n", gs.shape)
        # print("gs:\n", gs)

if __name__ == "__main__":
    main()