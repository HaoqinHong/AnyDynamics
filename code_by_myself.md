export MODEL_DIR=/opt/data/private/models/depthanything3/DA3NESTED-GIANT-LARGE
# This can be a Hugging Face repository or a local directory
# If you encounter network issues, consider using the following mirror: export HF_ENDPOINT=https://hf-mirror.com
# Alternatively, you can download the model directly from Hugging Face
export GALLERY_DIR=workspace/gallery
mkdir -p $GALLERY_DIR

# CLI auto mode with backend reuse
da3 backend --model-dir ${MODEL_DIR} --gallery-dir ${GALLERY_DIR} # Cache model to gpu
da3 auto assets/examples/SOH \
    --export-format glb \
    --export-dir ${GALLERY_DIR}/TEST_BACKEND/SOH \
    --use-backend

# CLI video processing with feature visualization
da3 video assets/examples/robot_unitree.mp4 \
    --fps 15 \
    --use-backend \
    --export-dir ${GALLERY_DIR}/TEST_BACKEND/robo \
    --export-format glb-feat_vis \
    --feat-vis-fps 15 \
    --process-res-method lower_bound_resize \
    --export-feat "11,21,31"

# CLI auto mode without backend reuse
da3 auto assets/examples/SOH \
    --export-format glb \
    --export-dir ${GALLERY_DIR}/TEST_CLI/SOH \
    --model-dir ${MODEL_DIR}


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