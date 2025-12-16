# VGGT On-the-Fly Feature Computation for LIBERO Evaluation

This guide explains how to use on-the-fly VGGT feature computation during LIBERO evaluation.

## Overview

The VGGT inference module (`vggt_inference.py`) computes VGGT scene features in real-time during evaluation, matching the features used during training. This eliminates the need for precomputed features stored in HDF5 files.

## Architecture

### VGGT Feature Pipeline

```
Observation (2 cameras: agentview + wrist)
    ↓
VGGT Model (facebook/VGGT-1B from HuggingFace)
    ├── Aggregator: Multi-view scene tokens [B, S=2, num_tokens, 2048]
    ├── Camera Head: Pose encoding [B, S=2, 9]
    ├── Depth Head: Depth predictions [B, S=2, H, W, 1]
    └── Point Head: 3D point cloud [B, S=2, H, W, 3]
    ↓
Pooling Strategy:
    ├── Scene tokens: mean(dim=[views, spatial]) → [2048]
    ├── Pose: mean(dim=views) → [9]
    ├── Depth stats: [mean, std] → [2]
    └── Point stats: [mean_xyz, std_xyz] → [6]
    ↓
Final Features: [2065] = 2048 + 9 + 2 + 6
```

### Feature Breakdown

| Component | Dimensions | Description |
|-----------|-----------|-------------|
| Scene tokens | 2048 | Pooled visual-geometric features from both cameras |
| Pose encoding | 9 | Relative camera pose parameters |
| Depth statistics | 2 | Mean and std of predicted depth |
| Point statistics | 6 | Mean and std of 3D point cloud (x,y,z) |
| **Total** | **2065** | Complete scene representation |

## Usage

### 1. Basic Usage (Without Docker)

```bash
# Run evaluation with VGGT features
python openpi/examples/libero/main.py \
    --use_vggt=True \
    --vggt_model_name="facebook/VGGT-1B" \
    --vggt_device="cuda" \
    --vggt_target_size=224 \
    --task_suite_name=libero_spatial \
    --num_trials_per_task=50
```

### 2. Test VGGT Inference Standalone

```bash
# Test that VGGT inference works correctly
python openpi/examples/libero/vggt_inference.py
```

Expected output:
```
Testing VGGT inference...
============================================================
Loading VGGT model 'facebook/VGGT-1B' on cuda with dtype=torch.bfloat16...
✓ VGGT model loaded successfully

Encoding observation...
✓ VGGT features shape: (2065,)
✓ Feature stats: min=..., max=..., mean=...

============================================================
✅ VGGT inference working correctly!

Feature breakdown:
  - Scene tokens: ... (2048 dims)
  - Pose encoding: ... (9 dims)
  - Depth stats: [...] (2 dims)
  - Point stats: ... (6 dims)
```

### 3. Docker Usage (Python Version Limitation)

**⚠️ Important:** VGGT requires Python 3.10+, but the LIBERO Docker container uses Python 3.8 for compatibility.

**Option A: Use Precomputed Features (Recommended)**

The recommended approach is to precompute VGGT features offline and use them during evaluation:

```bash
# 1. Precompute features (outside Docker, with Python 3.10+)
python vggt/precompute_3d_features.py \
    --suite=libero_spatial \
    --input_dir=LIBERO/libero/datasets \
    --output_dir=LIBERO/libero/datasets_with_vggt \
    --num_keyframes=3 \
    --target_size=224

# 2. Use precomputed features during training/evaluation
# (Features are automatically loaded from HDF5 files)
```

**Option B: Build Custom Docker Image with Python 3.10+**

If you need on-the-fly computation in Docker:

1. Create a new Dockerfile with Python 3.10+
2. Resolve LIBERO compatibility issues
3. Uncomment the VGGT installation lines in the Dockerfile

## Implementation Details

### VGGTInference Class

```python
from vggt_inference import VGGTInference

# Initialize encoder (loads model from HuggingFace)
encoder = VGGTInference(
    model_name="facebook/VGGT-1B",  # HuggingFace model ID
    device="cuda",                   # 'cuda' or 'cpu'
    target_size=224,                 # Image size (must be divisible by 14)
    num_keyframes=3,                 # Not used in single-frame mode
)

# Encode observation
features = encoder.encode_single_observation(
    agentview_image=obs["agentview_image"],  # [H, W, 3] uint8
    wrist_image=obs["robot0_eye_in_hand_image"],  # [H, W, 3] uint8
)

# features.shape = (2065,)
```

### Integration with main.py

The `main.py` evaluation script has been updated to:

1. **Initialize VGGT encoder** (if `--use_vggt=True`)
   ```python
   vggt_encoder = VGGTInference(...)
   ```

2. **Compute features at each timestep**
   ```python
   vggt_features = vggt_encoder.encode_single_observation(
       agentview_image=obs["agentview_image"],
       wrist_image=obs["robot0_eye_in_hand_image"],
   )
   ```

3. **Add features to observation dict**
   ```python
   element["observation/vggt_scene_features"] = vggt_features
   ```

4. **Send to policy server**
   ```python
   action_chunk = client.infer(element)["actions"]
   ```

## Performance Considerations

### Inference Speed

- **VGGT model size:** ~1.2GB (ViT-Large backbone)
- **Single frame encoding:** ~50-100ms on NVIDIA A100
- **Batch encoding:** More efficient for multiple observations

### Memory Usage

- **GPU memory:** ~2-3GB for VGGT model
- **Total GPU memory:** ~6-8GB (VGGT + policy model)

### Optimization Tips

1. **Use mixed precision:** Automatically enabled (bfloat16 on A100, float16 on older GPUs)
2. **Batch observations:** Process multiple cameras together
3. **Cache features:** For static scenes, reuse features across timesteps
4. **Precompute for training:** Always use precomputed features for faster training

## Comparison: On-the-Fly vs Precomputed

| Aspect | On-the-Fly | Precomputed |
|--------|-----------|-------------|
| **Speed** | ~50-100ms/frame | Instant (already in HDF5) |
| **Memory** | +2-3GB GPU | No extra GPU memory |
| **Storage** | No extra storage | +~10-20% HDF5 file size |
| **Flexibility** | Can change VGGT params | Fixed features |
| **Use Case** | Real-time evaluation | Training and fast evaluation |

## Troubleshooting

### Error: "VGGT module not available"

```bash
# Install VGGT package
cd vggt
pip install -e .
```

### Error: "Expected VGGT features shape (2065,), got ..."

This indicates a mismatch in the pooling strategy. Verify:
1. Scene tokens are pooled to [2048] (not [1024])
2. All components sum to 2065 dimensions

### Error: "CUDA out of memory"

Solutions:
1. Reduce batch size
2. Use CPU inference: `--vggt_device=cpu`
3. Use precomputed features instead

### Python Version Incompatibility

VGGT requires Python 3.10+. If using Python 3.8:
- Use precomputed features (recommended)
- Build separate container with Python 3.10+

## References

- **VGGT Paper:** [Visual Geometry Grounded Transformer](https://github.com/facebookresearch/vggt)
- **Model:** [facebook/VGGT-1B on HuggingFace](https://huggingface.co/facebook/VGGT-1B)
- **Precomputation Script:** `vggt/precompute_3d_features.py`
- **Inference Module:** `openpi/examples/libero/vggt_inference.py`

## Example: Full Evaluation Pipeline

```bash
# 1. Test VGGT inference
python openpi/examples/libero/vggt_inference.py

# 2. Run evaluation with VGGT (CPU fallback for Python 3.8)
python openpi/examples/libero/main.py \
    --use_vggt=True \
    --vggt_device=cpu \
    --task_suite_name=libero_spatial \
    --num_trials_per_task=10

# 3. Or use precomputed features (faster)
python openpi/examples/libero/main.py \
    --use_vggt=False \
    --task_suite_name=libero_spatial \
    --num_trials_per_task=50
```

## Notes

- **Default:** VGGT is disabled by default (`--use_vggt=False`)
- **Model download:** First run will download ~1.2GB model from HuggingFace
- **Checkpoint caching:** Models are cached in `~/.cache/huggingface/`
- **Image preprocessing:** Automatically handles resizing from 256x256 to 224x224

## Future Improvements

- [ ] Add keyframe selection for temporal modeling
- [ ] Support batched inference for multiple observations
- [ ] Add feature caching for static scenes
- [ ] Optimize memory usage with model quantization
- [ ] Support alternative VGGT model sizes (Base, Giant)
