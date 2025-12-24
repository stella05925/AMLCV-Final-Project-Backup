### 1. Install
Please follow the installation instructions in the README of each of the repos (LIBERO, openpi, vggt)

Download LIBERO-Spatial Dataset locally:
```bash
python benchmark_scripts/download_libero_datasets.py --datasets DATASET
# where DATASET is chosen from [libero_spatial, libero_object, libero_100, libero_goal]
```

### 2. Precompute 3D Scene features

```bash
python vggt/precompute_3d_features.py --suite libero_spatial --input_dir /path/to/your/LIBERO/libero/datasets/ --output_dir libero/datasets_with_vggt --num_keyframes 3 --device cuda --target_size 224
```

### 3. Finetune Baseline

```bash
uv run openpi/scripts/train.py pi0_libero_low_mem_finetune --exp-name=baseline --num-train-steps=5000 --batch-size=2 --overwrite
```

### 4. Finetune our model (pi0 + vggt)

```bash
uv run scripts/train.py pi0_libero_vggt --exp-name=pi0_vggt --num-train-steps=5000 --batch-size=2 --log-interval=100 --overwrite
```

### 5. Evaluate with LIBERO (with docker)

```bash
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi0_libero_vggt --policy.dir ./checkpoints"
export CLIENT_ARGS="--args.task-suite-name libero_spatial"
docker compose -f ./examples/libero/compose.yml up
```
note: to evaluate baseline model set --policy.config to pi0_libero_low_mem_finetune


### Model Checkpoints
- vggt features: https://huggingface.co/stellaaaa/Pi0_vggt_libero_spatial_5k/tree/main
- vggt latents: https://huggingface.co/stellaaaa/Pi0_vggt_libero_spatial_latents/tree/main 