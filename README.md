# Enhancing $\Pi_0$ with Spatial Knowledge
AML Final Project Report I Semester Winter 2025/2026

Stella Lin, Maria Rita Nogueira Lopes, Rafal Ciolek, Thomas Rames

### Branches:
- main (finetuning (pi0-vggt features) and baseline eval on libero)
- libero-enhanced (enhanced model eval on libero)
- libero-plus (baseline eval on libero plus)
- libero-plus-enhanced (enhanced model on libero plus)
- vggt-latents (finetuning (pi0-vggt latents) on libero)
- change-camera (camera change evaluations)

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

### 5. Run Evaluation

```bash
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi0_libero_vggt --policy.dir ./path/to/your/checkpoints"
export CLIENT_ARGS="--args.task-suite-name libero_spatial"
docker compose -f ./examples/libero/compose.yml up
```

### Model Checkpoints
- $\Pi_0$-vggt features: https://huggingface.co/stellaaaa/Pi0_vggt_libero_spatial_5k/tree/main
- $\Pi_0$-vggt latents: https://huggingface.co/stellaaaa/Pi0_vggt_libero_spatial_latents
