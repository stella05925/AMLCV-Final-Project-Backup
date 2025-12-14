# debug_compute_norm_stats.py
import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

print("=" * 60)
print("Step 1: Import modules")
print("=" * 60)

try:
    import numpy as np
    print("✓ numpy imported")
except Exception as e:
    print(f"✗ numpy failed: {e}")
    exit(1)

try:
    import openpi.models.model as _model
    print("✓ openpi.models.model imported")
except Exception as e:
    print(f"✗ openpi.models.model failed: {e}")
    exit(1)

try:
    import openpi.training.config as _config
    print("✓ openpi.training.config imported")
except Exception as e:
    print(f"✗ openpi.training.config failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

try:
    import openpi.training.data_loader as _data_loader
    print("✓ openpi.training.data_loader imported")
except Exception as e:
    print(f"✗ openpi.training.data_loader failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("Step 2: Load config")
print("=" * 60)

try:
    config = _config.get_config("pi0_libero_vggt")
    print(f"✓ Config loaded: {config.name}")
    print(f"  Model: {type(config.model).__name__}")
    print(f"  Action horizon: {config.model.action_horizon}")
    print(f"  Batch size: {config.batch_size}")
except Exception as e:
    print(f"✗ Config loading failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("Step 3: Create data config")
print("=" * 60)

try:
    data_config = config.data.create(config.assets_dirs, config.model)
    print(f"✓ Data config created")
    print(f"  Repo ID: {data_config.repo_id}")
    print(f"  Local path: {getattr(data_config, 'local_dataset_path', 'N/A')}")
    print(f"  Repack transforms: {len(data_config.repack_transforms.inputs)}")
    print(f"  Data transforms: {len(data_config.data_transforms.inputs)}")
except Exception as e:
    print(f"✗ Data config failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("Step 4: Create dataset (THIS IS WHERE IT LIKELY CRASHES)")
print("=" * 60)

try:
    print("Calling create_torch_dataset...")
    dataset = _data_loader.create_torch_dataset(
        data_config,
        config.model.action_horizon,
        config.model,
    )
    print(f"✓ Dataset created: {len(dataset)} samples")
except Exception as e:
    print(f"✗ Dataset creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("Step 5: Apply transforms")
print("=" * 60)

try:
    from openpi.training.data_loader import TransformedDataset
    
    transformed_dataset = TransformedDataset(
        dataset,
        [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs],
    )
    print(f"✓ Transforms applied")
except Exception as e:
    print(f"✗ Transform failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("Step 6: Load single sample")
print("=" * 60)

try:
    print("Getting sample 0...")
    sample = transformed_dataset[0]
    print(f"✓ Sample loaded")
    print(f"  Keys: {list(sample.keys())}")
    
    if 'observation' in sample:
        print(f"  Observation keys: {list(sample['observation'].keys())}")
        if 'vggt_scene_features' in sample['observation']:
            vggt_shape = sample['observation']['vggt_scene_features'].shape
            print(f"  ✓ VGGT features present: {vggt_shape}")
        else:
            print(f"  ✗ VGGT features missing")
    
    if 'actions' in sample:
        print(f"  Actions shape: {sample['actions'].shape}")
    
    if 'state' in sample:
        print(f"  State shape: {sample['state'].shape}")
        
except Exception as e:
    print(f"✗ Sample loading failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("Step 7: Create data loader")
print("=" * 60)

try:
    from openpi.training.data_loader import TorchDataLoader
    
    data_loader = TorchDataLoader(
        transformed_dataset,
        local_batch_size=2,
        num_workers=0,  # Single-threaded
        shuffle=False,
        num_batches=2,
    )
    print(f"✓ Data loader created")
except Exception as e:
    print(f"✗ Data loader failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("Step 8: Load a batch")
print("=" * 60)

try:
    print("Getting first batch...")
    
    # Get the raw PyTorch batch first (before JAX conversion)
    from openpi.training.data_loader import TorchDataLoader
    import torch
    
    # Create a simpler loader without JAX conversion
    torch_loader = torch.utils.data.DataLoader(
        transformed_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: x,  # No collation
    )
    
    raw_batch = next(iter(torch_loader))
    print(f"✓ Raw batch loaded (list of {len(raw_batch)} samples)")
    print(f"  First sample keys: {list(raw_batch[0].keys())}")
    
    # Check for strings
    for key, value in raw_batch[0].items():
        if isinstance(value, (str, np.str_)):
            print(f"  ⚠️  String found: '{key}' = {value}")
        elif isinstance(value, dict):
            for subkey, subval in value.items():
                if isinstance(subval, (str, np.str_)):
                    print(f"  ⚠️  String found: '{key}/{subkey}' = {subval}")
    
except Exception as e:
    print(f"✗ Batch loading failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("ALL STEPS PASSED!")
print("=" * 60)