# test_data_pipeline.py
from openpi.training import config as _config
import numpy as np

# Load config
config = _config.get_config("pi0_libero_vggt")

# Create data config
data_config = config.data.create(
    config.assets_dirs,
    config.model
)

print("✓ Data config created")
print(f"Repo ID: {data_config.repo_id}")
print(f"Repack transforms: {data_config.repack_transforms}")
print(f"Number of input transforms: {len(data_config.repack_transforms.inputs)}")

# Check if VGGT loader is in the pipeline
has_vggt = any("VGGT" in str(type(t)) for t in data_config.repack_transforms.inputs)
print(f"Has VGGT loader: {has_vggt}")