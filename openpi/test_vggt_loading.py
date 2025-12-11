# test_vggt_loading_simple.py
import logging
logging.basicConfig(level=logging.INFO)

from pathlib import Path
import sys
sys.path.insert(0, 'src')

from openpi.transforms import LoadVGGTFeatures

# Create loader
loader = LoadVGGTFeatures("/home/stella/projects/vggt/libero/datasets_with_vggt/libero_spatial")

# Test data
test_data = {
    'episode_index': 'Lifelong-Robot-Learning/LIBERO/task_0/episode_0',
    'frame_index': 0,
}

print("\nTesting VGGT loading...")
result = loader(test_data)

if 'observation' in result and 'vggt_scene_features' in result['observation']:
    features = result['observation']['vggt_scene_features']
    print(f"✓ Successfully loaded VGGT features: shape {features.shape}")
    print(f"  Feature values: min={features.min():.3f}, max={features.max():.3f}, mean={features.mean():.3f}")
else:
    print("✗ Failed to load VGGT features")