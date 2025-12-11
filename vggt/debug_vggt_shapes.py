# debug_vggt_shapes.py
import torch
import numpy as np
from vggt.models.vggt import VGGT

# Load model
device = "cuda"
dtype = torch.bfloat16
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model.eval()

# Create dummy input: 6 views (3 keyframes × 2 cameras) at 224×224
batch_size = 1
num_views = 6
images = torch.randn(batch_size, num_views, 3, 224, 224, device=device)

print("Input shape:", images.shape)
print("="*60)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Run aggregator
        aggregated_tokens_list, ps_idx = model.aggregator(images)
        
        print(f"\nAggregator outputs {len(aggregated_tokens_list)} feature levels:")
        for i, tokens in enumerate(aggregated_tokens_list):
            print(f"  Level {i}: {tokens.shape}")
        
        # Get final level
        scene_tokens = aggregated_tokens_list[-1]
        print(f"\n✓ Using final level: {scene_tokens.shape}")
        
        # Camera head
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        print(f"✓ Pose encoding: {pose_enc.shape}")
        
        # Depth head
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
        print(f"✓ Depth map: {depth_map.shape}")
        print(f"✓ Depth confidence: {depth_conf.shape}")
        
        # Point head
        point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
        print(f"✓ Point map: {point_map.shape}")
        print(f"✓ Point confidence: {point_conf.shape}")

print("\n" + "="*60)
print("Expected pooling:")
print(f"  scene_tokens.mean(dim=[1,2]): [{batch_size}, {scene_tokens.shape[-1]}]")
print(f"  pose_enc.mean(dim=1): [{batch_size}, {pose_enc.shape[-1]}]")
print(f"  depth stats: [{batch_size}, 2]")
print(f"  point stats: [{batch_size}, 6]")
print(f"  Total: {scene_tokens.shape[-1]} + {pose_enc.shape[-1]} + 2 + 6 = {scene_tokens.shape[-1] + pose_enc.shape[-1] + 8}")