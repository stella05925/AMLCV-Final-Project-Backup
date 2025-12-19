# precompute_vggt_features.py (UPDATED WITH RESIZING)
import torch
import torch.nn.functional as F
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

from vggt.models.vggt import VGGT


def preprocess_image_for_vggt(image_array, target_size=224):
    """
    Convert LIBERO observation image to VGGT input format.
    
    Args:
        image_array: numpy array [H, W, 3] in range [0, 255]
        target_size: Target image size (default: 224 for standard ViT)
    
    Returns:
        torch.Tensor: [3, target_size, target_size] in range [0, 1]
    """
    # Normalize to [0, 1]
    image = torch.from_numpy(image_array).float() / 255.0
    
    # Transpose to CHW format
    image = image.permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]
    
    # Resize to target_size using bilinear interpolation
    if image.shape[1] != target_size or image.shape[2] != target_size:
        image = F.interpolate(
            image.unsqueeze(0),  # Add batch dim: [1, 3, H, W]
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False,
            antialias=True  # Better quality downsampling
        ).squeeze(0)  # Remove batch dim: [3, target_size, target_size]
    
    return image


# def extract_vggt_features_from_episode(
#     model, 
#     demo_group,
#     keyframe_indices, 
#     device, 
#     dtype,
#     target_size=224
# ):
#     """
#     Extract VGGT scene features from an episode using keyframes.
    
#     Args:
#         model: VGGT model
#         demo_group: HDF5 group for a demo
#         keyframe_indices: Indices of keyframes to use
#         device: torch device
#         dtype: torch dtype
#         target_size: Target image size for VGGT
    
#     Returns:
#         np.ndarray: Pooled VGGT features
#     """
#     # Collect images from keyframes
#     keyframe_images = []
    
#     for idx in keyframe_indices:
#         # Load from HDF5: obs/agentview_rgb and obs/eye_in_hand_rgb
#         agentview_img = demo_group['obs']['agentview_rgb'][idx]  # [128, 128, 3]
#         wrist_img = demo_group['obs']['eye_in_hand_rgb'][idx]    # [128, 128, 3]
        
#         # Preprocess for VGGT (resize to 224x224)
#         agentview_tensor = preprocess_image_for_vggt(agentview_img, target_size)
#         wrist_tensor = preprocess_image_for_vggt(wrist_img, target_size)
        
#         keyframe_images.append(agentview_tensor)
#         keyframe_images.append(wrist_tensor)
    
#     # Stack images: [S, 3, 224, 224] where S = num_keyframes * 2 cameras
#     images = torch.stack(keyframe_images).to(device)
    
#     # Add batch dimension: [1, S, 3, 224, 224]
#     images = images.unsqueeze(0)
    
#     # Run VGGT
#     with torch.no_grad():
#         with torch.cuda.amp.autocast(dtype=dtype):
#             # Get internal features from aggregator
#             aggregated_tokens_list, ps_idx = model.aggregator(images)
            
#             # Extract features
#             scene_tokens = aggregated_tokens_list[-1]  # [B, S, num_tokens, embed_dim]
#             pose_enc = model.camera_head(aggregated_tokens_list)[-1]  # [B, S, 9]
#             depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
#             point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
    
#     # Pool features to fixed-size vector
#     features = pool_vggt_features(
#         scene_tokens=scene_tokens,
#         pose_enc=pose_enc,
#         depth_map=depth_map,
#         point_map=point_map,
#     )
    
#     return features


# def pool_vggt_features(scene_tokens, pose_enc, depth_map, point_map):
#     """
#     Pool VGGT outputs to a fixed-size feature vector.
    
#     Args:
#         scene_tokens: [B, S, num_tokens, embed_dim] - internal features
#         pose_enc: [B, S, 9] - camera parameters
#         depth_map: [B, S, H, W, 1] - depth predictions
#         point_map: [B, S, H, W, 3] - 3D point predictions
    
#     Returns:
#         np.ndarray: [B, feature_dim] - pooled scene features
#     """
#     # 1. Pool scene tokens (most informative)
#     tokens_pooled = scene_tokens.mean(dim=[1, 2])  # [B, embed_dim=1024]
    
#     # 2. Pool camera parameters
#     pose_pooled = pose_enc.mean(dim=1)  # [B, 9]
    
#     # 3. Pool depth statistics
#     depth_mean = depth_map.mean(dim=[1, 2, 3])  # [B, 1]
#     depth_std = depth_map.std(dim=[1, 2, 3])    # [B, 1]
#     depth_features = torch.cat([depth_mean, depth_std], dim=-1)  # [B, 2]
    
#     # 4. Pool point map statistics
#     point_mean = point_map.mean(dim=[1, 2, 3])  # [B, 3]
#     point_std = point_map.std(dim=[1, 2, 3])    # [B, 3]
#     point_features = torch.cat([point_mean, point_std], dim=-1)  # [B, 6]
    
#     # Concatenate all features
#     all_features = torch.cat([
#         tokens_pooled,     # [B, 1024]
#         pose_pooled,       # [B, 9]
#         depth_features,    # [B, 2]
#         point_features,    # [B, 6]
#     ], dim=-1)  # [B, 1041]
    
#     return all_features.cpu().numpy()

def extract_vggt_features_from_episode(
    model, 
    demo_group,
    keyframe_indices, 
    device, 
    dtype,
    target_size=224
):
    """
    Extract VGGT latent features from an episode using keyframes.
    
    Args:
        model: VGGT model
        demo_group: HDF5 group for a demo
        keyframe_indices: Indices of keyframes to use
        device: torch device
        dtype: torch dtype
        target_size: Target image size for VGGT
    
    Returns:
        np.ndarray: Pooled VGGT latent features
    """
    # Collect images from keyframes (same as before)
    keyframe_images = []
    
    for idx in keyframe_indices:
        agentview_img = demo_group['obs']['agentview_rgb'][idx]
        wrist_img = demo_group['obs']['eye_in_hand_rgb'][idx]
        
        agentview_tensor = preprocess_image_for_vggt(agentview_img, target_size)
        wrist_tensor = preprocess_image_for_vggt(wrist_img, target_size)
        
        keyframe_images.append(agentview_tensor)
        keyframe_images.append(wrist_tensor)
    
    images = torch.stack(keyframe_images).to(device)
    images = images.unsqueeze(0)  # [1, S, 3, 224, 224]
    
    # Run VGGT aggregator only
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Get latent representations from aggregator
            aggregated_tokens_list, patch_start_idx = model.aggregator(images)
            
            # Use the final iteration's tokens (most refined)
            final_tokens = aggregated_tokens_list[-1]  # [B, S, num_tokens, embed_dim]
    
    # Pool latent features to fixed-size vector
    features = pool_latent_features(final_tokens)
    
    return features


def pool_latent_features(tokens):
    """
    Pool VGGT latent tokens to a fixed-size feature vector.
    
    Args:
        tokens: [B, S, num_tokens, embed_dim] - latent representations
                where embed_dim = 2048 for VGGT-1B
    
    Returns:
        np.ndarray: [B, feature_dim] - pooled latent features
    """
    B, S, N, D = tokens.shape
    
    # Global average pooling across sequence and tokens
    global_pool = tokens.mean(dim=[1, 2])  # [B, 2048]
    
    return global_pool.cpu().numpy()  # [B, 2048]

def process_libero_dataset(
    dataset_path: str,
    output_path: str,
    num_keyframes: int = 3,
    device: str = "cuda",
    target_size: int = 224
):
    """
    Process a LIBERO dataset file and add VGGT features.
    
    Args:
        dataset_path: Path to input LIBERO .hdf5 file
        output_path: Path to save augmented .hdf5 file
        num_keyframes: Number of keyframes to use for VGGT
        device: Device to run VGGT on
        target_size: Target image size for VGGT (must be divisible by 14)
    """
    print(f"\n{'='*60}")
    print(f"Processing: {Path(dataset_path).name}")
    print(f"{'='*60}")
    
    # Validate target_size
    assert target_size % 14 == 0, f"target_size {target_size} must be divisible by 14"
    
    # Initialize VGGT
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"Loading VGGT model on {device} with dtype={dtype}...")
    print(f"Resizing images from 128x128 to {target_size}x{target_size}")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    print("✓ VGGT loaded")
    
    # Load LIBERO dataset
    with h5py.File(dataset_path, 'r') as f_in:
        # Create output file
        with h5py.File(output_path, 'w') as f_out:
            # Copy file-level attributes
            for key, value in f_in.attrs.items():
                f_out.attrs[key] = value
            
            # Create data group
            data_group = f_out.create_group('data')
            
            # Process each demo
            demo_keys = sorted([k for k in f_in['data'].keys() if k.startswith('demo_')])
            print(f"Found {len(demo_keys)} demonstrations")
            
            for demo_key in tqdm(demo_keys, desc="Processing demos"):
                demo_in = f_in['data'][demo_key]
                demo_out = data_group.create_group(demo_key)
                
                # Get episode length
                episode_length = len(demo_in['actions'])
                
                # Select keyframe indices
                if num_keyframes == 1:
                    keyframe_indices = [0]
                elif num_keyframes == 2:
                    keyframe_indices = [0, episode_length - 1]
                else:
                    # Evenly space keyframes
                    step = episode_length // (num_keyframes - 1)
                    keyframe_indices = [min(i * step, episode_length - 1) 
                                       for i in range(num_keyframes)]
                
                # Extract VGGT features from keyframes
                vggt_features = extract_vggt_features_from_episode(
                    model=model,
                    demo_group=demo_in,
                    keyframe_indices=keyframe_indices,
                    device=device,
                    dtype=dtype,
                    target_size=target_size,
                )
                
                # Copy original data structure recursively
                def copy_item(src, dst, name):
                    """Recursively copy HDF5 items."""
                    if isinstance(src[name], h5py.Group):
                        grp = dst.create_group(name)
                        for key in src[name].keys():
                            copy_item(src[name], grp, key)
                        # Copy group attributes
                        for attr_key, attr_value in src[name].attrs.items():
                            grp.attrs[attr_key] = attr_value
                    else:
                        dst.create_dataset(
                            name,
                            data=src[name][()],
                            compression='gzip'
                        )
                        # Copy dataset attributes
                        for attr_key, attr_value in src[name].attrs.items():
                            dst[name].attrs[attr_key] = attr_value
                
                # Copy all original data
                for key in demo_in.keys():
                    copy_item(demo_in, demo_out, key)
                
                # Add VGGT features (same for all timesteps in episode)
                vggt_features_repeated = np.repeat(
                    vggt_features,
                    episode_length,
                    axis=0
                )  # [T, 1041]
                
                demo_out.create_dataset(
                    'vggt_scene_features',
                    data=vggt_features_repeated,
                    compression='gzip'
                )
                
                # Copy demo-level attributes
                for attr_key, attr_value in demo_in.attrs.items():
                    demo_out.attrs[attr_key] = attr_value
    
    print(f"✓ Saved to {output_path}")


def process_entire_suite(
    suite_name: str,
    input_dir: str,
    output_dir: str,
    num_keyframes: int = 3,
    device: str = "cuda",
    target_size: int = 224
):
    """
    Process all tasks in a LIBERO suite.
    
    Args:
        suite_name: LIBERO suite name (e.g., 'libero_spatial')
        input_dir: Directory containing LIBERO .hdf5 files
        output_dir: Directory to save augmented .hdf5 files
        num_keyframes: Number of keyframes for VGGT
        device: Device to run VGGT on
        target_size: Target image size for VGGT
    """
    input_path = Path(input_dir) / suite_name
    output_path = Path(output_dir) / suite_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .hdf5 files
    hdf5_files = sorted(input_path.glob("*.hdf5"))
    
    if not hdf5_files:
        print(f"❌ No .hdf5 files found in {input_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing LIBERO Suite: {suite_name}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Found {len(hdf5_files)} dataset files")
    print(f"Image resize: 128x128 → {target_size}x{target_size}")
    print(f"{'='*60}\n")
    
    # Process each file
    successful = 0
    failed = 0
    
    for hdf5_file in hdf5_files:
        output_file = output_path / hdf5_file.name
        
        if output_file.exists():
            print(f"⏭️  Skipping {hdf5_file.name} (already exists)")
            continue
        
        try:
            process_libero_dataset(
                dataset_path=str(hdf5_file),
                output_path=str(output_file),
                num_keyframes=num_keyframes,
                device=device,
                target_size=target_size,
            )
            successful += 1
        except Exception as e:
            print(f"\n❌ Error processing {hdf5_file.name}:")
            print(f"   {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  ✓ Successful: {successful}")
    print(f"  ✗ Failed: {failed}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute VGGT features for LIBERO dataset"
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
        help="LIBERO suite name"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="LIBERO/libero/datasets",
        help="Directory containing LIBERO datasets"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="LIBERO/libero/datasets_with_vggt",
        help="Directory to save augmented datasets"
    )
    parser.add_argument(
        "--num_keyframes",
        type=int,
        default=3,
        help="Number of keyframes to use for VGGT (default: 3)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run VGGT on"
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=224,
        help="Target image size for VGGT (must be divisible by 14, default: 224)"
    )
    
    args = parser.parse_args()
    
    # Validate target_size
    if args.target_size % 14 != 0:
        raise ValueError(f"target_size must be divisible by 14, got {args.target_size}")
    
    process_entire_suite(
        suite_name=args.suite,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_keyframes=args.num_keyframes,
        device=args.device,
        target_size=args.target_size,
    )


if __name__ == "__main__":
    main()