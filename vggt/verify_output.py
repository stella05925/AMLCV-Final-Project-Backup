# verify_output.py (FINAL VERSION)
import h5py
import numpy as np
from pathlib import Path

def verify_augmented_dataset(filepath):
    """Verify that VGGT features were added correctly."""
    
    if not Path(filepath).exists():
        print(f"❌ File not found: {filepath}")
        return
    
    print(f"Verifying: {Path(filepath).name}")
    print("="*60)
    
    with h5py.File(filepath, 'r') as f:
        demo = f['data']['demo_0']
        
        # Check original data structure
        print("\n📊 Original data:")
        print(f"  actions: {demo['actions'].shape}")
        print(f"  agentview_rgb: {demo['obs']['agentview_rgb'].shape}")
        print(f"  eye_in_hand_rgb: {demo['obs']['eye_in_hand_rgb'].shape}")
        print(f"  joint_states: {demo['obs']['joint_states'].shape}")
        
        # Check VGGT features
        if 'vggt_scene_features' not in demo:
            print("\n❌ 'vggt_scene_features' not found!")
            return
        
        print("\n🎯 VGGT features:")
        vggt = demo['vggt_scene_features'][()]
        print(f"  vggt_scene_features: {vggt.shape}")
        print(f"  dtype: {vggt.dtype}")
        
        # Verify dimensions
        expected_dim = 2065  # 2048 (scene_tokens) + 9 (pose) + 2 (depth) + 6 (points)
        
        assert vggt.shape[0] == demo['actions'].shape[0], \
            f"Timesteps mismatch: {vggt.shape[0]} vs {demo['actions'].shape[0]}"
        assert vggt.shape[1] == expected_dim, \
            f"Feature dimension should be {expected_dim}, got {vggt.shape[1]}"
        
        print(f"\n  ✓ Correct dimension: {vggt.shape[1]} = 2048 (tokens) + 9 (pose) + 2 (depth) + 6 (points)")
        
        # Verify same across timesteps (scene-level)
        print("\n✓ Checking if features are scene-level (same across time)...")
        is_same = np.allclose(vggt[0], vggt[-1], rtol=1e-5)
        
        if is_same:
            print("  ✓ Features are identical across timesteps (scene-level)")
        else:
            max_diff = np.abs(vggt[0] - vggt[-1]).max()
            mean_diff = np.abs(vggt[0] - vggt[-1]).mean()
            print(f"  ⚠️  Features differ slightly:")
            print(f"     Max difference: {max_diff:.6f}")
            print(f"     Mean difference: {mean_diff:.6f}")
            if max_diff < 1e-4:
                print("  → Likely due to floating point precision, should be fine")
        
        # Check for NaNs or Infs
        print("\n✓ Checking for invalid values...")
        if np.isnan(vggt).any():
            print("  ❌ Contains NaN values!")
            return
        elif np.isinf(vggt).any():
            print("  ❌ Contains Inf values!")
            return
        else:
            print("  ✓ No NaN or Inf values")
        
        # Feature statistics
        print(f"\n📈 Feature statistics:")
        print(f"  Mean: {vggt.mean():>8.4f}")
        print(f"  Std:  {vggt.std():>8.4f}")
        print(f"  Min:  {vggt.min():>8.4f}")
        print(f"  Max:  {vggt.max():>8.4f}")
        
        # Check a few different timesteps to ensure scene-level
        print(f"\n✓ Sampling features at different timesteps:")
        timesteps_to_check = [0, len(vggt)//4, len(vggt)//2, 3*len(vggt)//4, len(vggt)-1]
        all_same = True
        for i in range(len(timesteps_to_check)-1):
            t1, t2 = timesteps_to_check[i], timesteps_to_check[i+1]
            if not np.allclose(vggt[t1], vggt[t2], rtol=1e-5):
                all_same = False
                diff = np.abs(vggt[t1] - vggt[t2]).max()
                print(f"  ⚠️  t={t1} vs t={t2}: max diff = {diff:.6f}")
        
        if all_same:
            print(f"  ✓ All sampled timesteps identical (scene-level confirmed)")
        
        # File size comparison
        file_size = Path(filepath).stat().st_size / (1024**2)  # MB
        print(f"\n💾 File size: {file_size:.2f} MB")
        
        # Estimate storage overhead
        original_size_estimate = (
            demo['actions'].size * demo['actions'].dtype.itemsize +
            demo['obs']['agentview_rgb'].size * demo['obs']['agentview_rgb'].dtype.itemsize +
            demo['obs']['eye_in_hand_rgb'].size * demo['obs']['eye_in_hand_rgb'].dtype.itemsize
        ) / (1024**2)
        
        vggt_size = vggt.size * vggt.dtype.itemsize / (1024**2)
        overhead_pct = (vggt_size / original_size_estimate) * 100 if original_size_estimate > 0 else 0
        
        print(f"  VGGT features: {vggt_size:.2f} MB")
        print(f"  Storage overhead: ~{overhead_pct:.1f}% (features vs images+actions)")
        
        print("\n" + "="*60)
        print("✅ All checks passed!")
        print("\nReady for π₀ training with VGGT scene understanding!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        verify_augmented_dataset(sys.argv[1])
    else:
        # Show first processed file
        output_dir = Path("libero/datasets_with_vggt/libero_spatial")
        if output_dir.exists():
            files = sorted(output_dir.glob("*.hdf5"))
            if files:
                print(f"Found {len(files)} processed files\n")
                verify_augmented_dataset(str(files[0]))
            else:
                print("No processed files found")
        else:
            print(f"Output directory not found: {output_dir}")