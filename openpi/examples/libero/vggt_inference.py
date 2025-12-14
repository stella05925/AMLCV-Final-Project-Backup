"""Real-time VGGT feature extraction for LIBERO evaluation.

This module computes VGGT scene features on-the-fly during evaluation,
matching the features used during training.
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

from vggt.models.vggt import VGGT

logger = logging.getLogger(__name__)


class VGGTInference:
    """Compute VGGT scene features on-the-fly during evaluation."""

    def __init__(
        self,
        model_name: str = "facebook/VGGT-1B",
        device: str = "cuda",
        target_size: int = 224,
        num_keyframes: int = 3,
    ):
        """
        Initialize VGGT encoder for inference.

        Args:
            model_name: HuggingFace model name (default: "facebook/VGGT-1B")
            device: Device to run inference on ('cuda' or 'cpu')
            target_size: Image size for VGGT (must be divisible by 14, default: 224)
            num_keyframes: Number of keyframes to use (default: 3, not used in single-frame mode)
        """
        assert target_size % 14 == 0, f"target_size {target_size} must be divisible by 14"

        self.device = device
        self.target_size = target_size
        self.num_keyframes = num_keyframes

        # Determine dtype based on GPU capabilities
        if device == "cuda" and torch.cuda.is_available():
            device_capability = torch.cuda.get_device_capability()[0]
            self.dtype = torch.bfloat16 if device_capability >= 8 else torch.float16
        else:
            self.dtype = torch.float32

        # Load VGGT model from HuggingFace
        logger.info(f"Loading VGGT model '{model_name}' on {device} with dtype={self.dtype}...")
        self.model = VGGT.from_pretrained(model_name).to(device)
        self.model.eval()
        logger.info("✓ VGGT model loaded successfully")

    def encode_single_observation(
        self,
        agentview_image: np.ndarray,
        wrist_image: np.ndarray,
    ) -> np.ndarray:
        """
        Encode a single observation (2 camera views) into VGGT scene features.

        This matches the precomputation strategy where we treat one observation
        as a "scene" with 2 cameras.

        Args:
            agentview_image: Agent view RGB image [H, W, 3] in range [0, 255], uint8
            wrist_image: Wrist camera RGB image [H, W, 3] in range [0, 255], uint8

        Returns:
            features: VGGT scene encoding [2065,]
                - scene_tokens: 2048 (pooled over views and spatial tokens)
                - pose_enc: 9 (camera parameters)
                - depth_stats: 2 (mean, std)
                - point_stats: 6 (mean_xyz, std_xyz)
        """
        # Preprocess both images
        agentview_tensor = self._preprocess_image(agentview_image)  # [3, 224, 224]
        wrist_tensor = self._preprocess_image(wrist_image)          # [3, 224, 224]

        # Stack as sequence: [2, 3, 224, 224]
        images = torch.stack([agentview_tensor, wrist_tensor], dim=0)

        # Add batch dimension: [1, 2, 3, 224, 224]
        images = images.unsqueeze(0).to(self.device)

        # Run VGGT inference
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype, enabled=(self.device == "cuda")):
                # Get internal features from aggregator
                aggregated_tokens_list, ps_idx = self.model.aggregator(images)

                # Extract features from different heads
                scene_tokens = aggregated_tokens_list[-1]  # [B, S, num_tokens, 2048]
                pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]  # [B, S, 9]
                depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, images, ps_idx)  # [B, S, H, W, 1]
                point_map, point_conf = self.model.point_head(aggregated_tokens_list, images, ps_idx)  # [B, S, H, W, 3]

        # Pool features to fixed-size vector
        features = self._pool_vggt_features(
            scene_tokens=scene_tokens,
            pose_enc=pose_enc,
            depth_map=depth_map,
            point_map=point_map,
        )

        # Return as numpy array [2065,]
        return features.squeeze(0)  # Remove batch dimension

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert LIBERO observation image to VGGT input format.

        Args:
            image: numpy array [H, W, 3] in range [0, 255], uint8

        Returns:
            torch.Tensor: [3, target_size, target_size] in range [0, 1]
        """
        # Normalize to [0, 1]
        image_float = torch.from_numpy(image).float() / 255.0

        # Transpose to CHW format
        image_chw = image_float.permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]

        # Resize to target_size using bilinear interpolation
        if image_chw.shape[1] != self.target_size or image_chw.shape[2] != self.target_size:
            image_chw = F.interpolate(
                image_chw.unsqueeze(0),  # Add batch dim: [1, 3, H, W]
                size=(self.target_size, self.target_size),
                mode='bilinear',
                align_corners=False,
                antialias=True  # Better quality downsampling
            ).squeeze(0)  # Remove batch dim: [3, target_size, target_size]

        return image_chw

    def _pool_vggt_features(
        self,
        scene_tokens: torch.Tensor,  # [B, S, num_tokens, 2048]
        pose_enc: torch.Tensor,       # [B, S, 9]
        depth_map: torch.Tensor,      # [B, S, H, W, 1]
        point_map: torch.Tensor,      # [B, S, H, W, 3]
    ) -> np.ndarray:
        """
        Pool VGGT outputs to a fixed-size feature vector.

        This matches the pooling strategy used during precomputation.

        Args:
            scene_tokens: Internal scene representations [B, S, num_tokens, 2048]
            pose_enc: Camera pose encodings [B, S, 9]
            depth_map: Predicted depth maps [B, S, H, W, 1]
            point_map: 3D point predictions [B, S, H, W, 3]

        Returns:
            np.ndarray: [B, 2065] - pooled scene features
        """
        # 1. Pool scene tokens (most informative)
        # Average over sequence (S) and spatial tokens (num_tokens)
        tokens_pooled = scene_tokens.mean(dim=[1, 2])  # [B, 2048]

        # 2. Pool camera parameters
        pose_pooled = pose_enc.mean(dim=1)  # [B, 9]

        # 3. Pool depth statistics
        depth_mean = depth_map.mean(dim=[1, 2, 3])  # [B, 1]
        depth_std = depth_map.std(dim=[1, 2, 3])    # [B, 1]
        depth_features = torch.cat([depth_mean, depth_std], dim=-1)  # [B, 2]

        # 4. Pool point map statistics
        point_mean = point_map.mean(dim=[1, 2, 3])  # [B, 3]
        point_std = point_map.std(dim=[1, 2, 3])    # [B, 3]
        point_features = torch.cat([point_mean, point_std], dim=-1)  # [B, 6]

        # Concatenate all features
        all_features = torch.cat([
            tokens_pooled,     # [B, 2048]
            pose_pooled,       # [B, 9]
            depth_features,    # [B, 2]
            point_features,    # [B, 6]
        ], dim=-1)  # [B, 2065]

        return all_features.cpu().numpy()


def test_vggt_inference():
    """Test VGGT inference with dummy data."""
    print("Testing VGGT inference...")
    print("=" * 60)

    # Create encoder
    encoder = VGGTInference(device="cuda" if torch.cuda.is_available() else "cpu")

    # Create dummy observations (LIBERO resolution)
    dummy_agentview = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    dummy_wrist = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Test encoding
    print("\nEncoding observation...")
    features = encoder.encode_single_observation(dummy_agentview, dummy_wrist)

    print(f"✓ VGGT features shape: {features.shape}")
    print(f"✓ Feature stats: min={features.min():.3f}, max={features.max():.3f}, mean={features.mean():.3f}")

    # Verify dimensions
    assert features.shape == (2065,), f"Expected (2065,), got {features.shape}"
    assert not np.isnan(features).any(), "Features contain NaN!"
    assert not np.isinf(features).any(), "Features contain Inf!"

    print("\n" + "=" * 60)
    print("✅ VGGT inference working correctly!")
    print("\nFeature breakdown:")
    print(f"  - Scene tokens: {features[:2048].mean():.4f} (2048 dims)")
    print(f"  - Pose encoding: {features[2048:2057].mean():.4f} (9 dims)")
    print(f"  - Depth stats: {features[2057:2059]} (2 dims)")
    print(f"  - Point stats: {features[2059:2065].mean():.4f} (6 dims)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_vggt_inference()
