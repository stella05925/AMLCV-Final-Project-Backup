# openpi/models/pi0_with_vggt.py
"""π₀ model augmented with VGGT scene features via LoRA."""

import logging
import flax.nnx as nnx
import jax.numpy as jnp
from typing_extensions import override

from openpi.models.pi0 import Pi0
from openpi.models import pi0_config
from openpi.models import model as _model
from openpi.shared import array_typing as at

logger = logging.getLogger(__name__)


# openpi/models/pi0_with_vggt.py

class Pi0WithVGGT(Pi0):
    def __init__(self, config: pi0_config.Pi0WithVGGTConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        
        self.vggt_config = config
        paligemma_config = self.PaliGemma.llm.configs[0]
        self.embed_dim = paligemma_config.width
        
        if config.use_vggt_features:
            # Parse VGGT feature dimensions
            # Features are: [scene_tokens (2048) | pose (9) | depth (2) | points (6)]
            self.scene_token_dim = 2048
            self.aux_feature_dim = 9 + 2 + 6  # 17 dims
            
            # Option 1: Compress scene tokens first (RECOMMENDED)
            self.scene_compressor = nnx.Linear(
                self.scene_token_dim,
                config.vggt_compressed_dim,  # e.g., 256
                rngs=rngs,
                use_bias=True,
            )
            
            # Then LoRA on compressed features + auxiliary features
            total_compressed_dim = config.vggt_compressed_dim + self.aux_feature_dim
            
            self.vggt_lora_down = nnx.Linear(
                total_compressed_dim,
                config.vggt_lora_rank,
                rngs=rngs,
                use_bias=False,
            )
            self.vggt_lora_up = nnx.Linear(
                config.vggt_lora_rank,
                self.embed_dim,
                rngs=rngs,
                use_bias=False,
            )
            
            if config.use_vggt_gating:
                self.vggt_gate = nnx.Sequential(
                    nnx.Linear(total_compressed_dim + self.embed_dim, 128, rngs=rngs),
                    lambda x: nnx.swish(x),
                    nnx.Linear(128, 1, rngs=rngs),
                )
            
            logger.info(f"Pi0WithVGGT initialized:")
            logger.info(f"  Scene tokens: {self.scene_token_dim} → {config.vggt_compressed_dim}")
            logger.info(f"  Auxiliary features: {self.aux_feature_dim} (kept as-is)")
            logger.info(f"  Total compressed: {total_compressed_dim}")
            logger.info(f"  LoRA rank: {config.vggt_lora_rank}")
    
    def fuse_vggt_features(
        self,
        tokens: at.Float[at.Array, "b s emb"],
        vggt_features: at.Float[at.Array, "b 2065"],
    ) -> at.Float[at.Array, "b s emb"]:
        """Fuse VGGT scene features with π₀ tokens using LoRA."""
        
        if not hasattr(self, '_logged_vggt'):
        logger.info(f"VGGT fusion active!")
        logger.info(f"  Input tokens shape: {tokens.shape}")
        logger.info(f"  VGGT features shape: {vggt_features.shape}")
        logger.info(f"  Scene tokens: {vggt_features[:, :2048].mean():.4f}")
        logger.info(f"  Aux features: {vggt_features[:, 2048:].mean():.4f}")
        self._logged_vggt = True
        
        # Split VGGT features
        scene_tokens = vggt_features[:, :self.scene_token_dim]  # [B, 2048]
        aux_features = vggt_features[:, self.scene_token_dim:]  # [B, 17]
        
        # Compress scene tokens
        scene_compressed = self.scene_compressor(scene_tokens)  # [B, 256]
        
        # Concatenate with auxiliary features
        vggt_compressed = jnp.concatenate([scene_compressed, aux_features], axis=-1)  # [B, 273]
        
        # LoRA fusion
        vggt_down = self.vggt_lora_down(vggt_compressed)  # [B, rank]
        vggt_adapted = self.vggt_lora_up(vggt_down)  # [B, embed_dim]
        
        # Broadcast to sequence length
        vggt_adapted = vggt_adapted[:, None, :]
        vggt_adapted = jnp.broadcast_to(vggt_adapted, tokens.shape)
        
        if self.vggt_config.use_vggt_gating:
            gate_input = jnp.concatenate([
                tokens,
                jnp.broadcast_to(
                    vggt_compressed[:, None, :],
                    (*tokens.shape[:2], vggt_compressed.shape[-1])
                )
            ], axis=-1)
            gate = nnx.sigmoid(self.vggt_gate(gate_input))
            fused = tokens + gate * vggt_adapted
        else:
            fused = tokens + vggt_adapted
        
        return fused
    
    @override
    @at.typecheck
    def embed_prefix(
        self, 
        obs: _model.Observation
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"]
    ]:
        """
        Embed observation prefix with VGGT fusion.
        
        This overrides the parent method to add VGGT feature fusion.
        """
        # Get standard π₀ prefix embedding (images + language)
        tokens, input_mask, ar_mask = super().embed_prefix(obs)
        
        # Fuse VGGT features if available and enabled
        if (self.vggt_config.use_vggt_features and 
            hasattr(obs, 'vggt_scene_features') and 
            obs.vggt_scene_features is not None):
            tokens = self.fuse_vggt_features(tokens, obs.vggt_scene_features)
        
        return tokens, input_mask, ar_mask