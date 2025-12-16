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


class Pi0WithVGGT(Pi0):
    def __init__(self, config: pi0_config.Pi0WithVGGTConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        
        self.vggt_config = config
        paligemma_config = self.PaliGemma.llm.configs[0]
        self.embed_dim = paligemma_config.width
        
        if config.use_vggt_features:
            # VGGT aggregator outputs 2048D latent features
            self.vggt_latent_dim = config.vggt_feature_dim  # 2048
            
            # Compression layer (2048D → compressed_dim)
            self.latent_compressor = nnx.Linear(
                self.vggt_latent_dim,
                config.vggt_compressed_dim,
                rngs=rngs,
                use_bias=True,
            )
            
            # LoRA layers on compressed features
            self.vggt_lora_down = nnx.Linear(
                config.vggt_compressed_dim,
                config.vggt_lora_rank,
                rngs=rngs,
                use_bias=False,
            )
            self.vggt_lora_up = nnx.Linear(
                config.vggt_lora_rank,
                self.embed_dim,  # π₀ embedding dimension
                rngs=rngs,
                use_bias=False,
            )
            
            # Optional gating mechanism
            if config.use_vggt_gating:
                fusion_dim = config.vggt_compressed_dim
                self.vggt_gate = nnx.Sequential(
                    nnx.Linear(fusion_dim + self.embed_dim, 128, rngs=rngs),
                    lambda x: nnx.swish(x),
                    nnx.Linear(128, 1, rngs=rngs),
                )
            
            logger.info(f"Pi0WithVGGT initialized:")
            logger.info(f"  VGGT latent dim: {self.vggt_latent_dim}")
            logger.info(f"  Compressed dim: {config.vggt_compressed_dim}")
            logger.info(f"  LoRA rank: {config.vggt_lora_rank}")
            logger.info(f"  π₀ embed dim: {self.embed_dim}")
            logger.info(f"  Gating enabled: {config.use_vggt_gating}")
    
    def fuse_vggt_features(
        self,
        tokens: at.Float[at.Array, "b s emb"],
        vggt_features: at.Float[at.Array, "b 2048"],
    ) -> at.Float[at.Array, "b s emb"]:
        """Fuse VGGT latent features with π₀ tokens using LoRA."""
        
        # Log first time only
        if not hasattr(self, '_logged_vggt'):
            logger.info(f"VGGT fusion active!")
            logger.info(f"  Input tokens shape: {tokens.shape}")
            logger.info(f"  VGGT features shape: {vggt_features.shape}")
            logger.info(f"  VGGT features mean: {vggt_features.mean():.4f}")
            logger.info(f"  VGGT features std: {vggt_features.std():.4f}")
            self._logged_vggt = True
        
        # Compress VGGT latent features
        vggt_compressed = self.latent_compressor(vggt_features)  # [B, 2048] → [B, 512]
        
        # LoRA fusion
        vggt_down = self.vggt_lora_down(vggt_compressed)  # [B, 512] → [B, rank]
        vggt_adapted = self.vggt_lora_up(vggt_down)       # [B, rank] → [B, embed_dim]
        
        # Broadcast to sequence length
        vggt_adapted = vggt_adapted[:, None, :]  # [B, 1, embed_dim]
        vggt_adapted = jnp.broadcast_to(vggt_adapted, tokens.shape)  # [B, s, embed_dim]
        
        # Apply gating if enabled
        if self.vggt_config.use_vggt_gating:
            # Concatenate tokens with compressed VGGT features for gating
            gate_input = jnp.concatenate([
                tokens,  # [B, s, embed_dim]
                jnp.broadcast_to(
                    vggt_compressed[:, None, :],  # [B, 1, compressed_dim]
                    (*tokens.shape[:2], vggt_compressed.shape[-1])  # [B, s, compressed_dim]
                )
            ], axis=-1)  # [B, s, embed_dim + compressed_dim]
            
            gate = nnx.sigmoid(self.vggt_gate(gate_input))  # [B, s, 1]
            fused = tokens + gate * vggt_adapted
        else:
            # Simple additive fusion
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