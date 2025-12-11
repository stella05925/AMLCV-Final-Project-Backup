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
    """π₀ enhanced with VGGT scene understanding via LoRA adapters."""
    
    def __init__(self, config: pi0_config.Pi0WithVGGTConfig, rngs: nnx.Rngs):
        # Initialize base Pi0 model first
        super().__init__(config, rngs)
        
        # Store VGGT config
        self.vggt_config = config
        
        if not config.use_vggt_features:
            logger.info("VGGT features disabled")
            return
        
        # Get embedding dimension - use config directly instead of inspecting model
        # The PaliGemma hidden size is known from the config
        if config.paligemma_variant.startswith("gemma_2b"):
            self.embed_dim = 2048
        elif config.paligemma_variant.startswith("gemma_7b"):
            self.embed_dim = 3072
        else:
            # Default for most PaliGemma variants
            self.embed_dim = 2048
        
        # LoRA adapter: VGGT features → PaliGemma space
        self.vggt_lora_down = nnx.Linear(
            config.vggt_feature_dim,
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
        
        # Optional: Learned gating mechanism
        if config.use_vggt_gating:
            self.vggt_gate = nnx.Sequential(
                nnx.Linear(config.vggt_feature_dim + self.embed_dim, 128, rngs=rngs),
                lambda x: nnx.swish(x),
                nnx.Linear(128, 1, rngs=rngs),
            )
        
        logger.info(f"Pi0WithVGGT initialized:")
        logger.info(f"  VGGT dim: {config.vggt_feature_dim}")
        logger.info(f"  PaliGemma dim: {self.embed_dim}")
        logger.info(f"  LoRA rank: {config.vggt_lora_rank}")
        logger.info(f"  Gating: {config.use_vggt_gating}")
    
    @at.typecheck
    def fuse_vggt_features(
        self,
        tokens: at.Float[at.Array, "b s emb"],
        vggt_features: at.Float[at.Array, "b vggt_dim"],
    ) -> at.Float[at.Array, "b s emb"]:
        """
        Fuse VGGT scene features with π₀ tokens using LoRA.
        
        Args:
            tokens: π₀ prefix tokens [batch, seq_len, embed_dim]
            vggt_features: VGGT scene features [batch, vggt_dim]
        
        Returns:
            Fused tokens [batch, seq_len, embed_dim]
        """
        # LoRA low-rank adaptation
        vggt_down = self.vggt_lora_down(vggt_features)  # [B, rank]
        vggt_adapted = self.vggt_lora_up(vggt_down)     # [B, embed_dim]
        
        # Broadcast to sequence length
        vggt_adapted = vggt_adapted[:, None, :]  # [B, 1, embed_dim]
        vggt_adapted = jnp.broadcast_to(
            vggt_adapted,
            tokens.shape
        )  # [B, S, embed_dim]
        
        if self.vggt_config.use_vggt_gating:
            # Learned gating: decide how much VGGT to use per token
            gate_input = jnp.concatenate([
                tokens,
                jnp.broadcast_to(
                    vggt_features[:, None, :],
                    (*tokens.shape[:2], self.vggt_config.vggt_feature_dim)
                )
            ], axis=-1)  # [B, S, embed_dim + vggt_dim]
            
            gate = nnx.sigmoid(self.vggt_gate(gate_input))  # [B, S, 1]
            
            # Gated residual connection
            fused = tokens + gate * vggt_adapted
        else:
            # Simple residual connection
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