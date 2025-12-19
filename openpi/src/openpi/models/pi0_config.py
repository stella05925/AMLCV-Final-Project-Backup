import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    vggt_compressed_dim: int = 256

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)

@dataclasses.dataclass(frozen=True, kw_only=True)
class Pi0WithVGGTConfig(Pi0Config):
    """Configuration for π₀ model augmented with VGGT features."""
    
    # VGGT-specific parameters
    use_vggt_features: bool = True
    vggt_feature_dim: int = 2048  # Dimension of VGGT latent features
    vggt_compressed_dim: int | None = 512  # Set to None to disable compression
    vggt_lora_rank: int = 16  # LoRA rank for VGGT fusion
    use_vggt_gating: bool = True  # Use learned gating mechanism
    
    def __post_init__(self):
        """Validate configuration."""
        # Call parent's __post_init__ if it exists
        if hasattr(super(), '__post_init__'):
            super().__post_init__()
        
        # Validate VGGT config
        if self.use_vggt_features:
            assert self.vggt_feature_dim > 0, "vggt_feature_dim must be positive"
            assert self.vggt_lora_rank > 0, "vggt_lora_rank must be positive"
            if self.vggt_compressed_dim is not None:
                assert self.vggt_compressed_dim > 0, "vggt_compressed_dim must be positive"
                assert self.vggt_compressed_dim < self.vggt_feature_dim, \
                    f"Compression dim ({self.vggt_compressed_dim}) should be < feature dim ({self.vggt_feature_dim})"
    
    @override
    @property
    def model_type(self) -> _model.ModelType:
        """Return PI0 model type (VGGT is just an augmentation)."""
        return _model.ModelType.PI05 if self.pi05 else _model.ModelType.PI0
    
    @override
    def create_model(self, rngs: nnx.Rngs) -> _model.BaseModel:
        """Create Pi0WithVGGT model instance."""
        # Import here to avoid circular dependency
        from openpi.models.pi0_with_vggt import Pi0WithVGGT
        return Pi0WithVGGT(self, rngs)

    @override
    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """
        Get freeze filter for LoRA + VGGT training.
        
        Returns a filter that FREEZES everything EXCEPT:
        - LoRA parameters in PaliGemma and ActionExpert (from base Pi0Config)
        - VGGT-specific parameters:
            * vggt_lora_down, vggt_lora_up (LoRA adapters)
            * vggt_gate (optional gating mechanism)
            * latent_compressor (optional compression layer)
        
        Returns:
            Filter that returns True for parameters to FREEZE
        """
        # Get the base LoRA freeze filter from parent class
        base_freeze_filter = super().get_freeze_filter()
        
        # Create VGGT-specific trainable filter
        def is_vggt_trainable(path, value):
            """Returns True if this is a VGGT parameter that should be trained."""
            path_str = "/".join(str(p) for p in path)
            return any([
                "vggt_lora_down" in path_str,
                "vggt_lora_up" in path_str,
                "vggt_gate" in path_str,
                "latent_compressor" in path_str,  # Add compression layer
            ])
        
        # Combine filters:
        # - base_freeze_filter returns True for params to FREEZE
        # - nnx.Not(base_freeze_filter) returns True for params to TRAIN (base LoRA)
        # - is_vggt_trainable returns True for VGGT params to TRAIN
        # - nnx.Any(...) returns True if EITHER should be trained
        # - nnx.Not(nnx.Any(...)) returns True for params to FREEZE
        
        trainable_filter = nnx.Any(
            nnx.Not(base_freeze_filter),  # Base LoRA params (trainable)
            is_vggt_trainable             # VGGT params (trainable)
        )
        
        # Return the freeze filter (inverse of trainable)
        return nnx.Not(trainable_filter)