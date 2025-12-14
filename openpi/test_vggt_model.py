# test_vggt_model.py
import jax
import flax.nnx as nnx
from openpi.models import pi0_config

# Create config
config = pi0_config.Pi0WithVGGTConfig(
    vggt_feature_dim=2065,
    vggt_lora_rank=16,
    use_vggt_gating=True,
)

# Initialize model
rngs = nnx.Rngs(42)
model = config.create_model(rngs)

print("✓ Model created successfully!")
print(f"VGGT feature dim: {config.vggt_feature_dim}")
print(f"LoRA rank: {config.vggt_lora_rank}")