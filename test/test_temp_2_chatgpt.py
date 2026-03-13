import torch
from transformers import AutoConfig, Gemma3Config, Gemma3ForCausalLM
from pathlib import Path
from safetensors.torch import load_file


# =========================================================
# 1. Paths
# =========================================================

gemma_root = "/Users/donald/Library/Application Support/LTXDesktop/models/gemma-3-1b-it"
sf_path = str(Path(gemma_root) / "model.safetensors")


# =========================================================
# 2. Load raw config from HF checkpoint
# =========================================================

raw_config = AutoConfig.from_pretrained(
    gemma_root,
    local_files_only=True
)

print("Loaded HF config")


# =========================================================
# 3. Build LTX Gemma text config
# =========================================================

from ltx_core.text_encoders.gemma.config import GEMMA3_CONFIG_FOR_LTX

base = GEMMA3_CONFIG_FOR_LTX.to_dict()

text_fields = [
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "head_dim",
    "sliding_window",
    "sliding_window_pattern",
    "rope_theta",
    "rope_local_base_freq",
    "vocab_size",
]

for field in text_fields:
    if hasattr(raw_config, field):
        base["text_config"][field] = getattr(raw_config, field)

from transformers import Gemma3TextConfig

text_cfg_dict = base["text_config"]

text_config = Gemma3TextConfig(
    **{
        k: v
        for k, v in text_cfg_dict.items()
        if k in Gemma3TextConfig.__init__.__code__.co_varnames
    }
)

print("Built Gemma3TextConfig")


# =========================================================
# 4. Create model on meta device (no memory allocation)
# =========================================================

with torch.device("meta"):
    model = Gemma3ForCausalLM(text_config)

print("Model initialized on meta")


# =========================================================
# 5. Inspect model parameters
# =========================================================

model_keys = set(dict(model.named_parameters()).keys())

print(f"Model params: {len(model_keys)}")


# =========================================================
# 6. Load safetensors directly
# =========================================================

sd = load_file(sf_path)

print("Loaded safetensors")

sd_keys = set(sd.keys())

print("Raw checkpoint keys:", len(sd_keys))
print(list(sd_keys)[:20])


# =========================================================
# 7. Fix tied weights (Gemma ties lm_head with embedding)
# =========================================================

if "model.embed_tokens.weight" in sd:
    sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
    print("Added lm_head.weight (tied embedding)")


# =========================================================
# 8. Move model structure to MPS without initializing weights
# =========================================================

device = torch.device("mps")

model = model.to_empty(device=device)

print("Model moved to MPS with empty tensors")


# =========================================================
# 9. Load weights
# =========================================================

missing_keys, unexpected_keys = model.load_state_dict(
    sd,
    strict=False
)

print("\nAfter load_state_dict:")
print("missing_keys:", len(missing_keys))
print("unexpected_keys:", len(unexpected_keys))


# =========================================================
# 10. Check if any meta tensors remain
# =========================================================

meta_params = [
    n for n, p in model.named_parameters()
    if p.device.type == "meta"
]

meta_buffers = [
    n for n, b in model.named_buffers()
    if b.device.type == "meta"
]

print("\nMeta params:", len(meta_params))
print("Meta buffers:", len(meta_buffers))


# =========================================================
# 11. Quick sanity check
# =========================================================

p = next(model.parameters())

print("\nSample parameter stats:")
print("device:", p.device)
print("mean:", p.mean().item())
print("std:", p.std().item())


print("\nModel loaded successfully on MPS!")