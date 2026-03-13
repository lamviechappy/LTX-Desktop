import torch
from transformers import AutoConfig, Gemma3Config, Gemma3ForConditionalGeneration
from safetensors import safe_open
from pathlib import Path

gemma_root = '/Users/donald/Library/Application Support/LTXDesktop/models/gemma-3-1b-it'
raw_config = AutoConfig.from_pretrained(gemma_root, local_files_only=True)

# Build config như encoder_configurator đang làm
from ltx_core.text_encoders.gemma.config import GEMMA3_CONFIG_FOR_LTX
base = GEMMA3_CONFIG_FOR_LTX.to_dict()
text_fields = ['hidden_size','intermediate_size','num_attention_heads','num_hidden_layers',
               'num_key_value_heads','head_dim','sliding_window','sliding_window_pattern',
               'rope_theta','rope_local_base_freq','vocab_size']
for field in text_fields:
    if hasattr(raw_config, field):
        base['text_config'][field] = getattr(raw_config, field)
gemma_config = Gemma3Config.from_dict(base)

# Build meta model
with torch.device('meta'):
    model = Gemma3ForConditionalGeneration(gemma_config)

# Load và transform keys
from ltx_core.text_encoders.gemma.encoders.encoder_configurator import GEMMA_LLM_KEY_OPS_1B
sf_path = str(Path(gemma_root) / 'model.safetensors')
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
loader = SafetensorsModelStateDictLoader()
sd = loader.load([sf_path], sd_ops=GEMMA_LLM_KEY_OPS_1B, device=torch.device('cpu'))

# Check match
model_keys = set(dict(model.named_parameters()).keys())
sd_keys = set(sd.sd.keys())
missing = model_keys - sd_keys
extra = sd_keys - model_keys
print(f'Model params: {len(model_keys)}')
print(f'SD keys: {len(sd_keys)}')
print(f'Missing (still meta after load): {len(missing)}')
print('Missing sample:', list(missing)[:10])
print(f'Extra (in SD but not model): {len(extra)}')
print('Extra sample:', list(extra)[:5])