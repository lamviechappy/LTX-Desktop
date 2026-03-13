import torch
from transformers import AutoConfig, Gemma3TextConfig, Gemma3ForCausalLM
from pathlib import Path

gemma_root = '/Users/donald/Library/Application Support/LTXDesktop/models/gemma-3-1b-it'
raw_config = AutoConfig.from_pretrained(gemma_root, local_files_only=True)

from ltx_core.text_encoders.gemma.config import GEMMA3_CONFIG_FOR_LTX
base = GEMMA3_CONFIG_FOR_LTX.to_dict()
text_fields = ['hidden_size','intermediate_size','num_attention_heads','num_hidden_layers',
               'num_key_value_heads','head_dim','sliding_window','sliding_window_pattern',
               'rope_theta','rope_local_base_freq','vocab_size']
for field in text_fields:
    if hasattr(raw_config, field):
        base['text_config'][field] = getattr(raw_config, field)

text_config = Gemma3TextConfig(**{k: v for k, v in base['text_config'].items()
                                   if k in Gemma3TextConfig.__init__.__code__.co_varnames})

with torch.device('meta'):
    model = Gemma3ForCausalLM(text_config)

from ltx_core.text_encoders.gemma.encoders.encoder_configurator import GEMMA_LLM_KEY_OPS_1B
sf_path = str(Path(gemma_root) / 'model.safetensors')
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
loader = SafetensorsModelStateDictLoader()
sd = loader.load([sf_path], sd_ops=GEMMA_LLM_KEY_OPS_1B, device=torch.device('cpu'))

# === FIX: Remap keys ===
# SD có prefix "model.language_model." nhưng model expect "model."
remapped = {}
for k, v in sd.sd.items():
    new_key = k.replace('model.language_model.', 'model.')
    remapped[new_key] = v

# Drop lm_head nếu tied
remapped.pop('model.lm_head.weight', None)
remapped.pop('lm_head.weight', None)

# Debug overlap sau remap
model_params = set(dict(model.named_parameters()).keys())
model_buffers = set(dict(model.named_buffers()).keys())
model_all = model_params | model_buffers
sd_keys = set(remapped.keys())

print(f'Model params: {len(model_params)}, buffers: {len(model_buffers)}')
print(f'SD keys after remap: {len(sd_keys)}')
print(f'Overlap: {len(model_all & sd_keys)}')
print(f'Missing: {len(model_all - sd_keys)}')
print(f'Extra: {len(sd_keys - model_all)} -> {list(sd_keys - model_all)[:5]}')

# Load
missing_keys, unexpected_keys = model.load_state_dict(remapped, strict=False, assign=True)
print(f'\nAfter load: missing={len(missing_keys)}, unexpected={len(unexpected_keys)}')
if missing_keys:
    print('Missing sample:', missing_keys[:5])

# Materialize còn lại meta params
for name, param in list(model.named_parameters()):
    if param.device.type == 'meta':
        parts = name.split('.')
        module = model
        for part in parts[:-1]:
            module = getattr(module, part)
        setattr(module, parts[-1], torch.nn.Parameter(
            torch.zeros(param.shape, dtype=param.dtype, device='cpu')))

# Materialize meta BUFFERS (đây là bug trước!)
for name, buf in list(model.named_buffers()):
    if buf is not None and buf.device.type == 'meta':
        parts = name.split('.')
        module = model
        for part in parts[:-1]:
            module = getattr(module, part)
        module.register_buffer(parts[-1],
            torch.zeros(buf.shape, dtype=buf.dtype, device='cpu'))

# Verify
still_meta_p = [(n, p.shape) for n, p in model.named_parameters() if p.device.type == 'meta']
still_meta_b = [(n, b.shape) for n, b in model.named_buffers() if b is not None and b.device.type == 'meta']
print(f'Still meta params: {len(still_meta_p)}, buffers: {len(still_meta_b)}')

if not still_meta_p and not still_meta_b:
    model = model.to(torch.device('mps'))
    print('✅ Model loaded on MPS successfully!')
else:
    print('❌ Still meta:', still_meta_p[:3], still_meta_b[:3])