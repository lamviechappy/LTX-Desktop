# test_temp_2.py
import torch
from pathlib import Path
from transformers import AutoConfig, Gemma3TextConfig, Gemma3ForCausalLM
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from ltx_core.text_encoders.gemma.encoders.encoder_configurator import GEMMA_LLM_KEY_OPS_1B
from ltx_core.text_encoders.gemma.config import GEMMA3_CONFIG_FOR_LTX

gemma_root = '/Users/donald/Library/Application Support/LTXDesktop/models/gemma-3-1b-it'
sf_path = str(Path(gemma_root) / 'model.safetensors')

raw_config = AutoConfig.from_pretrained(gemma_root, local_files_only=True)
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

loader = SafetensorsModelStateDictLoader()
sd = loader.load([sf_path], sd_ops=GEMMA_LLM_KEY_OPS_1B, device=torch.device('cpu'))

missing_keys, unexpected_keys = model.load_state_dict(sd.sd, strict=False, assign=True)
print(f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")

# Print tên 3 buffers còn meta để biết chúng là gì
still_meta_b = [(n, b.shape, b.dtype) for n, b in model.named_buffers() 
                if b is not None and b.device.type == 'meta']
print(f"Meta buffers ({len(still_meta_b)}):")
for name, shape, dtype in still_meta_b:
    print(f"  {name}: {shape} {dtype}")

# Materialize meta buffers
for name, buf in list(model.named_buffers()):
    if buf is not None and buf.device.type == 'meta':
        parts = name.split('.')
        module = model
        for part in parts[:-1]:
            module = getattr(module, part)
        module.register_buffer(parts[-1],
            torch.zeros(buf.shape, dtype=buf.dtype, device='cpu'))

# Verify
still_meta_p = [n for n, p in model.named_parameters() if p.device.type == 'meta']
still_meta_b = [n for n, b in model.named_buffers() if b is not None and b.device.type == 'meta']
print(f"Still meta: params={len(still_meta_p)}, buffers={len(still_meta_b)}")

if not still_meta_p and not still_meta_b:
    model = model.to(torch.device('mps'))
    print("✅ Model loaded on MPS!")