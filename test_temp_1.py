import torch
from transformers import AutoConfig, Gemma3Config, Gemma3ForCausalLM
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

# Dùng text_config trực tiếp thay vì full multimodal config
from transformers import Gemma3TextConfig
text_config = Gemma3TextConfig(**{k: v for k, v in base['text_config'].items() 
                                   if k in Gemma3TextConfig.__init__.__code__.co_varnames})

# Build model chỉ phần text
with torch.device('meta'):
    model = Gemma3ForCausalLM(text_config)

from ltx_core.text_encoders.gemma.encoders.encoder_configurator import GEMMA_LLM_KEY_OPS_1B
sf_path = str(Path(gemma_root) / 'model.safetensors')
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
loader = SafetensorsModelStateDictLoader()
sd = loader.load([sf_path], sd_ops=GEMMA_LLM_KEY_OPS_1B, device=torch.device('cpu'))

# Load với assign=True để tránh lỗi meta tensor
model.load_state_dict(sd.sd, strict=False, assign=True)
model = model.to(torch.device('mps'))  # hoặc 'cpu'
print("Loaded successfully!")