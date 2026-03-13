# test_debug_loader2.py
from safetensors import safe_open
from pathlib import Path
import torch

gemma_root = '/Users/donald/Library/Application Support/LTXDesktop/models/gemma-3-1b-it'
sf_path = str(Path(gemma_root) / 'model.safetensors')

# Test 1: Load KHÔNG dùng sd_ops
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
loader = SafetensorsModelStateDictLoader()

print("=== Test 1: No sd_ops ===")
sd_raw = loader.load([sf_path], device=torch.device('cpu'))
print(f"Keys: {len(sd_raw.sd)}")
print(f"Sample: {list(sd_raw.sd.keys())[:3]}")

# Test 2: Xem SDOps API thực sự có gì
print("\n=== Test 2: SDOps API ===")
from ltx_core.loader.sft_loader import SDOps
help(SDOps)