# test_debug_loader.py - chạy cái này trước để xem raw keys trong safetensors
from safetensors import safe_open
from pathlib import Path

gemma_root = '/Users/donald/Library/Application Support/LTXDesktop/models/gemma-3-1b-it'
sf_path = str(Path(gemma_root) / 'model.safetensors')

print("=== RAW KEYS IN SAFETENSORS ===")
with safe_open(sf_path, framework="pt", device="cpu") as f:
    keys = list(f.keys())
    print(f"Total keys: {len(keys)}")
    print("First 10:", keys[:10])
    print("Last 5:", keys[-5:])