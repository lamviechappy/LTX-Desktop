# test_ops.py
import torch
from transformers import AutoConfig, Gemma3TextConfig
from transformers.models.gemma3 import Gemma3ForCausalLM
from pathlib import Path

gemma_root = '/Users/donald/Library/Application Support/LTXDesktop/models/gemma-3-1b-it'

# Simulate GemmaTextEncoderConfigurator
from ltx_core.text_encoders.gemma.encoders.encoder_configurator import GemmaTextEncoderConfigurator
encoder = GemmaTextEncoderConfigurator.from_config({"gemma_root_path": gemma_root})

print(f"model type: {type(encoder.model).__name__}")
print(f"matcher result: {hasattr(encoder, 'model') and type(encoder.model).__name__ == 'Gemma3ForCausalLM'}")

# Check buffers trước khi populate
print("\nBuffers trước populate:")
for name, buf in encoder.model.named_buffers():
    print(f"  {name}: device={buf.device if buf is not None else 'None'}")

# Apply GEMMA_MODEL_OPS_1B
from ltx_core.text_encoders.gemma.encoders.encoder_configurator import GEMMA_MODEL_OPS_1B
print(f"\nGEMMA_MODEL_OPS_1B matcher on encoder: {GEMMA_MODEL_OPS_1B.matcher(encoder)}")

if GEMMA_MODEL_OPS_1B.matcher(encoder):
    GEMMA_MODEL_OPS_1B.mutator(encoder)
    print("Populated buffers successfully")
else:
    print("❌ Matcher FAILED - không apply được ops!")

# Check buffers sau populate
print("\nBuffers sau populate:")
for name, buf in encoder.model.named_buffers():
    print(f"  {name}: device={buf.device if buf is not None else 'None'}")