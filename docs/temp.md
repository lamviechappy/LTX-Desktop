python3 -c "
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
loader = SafetensorsModelStateDictLoader()
# meta = loader.metadata('/Users/donald/Library/Application Support/LTXDesktop/models/ltx-video-2b-v0.9.5-distilled.safetensors')
meta = loader.metadata('/Users/donald/Library/Application Support/LTXDesktop/models/ltx-2-19b-dev-fp4.safetensors')
print(list(meta.keys())[:20])
print()
# tìm key liên quan gemma
for k,v in meta.items():
    if 'gemma' in k.lower() or 'hidden' in k.lower() or 'layer' in k.lower():
        print(f'{k}: {v}')
"


python3 -c "
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
loader = SafetensorsModelStateDictLoader()
meta = loader.metadata('/Users/donald/Library/Application Support/LTXDesktop/models/ltx-2-19b-dev-fp4.safetensors')
print(list(meta.keys())[:20])
print()
for k,v in meta.items():
    if 'gemma' in k.lower() or 'hidden' in k.lower() or 'layer' in k.lower():
        print(f'{k}: {v}')
"