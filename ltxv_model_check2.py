from safetensors import safe_open

path = '/Users/donald/Library/Application Support/LTXDesktop/models/ltxv-2b-0.9.8-distilled.safetensors'
with safe_open(path, framework='pt', device='cpu') as f:
    keys = list(f.keys())
    # Tìm tất cả keys liên quan text encoding
    text_keys = [k for k in keys if any(x in k for x in 
        ['text', 'gemma', 'embed', 'caption', 'connector', 'aggregate', 'projection', 'language'])]
    print(f'Text-related keys ({len(text_keys)}):')
    for k in text_keys:
        t = f.get_tensor(k)
        print(f'  {k}: {t.shape}')