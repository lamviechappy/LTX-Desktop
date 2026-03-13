from safetensors import safe_open

path = '/Users/donald/Library/Application Support/LTXDesktop/models/ltxv-2b-0.9.8-distilled.safetensors'
with safe_open(path, framework='pt', device='cpu') as f:
    keys = list(f.keys())
    print(f'Total keys: {len(keys)}')
    print('First 30:')
    for k in keys[:30]:
        t = f.get_tensor(k)
        print(f'  {k}: {t.shape}')