from safetensors import safe_open
path = '/Users/donald/Library/Application Support/LTXDesktop/models/ltxv-2b-0.9.8-distilled.safetensors'
with safe_open(path, framework='pt', device='cpu') as f:
    meta = f.metadata()
    print('Metadata:', meta)
    keys = list(f.keys())
    print(f'Total keys: {len(keys)}')
    # tất cả top-level prefixes
    prefixes = set(k.split('.')[0] for k in keys)
    print('Top-level prefixes:', prefixes)