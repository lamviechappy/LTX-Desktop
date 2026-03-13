from safetensors import safe_open

path = '/Users/donald/Library/Application Support/LTXDesktop/models/ltx-2-19b-dev-fp4.safetensors'
with safe_open(path, framework='pt', device='cpu') as f:
    keys = list(f.keys())
    # Check aggregate_embed và caption_projection dims
    for k in keys:
        if any(x in k for x in ['aggregate_embed', 'caption_projection', 'embeddings_connector', 'text_embedding']):
            t = f.get_tensor(k)
            print(f'{k}: {t.shape}')