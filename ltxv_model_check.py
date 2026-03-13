
from safetensors import safe_open

# path = '/Users/donald/Library/Application Support/LTXDesktop/models/ltx-2-19b-dev-fp4.safetensors'

path = '/Users/donald/Library/Application Support/LTXDesktop/models/ltxv-2b-0.9.8-distilled.safetensors'

# paths = ['/Users/donald/Library/Application Support/LTXDesktop/models/ltx-2-19b-dev-fp4.safetensors', '/Users/donald/Library/Application Support/LTXDesktop/models/ltxv-2b-0.9.8-distilled.safetensors', '/Users/donald/Library/Application Support/LTXDesktop/models/ltx-2-spatial-upscaler-x2-1.0.safetensors']

# for path in paths:

#     with safe_open(path, framework='pt', device='cpu') as f:
#         keys = list(f.keys())
#         # Tìm aggregate_embed
#         agg_keys = [k for k in keys if 'aggregate_embed' in k or 'text_embedding' in k or 'embeddings_connector' in k]
#         print('Aggregate/embedding keys:')
#         for k in agg_keys:
#             t = f.get_tensor(k)
#             print(f'  {k}: {t.shape}')
#     print("========")

with safe_open(path, framework='pt', device='cpu') as f:
    keys = list(f.keys())
    # Tìm aggregate_embed
    agg_keys = [k for k in keys if 'aggregate_embed' in k or 'text_embedding' in k or 'embeddings_connector' in k]
    print('Aggregate/embedding keys:')
    for k in agg_keys:
        t = f.get_tensor(k)
        print(f'  {k}: {t.shape}')
    print("========")