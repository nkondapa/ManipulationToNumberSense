
paths = [
    'model1.py',
    'model2.py',
    'model2_5obj.py',
    'model2_8obj.py',
    'model2_action_size.py',
    'model2_emb_dim.py',
    'model2_jitter.py',
    'model2_jitter_5obj.py',
    'model2_jitter_8obj.py',
    'model2_jitter_action_size.py',
    'model2_jitter_emb_dim.py',
]


for path in paths:

    try:
        exec(open(path).read())

    except Exception as e:
        print(f'{path} failed...')
