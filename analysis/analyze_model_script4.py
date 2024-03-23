from analysis.analyze_model import AnalyzeModel

pair_datasets = [
    {'n_obj': 8, 'dsize': 8000, 'dataset_name': 'random_contrast_random_sizes_test_set', 'set_num': 0}
]

# single_datasets = [
#     {'n_obj': 30, 'dsize': 150, 'dataset_name': 'random_contrast_random_sizes_test_set', 'set_num': 1}
# ]


dims = [1, 2, 8, 16, 64, 256]
for dim in dims:
    exp_name = f'model2_ed={dim}'
    AnalyzeModel(experiment_name=exp_name, pair_datasets=pair_datasets, single_datasets=[])

