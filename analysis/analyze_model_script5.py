from analysis.analyze_model import AnalyzeModel

pair_datasets = [
    {'n_obj': 8, 'dsize': 8000, 'dataset_name': 'random_contrast_random_sizes_test_set', 'set_num': 0}
]

single_datasets = [
    {'n_obj': 30, 'dsize': 150, 'dataset_name': 'random_contrast_random_sizes_test_set', 'set_num': 1}
]

seeds = range(8)
for num_obj, act_size in [(3, 3), (5, 3), (5, 5)]:
    for seed in seeds:
        exp_name = f'model2_{seed}_num_obj={num_obj}_action_size={act_size}'
        AnalyzeModel(experiment_name=exp_name, pair_datasets=pair_datasets, single_datasets=single_datasets)
