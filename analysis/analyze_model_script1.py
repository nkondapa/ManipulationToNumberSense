from analysis.analyze_model import AnalyzeModel

pair_datasets = [
    {'n_obj': 8, 'dsize': 8000, 'dataset_name': 'random_contrast_random_sizes_test_set', 'set_num': 0}
]

single_datasets = [
    {'n_obj': 30, 'dsize': 150, 'dataset_name': 'random_contrast_random_sizes_test_set', 'set_num': 1}
]


seeds = list(range(0, 30))
for seed in seeds:
    exp_name = f'model2_{seed}'
    AnalyzeModel(experiment_name=exp_name, pair_datasets=pair_datasets, single_datasets=single_datasets)


pair_datasets = [
    {'n_obj': 8, 'dsize': 8000, 'dataset_name': 'jittered_contrast_jittered_sizes_test_set', 'set_num': 0}
]

single_datasets = [
    {'n_obj': 30, 'dsize': 150, 'dataset_name': 'jittered_contrast_jittered_sizes_test_set', 'set_num': 1}
]


noise_opts = [False]
noise_opt = noise_opts[0]
seeds = list(range(0, 30))
for cj, sj in [(5, 3)]:
    for seed in seeds:
        pair_datasets[0]['noise_opt'] = noise_opt
        single_datasets[0]['noise_opt'] = noise_opt
        pair_datasets[0]['color_jitter'] = 5
        pair_datasets[0]['size_jitter'] = 3
        exp_name = f'model2_{seed}_noise={False}_cj={cj}_sj={sj}'
        AnalyzeModel(experiment_name=exp_name, pair_datasets=pair_datasets, single_datasets=single_datasets)


pair_datasets = [
    {'n_obj': 8, 'dsize': 8000, 'dataset_name': 'high_contrast_same_size_test_set', 'set_num': 0}
]

single_datasets = [
    {'n_obj': 30, 'dsize': 150, 'dataset_name': 'high_contrast_same_size_test_set', 'set_num': 1}
]
single_datasets.extend(pair_datasets)

seeds = list(range(0, 30))
for seed in seeds:
    exp_name = f'model1_{seed}'
    AnalyzeModel(experiment_name=exp_name, pair_datasets=pair_datasets, single_datasets=single_datasets)
