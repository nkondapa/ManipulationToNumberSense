from analysis.analyze_model import AnalyzeModel

pair_datasets = [
    {'n_obj': 8, 'dsize': 8000, 'dataset_name': 'random_contrast_random_sizes_test_set', 'set_num': 0}
]

single_datasets = [
    {'n_obj': 30, 'dsize': 150, 'dataset_name': 'random_contrast_random_sizes_test_set', 'set_num': 1}
]


num_objects = [5, 8]
for no in num_objects:
    exp_name = f'model2_{no}obj_0'
    AnalyzeModel(experiment_name=exp_name, pair_datasets=pair_datasets, single_datasets=single_datasets)


