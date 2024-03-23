from plotting.action_prediction_task_plots import *
from analysis.analyze_model import AnalyzeModel
import matplotlib.pyplot as plt
import os
import json

FIGURE_OUTPUT_FOLDER = '../figures/'
FOLDER_PATH = FIGURE_OUTPUT_FOLDER + '/action_prediction_task/'
os.makedirs(FOLDER_PATH, exist_ok=True)

# Compare Datasets/Models
exp1 = 'model1_0'
exp2 = 'model2_0'
d1 = {'n_obj': 8, 'dsize': 8000, 'dataset_name': 'high_contrast_same_size_test_set', 'set_num': 0}
d2 = {'n_obj': 8, 'dsize': 8000, 'dataset_name': 'random_contrast_random_sizes_test_set', 'set_num': 0}

a1 = AnalyzeModel.load_analysis(experiment_name=exp1, dataset_params=d1, target_file='correct_per_action.json')
a2 = AnalyzeModel.load_analysis(experiment_name=exp2, dataset_params=d2, target_file='correct_per_action.json')

fig, axes = plt.subplots(1, 3, constrained_layout=True)
fig.set_size_inches(12, 4)

_, n1 = plot_action_error(a1, axes, 3, 'Model 1')
_, n2 = plot_action_error(a2, axes, 3, 'Model 2')

# add chance level
take_chance = [2/3] * 7 + [1/2]
shake_chance = [1/2] + [2/3] * 7 + [1/2]
put_chance = [1/2] + [2/3] * 7
# take_chance = [2/3] * 8
# shake_chance = [2/3] * 9
# put_chance = [2/3] * 8
axes[0].plot(range(1, len(take_chance) + 1), take_chance, '--', color='red', label='chance')
axes[1].plot(range(len(shake_chance)), shake_chance, '--', color='red')
axes[2].plot(range(len(put_chance)), put_chance, '--', color='red')

axes[0].legend()
axes[0].set_ylabel('Error', fontsize=20)
axes[0].set_xlabel('Number of Objects before Action', fontsize=15)
axes[1].set_yticks([])
axes[2].set_yticks([])
fig.savefig(FOLDER_PATH + 'action_prediction_task_error_model_comparison.pdf')
with open(FOLDER_PATH + 'action_prediction_task_error_model_comparison.txt', 'w') as f:
    nums = np.concatenate(n1 + n2)
    json.dump([n1, n2, int(min(nums)), int(max(nums))], f, indent=1)


# Compare Training Limit
exp1 = 'model2_0'
exp2 = 'model2_5obj_0'
exp3 = 'model2_8obj_0'
d2 = {'n_obj': 8, 'dsize': 8000, 'dataset_name': 'random_contrast_random_sizes_test_set', 'set_num': 0}

a1 = AnalyzeModel.load_analysis(experiment_name=exp1, dataset_params=d2, target_file='correct_per_action.json')
a2 = AnalyzeModel.load_analysis(experiment_name=exp2, dataset_params=d2, target_file='correct_per_action.json')
a3 = AnalyzeModel.load_analysis(experiment_name=exp3, dataset_params=d2, target_file='correct_per_action.json')

fig, axes = plt.subplots(1, 3, constrained_layout=True)
fig.set_size_inches(12, 4)

_, n1 = plot_action_error(a1, axes, 3, '(3)', plot_params={'training_limit_shadow_visible': False,
                                                           'post_split_marker': '--o'})

_, n2 = plot_action_error(a2, axes, 5, '(5)', plot_params={'training_limit_shadow_visible': False,
                                                           'post_split_marker': '--o'})

_, n3 = plot_action_error(a3, axes, 8, '(8)', plot_params={'training_limit_shadow_visible': False,
                                                           'post_split_marker': '--o'})

axes[0].legend()
axes[0].set_ylabel('Error', fontsize=20)
axes[0].set_xlabel('Number of Objects before Action', fontsize=15)
fig.savefig(FOLDER_PATH + 'action_prediction_task_error_training_limit_comparison.pdf')
with open(FOLDER_PATH + 'action_prediction_task_error_training_limit_comparison.txt', 'w') as f:
    nums = np.concatenate(n1 + n2 + n3).astype(int)
    json.dump([n1, n2, n3, int(min(nums)), int(max(nums))], f, indent=1)

# Compare Embedding Dimension
exp1 = 'model2_ed=1'
exp2 = 'model2_ed=2'
exp3 = 'model2_ed=8'
exp4 = 'model2_ed=16'
exp5 = 'model2_ed=64'
exp6 = 'model2_ed=256'
d2 = {'n_obj': 8, 'dsize': 8000, 'dataset_name': 'random_contrast_random_sizes_test_set', 'set_num': 0}

a1 = AnalyzeModel.load_analysis(experiment_name=exp1, dataset_params=d2, target_file='correct_per_action.json')
a2 = AnalyzeModel.load_analysis(experiment_name=exp2, dataset_params=d2, target_file='correct_per_action.json')
a3 = AnalyzeModel.load_analysis(experiment_name=exp3, dataset_params=d2, target_file='correct_per_action.json')
a4 = AnalyzeModel.load_analysis(experiment_name=exp4, dataset_params=d2, target_file='correct_per_action.json')
a5 = AnalyzeModel.load_analysis(experiment_name=exp5, dataset_params=d2, target_file='correct_per_action.json')
a6 = AnalyzeModel.load_analysis(experiment_name=exp6, dataset_params=d2, target_file='correct_per_action.json')

fig, ax = plt.subplots(1, 1, constrained_layout=True)
fig.set_size_inches(8, 4)

ax, num_samples = plot_action_error_against_embedding_dimension([a1, a2, a3, a4, a5, a6], [1, 2, 8, 16, 64, 256], ax, 3)
ax.set_ylabel('Error', fontsize=16)
ax.set_xlabel('Embedding Dimension', fontsize=16)

fig.savefig(FOLDER_PATH + 'action_prediction_task_error_embedding_dimension_comparison.pdf')
with open(FOLDER_PATH + 'action_prediction_task_error_embedding_dimension_comparison.txt', 'w') as f:
    v = list(num_samples.values())
    nums = np.concatenate(v[0] + v[1] + v[2])
    json.dump([num_samples, int(min(nums)), int(max(nums))], f, indent=1)