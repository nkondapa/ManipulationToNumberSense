from plotting.rescaled_embeddings_plots import *
from analysis.analyze_model import AnalyzeModel
import matplotlib.pyplot as plt
import os
import json

FIGURE_OUTPUT_FOLDER = '../figures/'
FOLDER_PATH = FIGURE_OUTPUT_FOLDER + '/perception_figures/'
os.makedirs(FOLDER_PATH, exist_ok=True)

d2 = {'n_obj': 30, 'dsize': 150, 'dataset_name': 'random_contrast_random_sizes_test_set', 'set_num': 1}
main_exp = 'model2_0'
main_embeddings = AnalyzeModel.load_analysis(experiment_name=main_exp, dataset_params=d2, target_file='embeddings.npy')
main_embedding_counts = AnalyzeModel.load_analysis(experiment_name=main_exp, dataset_params=d2,
                                                   target_file='embeddings_counts.npy')

experiments = ['model2_0', 'model2_1', 'model2_2', 'model2_3',
               'model2_4', 'model2_5', 'model2_6', 'model2_7']

model_fits = []
model_r2s = []
main_model_fit = None
main_model_r2 = None
for exp in experiments:
    embeddings = AnalyzeModel.load_analysis(experiment_name=exp, dataset_params=d2, target_file='embeddings.npy')
    embedding_counts = AnalyzeModel.load_analysis(experiment_name=exp, dataset_params=d2,
                                                  target_file='embeddings_counts.npy')
    rescaled_embeddings = AnalyzeModel.rescale_embeddings(embeddings, embedding_counts, 1)
    fit, r2 = AnalyzeModel.fit_power_function(rescaled_embeddings)
    if exp != main_exp:
        model_fits.append(fit)
        model_r2s.append(r2)
    else:
        main_model_fit = fit
        main_model_r2 = r2

fig, ax = plt.subplots(1, 1, constrained_layout=True)
rescaled_embeddings = AnalyzeModel.rescale_embeddings(main_embeddings, main_embedding_counts, 1)
generate_violinplot(rescaled_embeddings, ax, {'add_human_estimate': True,
                                              'add_multiple_model_fits': model_fits,
                                              'add_main_model_fit': main_model_fit,
                                              })

fig.savefig(FOLDER_PATH + 'perceived_numerosity.pdf')


fig, ax = plt.subplots(1, 1, constrained_layout=True)
generate_psychometric_curve(rescaled_embeddings, 16, list(range(0, 30)), ax)
_, shape_list = add_human_data_burr(ax)
d2['split_name'] = 'train.pt'
dataloader, _ = AnalyzeModel.load_dataloader(d2, 'as_singles')
m16 = np.where(embedding_counts == 16)[0][0]
m20 = np.where(embedding_counts == 20)[0][0]
m12 = np.where(embedding_counts == 12)[0][0]
# ax = add_reference_test_image_pairs(dataloader.dataset.data[[m12, m16, m20]], ax)
legend = ax.legend(fontsize=14, loc='center left')
plt.setp(legend.get_title(), fontsize=14)
with open(FOLDER_PATH + 'shape_list.txt', 'w') as f:
    [f.write(str(sl) + '\n') for sl in shape_list]

fig.savefig(FOLDER_PATH + 'proportion_more_figure.pdf')

fig, axes = plt.subplots(1, 3, constrained_layout=True)
generate_reference_test_image_triplet_panel(dataloader.dataset.data[[m12, m16, m20]], axes)
fig.set_size_inches(6, 2)
fig.savefig(FOLDER_PATH + 'comparison_task_image_panel.pdf')
