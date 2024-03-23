from plotting.embedding_space_plots import *
from analysis.analyze_model import AnalyzeModel
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import json
from matplotlib.gridspec import GridSpec

FOLDER_PATH = '../figures/embedding_plots/'
os.makedirs(FOLDER_PATH, exist_ok=True)


def generate_main_figure(exp_name, dataset_params, **kwargs):
    folder_path = FOLDER_PATH + f'/{exp_name}/'
    os.makedirs(folder_path, exist_ok=True)
    embeddings = AnalyzeModel.load_analysis(experiment_name=exp_name, dataset_params=dataset_params,
                                            target_file='embeddings.npy')
    embedding_counts = AnalyzeModel.load_analysis(experiment_name=exp_name, dataset_params=dataset_params,
                                                  target_file='embeddings_counts.npy')
    cluster_indices_dict, labels = AnalyzeModel.hdbscan_clustering(embeddings, min_cluster_size=50)
    inset_axes = kwargs.get('inset_axes', [0.55, 0.55, 0.4, 0.4])
    # Embeddings 1
    # Ground Truth
    random_subset = np.arange(embeddings.shape[0])
    # np.random.shuffle(random_subset)
    # random_subset = random_subset[0:1000]

    fig, ax = plt.subplots(1, 1)

    visualize_embedding_space_with_labels(embeddings[random_subset], embedding_counts[random_subset], ax,
                                          {'color_palette': 'crest',
                                           'inset': True,
                                           'inset_axes': inset_axes,
                                           })
    fig.savefig(folder_path + 'embedding_ground_truth.pdf')
    fig.savefig(folder_path + 'embedding_ground_truth.png', dpi=300)

    # Clustering
    fig, ax = plt.subplots(1, 1)
    visualize_embedding_space_with_labels(embeddings[random_subset], labels[random_subset], ax,
                                          {'color_palette': 'bright',
                                           'inset': True,
                                           'inset_limit': 5,
                                           'label_every': 1,
                                           'inset_axes': inset_axes,
                                           'inset_labels': ['A0', 'B0', 'C0', 'D0'],
                                           'apply_tight_layout': False,
                                           })
    fig.savefig(folder_path + 'embedding_unsupervised.pdf')

    # Confusion Matrix
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    fig.set_size_inches(13, 5)
    conf_matrix, column_names, row_names = AnalyzeModel.generate_confusion_matrix(labels, embedding_counts,
                                                                                  remove_uncertain=True, sort=True)
    visualize_confusion_matrix(conf_matrix, ax, {'column_names': column_names,
                                                 'row_names': row_names,
                                                 'transpose': True,
                                                 'cmap': 'Blues',
                                                 })
    fig.savefig(folder_path + 'embedding_confusion_matrix.pdf')

    summ_fig = plt.figure()
    gs = GridSpec(2, 2, figure=summ_fig)
    axes = [summ_fig.add_subplot(gs[0, 0]), summ_fig.add_subplot(gs[0, 1]), summ_fig.add_subplot(gs[1, :])]
    summ_fig.set_size_inches(15, 8)

    visualize_embedding_space_with_labels(embeddings[random_subset], embedding_counts[random_subset], axes[0],
                                          {'color_palette': 'crest',
                                           'inset': True,
                                           'inset_axes': inset_axes,
                                           'annotate': 'A',
                                           'apply_tight_layout': False,
                                           })

    visualize_embedding_space_with_labels(embeddings[random_subset], labels[random_subset], axes[1],
                                          {'color_palette': 'bright',
                                           'inset': True,
                                           'inset_limit': 5,
                                           'label_every': 1,
                                           'inset_axes': inset_axes,
                                           'annotate': 'B',
                                           'inset_labels': ['A0', 'B0', 'C0', 'D0'],
                                           })
    axes[0].set_ylabel('dim 2', fontsize=18)
    axes[0].set_xlabel('dim 1', fontsize=18)
    axes[1].set_xlabel('dim 1', fontsize=18)
    axes[1].set_ylabel('')

    visualize_confusion_matrix(conf_matrix, axes[2], {'column_names': column_names,
                                                      'row_names': row_names,
                                                      'transpose': True,
                                                      'cmap': 'Blues',
                                                      'annotate': 'C',
                                                      })
    plt.tight_layout()
    summ_fig.savefig(folder_path + 'embedding_summary_plot.pdf')
    plt.close()


def generate_supplemental_figure_zoomed_in(exp, dataset_params):
    # Load Datasets
    d1 = {'n_obj': 8, 'dsize': 8000, 'dataset_name': 'high_contrast_same_size_test_set', 'set_num': 0}
    d2 = {'n_obj': 8, 'dsize': 8000, 'dataset_name': 'random_contrast_random_sizes_test_set', 'set_num': 0}
    folder_path = FOLDER_PATH + f'/{exp}/'
    os.makedirs(folder_path, exist_ok=True)

    embeddings = AnalyzeModel.load_analysis(experiment_name=exp, dataset_params=dataset_params,
                                            target_file='embeddings.npy')
    embedding_actions = AnalyzeModel.load_analysis(experiment_name=exp, dataset_params=dataset_params,
                                                   target_file='embeddings_actions.npy')

    cluster_indices_dict, labels = AnalyzeModel.hdbscan_clustering(embeddings, min_cluster_size=30)

    fig, ax = plt.subplots(1, 1)
    nodeG = AnalyzeModel.generate_node_graph(cluster_indices_dict)
    edgeG = AnalyzeModel.generate_topology(embedding_actions)
    visualize_embedding_space_with_topology(embeddings, labels, nodeG, edgeG, ax,
                                            {'color_palette': 'bright',
                                             'inset': False,
                                             'label_every': 1,
                                             'zoom_in': 4,
                                             })
    # fig.savefig(folder_path + 'embedding_unsupervised_zoom.pdf')
    fig.savefig(folder_path + 'embedding_unsupervised_zoom.png', dpi=300)
    plt.close(fig)


def generate_supplemental_figure_histogram(exp, dataset_params):
    folder_path = FOLDER_PATH + f'/{exp}/'
    os.makedirs(folder_path, exist_ok=True)

    embeddings = AnalyzeModel.load_analysis(experiment_name=exp, dataset_params=dataset_params,
                                            target_file='embeddings.npy')
    embedding_actions = AnalyzeModel.load_analysis(experiment_name=exp, dataset_params=dataset_params,
                                                   target_file='embeddings_actions.npy')
    embedding_counts = AnalyzeModel.load_analysis(experiment_name=exp, dataset_params=dataset_params,
                                                  target_file='embeddings_counts.npy')

    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    training_limit = 3
    generate_histogram(embedding_actions, embeddings, ax,
                       plot_params={'training_limit': training_limit,
                                    'embedding_counts': embedding_counts,
                                    })

    fig.savefig(folder_path + 'embedding_histogram.pdf')


def generate_supplemental_figure_extra_embedding_space_visualizations(experiments, datasets):
    annotations = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    for i, exp in enumerate(experiments):
        path = FOLDER_PATH + f'/{exp}/'
        os.makedirs(path, exist_ok=True)
        # Load Datasets
        d = datasets[i]
        embeddings = AnalyzeModel.load_analysis(experiment_name=exp, dataset_params=d, target_file='embeddings.npy')
        embedding_counts = AnalyzeModel.load_analysis(experiment_name=exp, dataset_params=d,
                                                      target_file='embeddings_counts.npy')

        # Embeddings
        # Ground Truth
        fig, ax = plt.subplots(1, 1)
        visualize_embedding_space_with_labels(embeddings, embedding_counts, ax, {'color_palette': 'crest',
                                                                                 'inset': True,
                                                                                 # 'inset_axes': [0.05, 0.52, 0.4, 0.4],
                                                                                 'inset_axes': [0.55, 0.55, 0.4, 0.4],
                                                                                 # 'annotate': annotations[i],
                                                                                 })
        fig.savefig(path + exp + '.png', dpi=300)
        plt.close()


def vary_min_cluster_size_plots(exp, dataset_params):
    os.makedirs(FOLDER_PATH + '/min_cluster_size/' + exp, exist_ok=True)

    embeddings = AnalyzeModel.load_analysis(experiment_name=exp, dataset_params=dataset_params,
                                            target_file='embeddings.npy')

    min_cluster_sizes = list(range(5, 110, 5))
    for mcs in min_cluster_sizes:
        cluster_indices_dict1, labels = AnalyzeModel.hdbscan_clustering(embeddings, min_cluster_size=mcs)

        # Clustering
        fig, ax = plt.subplots(1, 1)
        visualize_embedding_space_with_labels(embeddings, labels, ax, {'color_palette': 'bright',
                                                                       'label_every': 1,
                                                                       })
        fig.savefig(FOLDER_PATH + '/min_cluster_size/' + exp + f'/{mcs}')
        plt.close(fig)


def vary_embedding_dimension_check_linearity(experiments):
    d2 = {'n_obj': 30, 'dsize': 150, 'dataset_name': 'random_contrast_random_sizes_test_set', 'set_num': 1}

    dimension_linearity_dict = {}
    for exp in experiments:
        embeddings = AnalyzeModel.load_analysis(experiment_name=exp, dataset_params=d2, target_file='embeddings.npy')
        embedding_counts = AnalyzeModel.load_analysis(experiment_name=exp, dataset_params=d2,
                                                      target_file='embeddings_counts.npy')
        r2_list = AnalyzeModel.linear_regression_embedding_dimension_fits(embeddings)
        dimension_linearity_dict[embeddings.shape[1]] = (min(r2_list), r2_list)

        # pca = PCA(n_components=2)
        # embeddings2d = pca.fit_transform(embeddings)
        # PCA(n_components=2)
        #
        # fig, ax = plt.subplots(1, 1)
        # visualize_embedding_space_with_labels(embeddings2d, embedding_counts, ax, {'color_palette': 'crest',
        #                                                                            #'inset': True,
        #                                                                            #'inset_axes': [0.05, 0.55, 0.4, 0.4],
        #                                                                            })
        # os.makedirs(AUX_FIGURE_OUTPUT_FOLDER + exp, exist_ok=True)
        # fig.savefig(AUX_FIGURE_OUTPUT_FOLDER + exp + '/embeddings2d_pca.pdf')

        fig, axes = plt.subplots(3, 3)
        for i in range(1, min(8, embeddings.shape[1])):
            visualize_embedding_space_with_labels(embeddings[:, [0, i]], embedding_counts, axes.flatten()[i],
                                                  {'color_palette': 'crest'})
        os.makedirs(AUX_FIGURE_OUTPUT_FOLDER + exp, exist_ok=True)
        fig.savefig(AUX_FIGURE_OUTPUT_FOLDER + exp + '/embeddings_iterated.pdf')

    with open(AUX_FIGURE_OUTPUT_FOLDER + 'embedding_dimension_regression_fits.txt', 'w') as f:
        json.dump(dimension_linearity_dict, f, indent=1)


# INCOMPLETE
def embedding_vs_intensity_figure(exp):
    d2 = {'n_obj': 30, 'dsize': 150, 'dataset_name': 'random_contrast_random_sizes_test_set', 'set_num': 1}
    dataset_folder = AnalyzeModel.generate_dataset_folder(d2)
    data = np.load('../data/raw/' + dataset_folder + 'data_test.npy')
    intensities = np.sum(data, axis=(0, 1, 2))
    intensities /= np.max(intensities)
    embeddings = AnalyzeModel.load_analysis(experiment_name=exp, dataset_params=d2, target_file='embeddings.npy')
    fig, ax = plt.subplots(1, 1)
    visualize_embedding_space_vs_intensity(embeddings, intensities, ax)


exp1 = 'model1_0'
exp2 = 'model2_0'
exp3 = 'model2_0_noise=False_cj=5_sj=3'
exp4 = 'model2_0_num_obj=3_action_size=3'

d1 = {'n_obj': 30, 'dsize': 150, 'dataset_name': 'high_contrast_same_size_test_set', 'set_num': 1}
d2 = {'n_obj': 30, 'dsize': 150, 'dataset_name': 'random_contrast_random_sizes_test_set', 'set_num': 1}
d3 = {'n_obj': 30, 'dsize': 150, 'dataset_name': 'jittered_contrast_jittered_sizes_test_set', 'set_num': 1,
      'noise_opt': False}

# generate_main_figure(exp1, dataset_params=d1,
#                      inset_axes=[0.55, 0.1, 0.4, 0.4])
generate_main_figure(exp2, dataset_params=d2,
                     inset_axes=[0.55, 0.55, 0.4, 0.4])
# generate_main_figure(exp3, dataset_params=d3,
#                      inset_axes=[0.55, 0.05, 0.4, 0.4])
# generate_main_figure(exp4, dataset_params=d2,
#                      inset_axes=[0.55, 0.55, 0.4, 0.4])
# d1_actions = {'n_obj': 8, 'dsize': 8000, 'dataset_name': 'high_contrast_same_size_test_set', 'set_num': 0}
# d2_actions = {'n_obj': 8, 'dsize': 8000, 'dataset_name': 'random_contrast_random_sizes_test_set', 'set_num': 0}
# generate_supplemental_figure_zoomed_in(exp1, d1_actions)
# generate_supplemental_figure_zoomed_in(exp2, d2_actions)
#
# generate_supplemental_figure_histogram(exp1, d1_actions)
# generate_supplemental_figure_histogram(exp2, d2_actions)
#
experiments = ['model2_5obj_0', 'model2_8obj_0']#, 'model2_1', 'model2_2']
datasets = [d2] * 4
generate_supplemental_figure_extra_embedding_space_visualizations(experiments, datasets)
# vary_min_cluster_size_plots(exp2, dataset_params=d2)
