import os

from analysis.analyze_model import AnalyzeModel
import numpy as np
import matplotlib.pyplot as plt
from plotting.embedding_space_plots import *
from scipy.linalg import svd

FOLDER_PATH = '../figures/reproducibility/'
os.makedirs(FOLDER_PATH, exist_ok=True)
os.makedirs(FOLDER_PATH + '/confusion_matrix/', exist_ok=True)
eids = list(range(30))


def plot(ax, embeddings, embedding_counts):
    x_mean = np.mean(embeddings, axis=0).reshape(1, -1)
    X = embeddings - x_mean

    (U, S, Vh) = svd(X, full_matrices=True)
    S = np.diag(S)
    D = x_mean + U[:, 0, None] @ (S[0, 0] * Vh[0, :, None].T)
    score = np.linalg.norm(embeddings - D, ord='fro') / np.linalg.norm(embeddings, ord='fro')

    visualize_embedding_space_with_labels(embeddings[mask, :], embedding_counts[mask],
                                          ax, {'color_palette': 'crest',
                                               'legend_visible': False,
                                               'apply_tight_layout': False,
                                               })
    ax.set_title('Error: %1.1f%%' % (score * 100), fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')


def plot2(ax, embeddings, embedding_counts):
    visualize_embedding_space_with_labels(embeddings[mask, :], embedding_counts[mask],
                                          ax, {'color_palette': 'bright',
                                               'legend_visible': False,
                                               'apply_tight_layout': False,
                                               })
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')


def plot3(ax, ax2, embedding, embedding_counts, min_cluster_size=30):
    cluster_indices_dict, labels = AnalyzeModel.hdbscan_clustering(embedding, min_cluster_size=min_cluster_size)
    visualize_embedding_space_with_labels(embedding, labels, ax, {'color_palette': 'bright',
                                                                  'inset': False,
                                                                  'label_every': 1, 'label_limit': 10,
                                                                  })

    conf_matrix, column_names, row_names = AnalyzeModel.generate_confusion_matrix(labels, embedding_counts,
                                                                                  remove_uncertain=True, sort=True)

    visualize_confusion_matrix(conf_matrix, ax2, {'column_names': column_names,
                                                  'row_names': row_names,
                                                  # 'colbar_shrink': 0.8,
                                                  # 'annotate': 'C',
                                                  # 'annotate_pos': (0.028, 0.9),
                                                  'transpose': True,
                                                  'cmap': 'Blues',
                                                  })


mask = np.array(([1] + [0] * 14) * 310, dtype=np.bool)  # every 10th point
for i, (cj, sj) in enumerate([(5, 3)]):
    _fig, _axes = plt.subplots(5, 6)
    _fig2, _axes2 = plt.subplots(5, 6)
    _fig3, _axes3 = plt.subplots(5, 6)
    # _fig4, _axes4 = plt.subplots(5, 6)
    for c, eid in enumerate(eids):
        d2 = {'n_obj': 30, 'dsize': 150, 'dataset_name': 'jittered_contrast_jittered_sizes_test_set',
              'set_num': 1, 'noise_opt': False}
        seed = eid
        exp = f'model2_{seed}_noise={False}_cj={cj}_sj={sj}'
        _embeddings = AnalyzeModel.load_analysis(experiment_name=exp, dataset_params=d2, target_file='embeddings.npy')
        _embedding_counts = AnalyzeModel.load_analysis(experiment_name=exp, dataset_params=d2,
                                                       target_file='embeddings_counts.npy')
        _ax = _axes.flatten()[c]
        _ax2 = _axes2.flatten()[c]
        _ax3 = _axes3.flatten()[c]
        plot(_ax, _embeddings, _embedding_counts)
        plot2(_ax2, _embeddings, _embedding_counts)

        _fig4, _axes4 = plt.subplots(1, 1)
        _fig4.set_size_inches(15, 4)
        plot3(_ax3, _axes4, _embeddings, _embedding_counts)
        _fig4.savefig(FOLDER_PATH + f'/confusion_matrix/model2_jitter_cm_{eid}.pdf')
        plt.close(_fig4.number)

    _fig.set_size_inches(14, 8)
    _fig2.set_size_inches(14, 8)
    _fig3.set_size_inches(14, 8)
    _fig.savefig(FOLDER_PATH + f'model2_jitter.pdf')
    _fig2.savefig(FOLDER_PATH + f'model2_jitter_gt.pdf')
    _fig3.savefig(FOLDER_PATH + f'model2_jitter_unsup_clust.pdf')
    # _fig4.savefig(FOLDER_PATH + f'model2_jitter_cm.png')

mask = np.array(([1] + [0] * 14) * 310, dtype=np.bool)  # every 10th point
_fig, _axes = plt.subplots(2, 4)
_fig2, _axes2 = plt.subplots(2, 4)
_fig3, _axes3 = plt.subplots(2, 4)
for c, eid in enumerate(range(8)):
    d2 = {'n_obj': 30, 'dsize': 150, 'dataset_name': 'random_contrast_random_sizes_test_set',
          'set_num': 1}
    seed = eid
    exp = f'model2_{seed}_num_obj=3_action_size=3'
    _embeddings = AnalyzeModel.load_analysis(experiment_name=exp, dataset_params=d2, target_file='embeddings.npy')
    _embedding_counts = AnalyzeModel.load_analysis(experiment_name=exp, dataset_params=d2,
                                                   target_file='embeddings_counts.npy')
    _ax = _axes.flatten()[c]
    _ax2 = _axes2.flatten()[c]
    _ax3 = _axes3.flatten()[c]
    plot(_ax, _embeddings, _embedding_counts)
    plot2(_ax2, _embeddings, _embedding_counts)

    _fig4, _axes4 = plt.subplots(1, 1)
    _fig4.set_size_inches(15, 4)
    plot3(_ax3, _axes4, _embeddings, _embedding_counts)
    _fig4.savefig(FOLDER_PATH + f'/confusion_matrix/model2_action_size_cm_{eid}.pdf')
    plt.close(_fig4.number)

_fig.set_size_inches(14, 8)
_fig2.set_size_inches(14, 8)
_fig3.set_size_inches(14, 8)
_fig.savefig(FOLDER_PATH + f'model2_action_size.pdf')
_fig2.savefig(FOLDER_PATH + f'model2_action_size_gt.pdf')
_fig3.savefig(FOLDER_PATH + f'model2_action_size_unsup_clust.pdf')

model_choice = ['model1', 'model2']
dset_choice = ['high_contrast_same_size_test_set', 'random_contrast_random_sizes_test_set']
for i, model_name in enumerate(model_choice):
    _fig, _axes = plt.subplots(5, 6)
    _fig2, _axes2 = plt.subplots(5, 6)
    _fig3, _axes3 = plt.subplots(5, 6)
    for c, eid in enumerate(eids):
        d2 = {'n_obj': 30, 'dsize': 150, 'dataset_name': dset_choice[i],
              'set_num': 1}
        seed = eid
        exp = f'{model_name}_{seed}'
        _embeddings = AnalyzeModel.load_analysis(experiment_name=exp, dataset_params=d2, target_file='embeddings.npy')
        _embedding_counts = AnalyzeModel.load_analysis(experiment_name=exp, dataset_params=d2,
                                                       target_file='embeddings_counts.npy')
        _ax = _axes.flatten()[c]
        _ax2 = _axes2.flatten()[c]
        _ax3 = _axes3.flatten()[c]
        plot(_ax, _embeddings, _embedding_counts)
        plot2(_ax2, _embeddings, _embedding_counts)

        _fig4, _axes4 = plt.subplots(1, 1)
        plot3(_ax3, _axes4, _embeddings, _embedding_counts)
        _fig4.savefig(FOLDER_PATH + f'/confusion_matrix/{model_name}_cm_{eid}.pdf')
        plt.close(_fig4.number)

    _fig.set_size_inches(14, 8)
    _fig2.set_size_inches(14, 8)
    _fig3.set_size_inches(14, 8)
    _fig.savefig(FOLDER_PATH + f'{model_name}.pdf')
    _fig2.savefig(FOLDER_PATH + f'{model_name}_gt.pdf')
    _fig3.savefig(FOLDER_PATH + f'{model_name}_unsup_clust.pdf')
