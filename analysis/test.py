from torchvision import transforms
from data_loaders.online_sequential_pair_data_loader_old import OnlineSequentialPairDataLoader


# dataloader_params2 = {'num_batches': 1, 'batch_size': 180, 'experiment_name': 'temp',
#                       'diameter_ranges': [(10, 30)],
#                       'color_ranges': [(25, 255)],
#                       'shape_types_options': [['square']],
#                       'num_objects': 3,
#                       'seed': 0,
#                       'epoch': 1,
#                       }
#
# dl = OnlineSequentialPairDataLoader(**dataloader_params2, transforms=transforms.Compose([transforms.ToTensor()]))


from plotting.embedding_space_plots import *
from analysis.analyze_model import AnalyzeModel
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import json

FIGURE_OUTPUT_FOLDER = '../figures/'
AUX_FIGURE_OUTPUT_FOLDER = '../aux_figures/embedding_plots/'
FOLDER_PATH = FIGURE_OUTPUT_FOLDER + '/embedding_plots/'
os.makedirs(FOLDER_PATH, exist_ok=True)


def generate_main_figure(exp1):
    # Load Datasets
    d1 = {'n_obj': 30, 'dsize': 150, 'dataset_name': 'random_contrast_random_sizes_test_set', 'set_num': 1}

    embeddings1 = AnalyzeModel.load_analysis(experiment_name=exp1, dataset_params=d1, target_file='embeddings.npy')
    embedding_counts1 = AnalyzeModel.load_analysis(experiment_name=exp1, dataset_params=d1,
                                                   target_file='embeddings_counts.npy')
    cluster_indices_dict1, labels1 = AnalyzeModel.hdbscan_clustering(embeddings1, min_cluster_size=30)

    # Embeddings 1
    # Ground Truth
    fig, ax = plt.subplots(1, 1)
    visualize_embedding_space_with_labels(embeddings1, embedding_counts1, ax, {'color_palette': 'crest',
                                                                               'inset': True,
                                                                               'inset_axes': [0.05, 0.55, 0.4, 0.4],
                                                                               })

    # Clustering
    fig, ax = plt.subplots(1, 1)
    visualize_embedding_space_with_labels(embeddings1, labels1, ax, {'color_palette': 'bright',
                                                                     'inset': True,
                                                                     'label_every': 1,
                                                                     'inset_axes': [0.05, 0.55, 0.4, 0.4],
                                                                     })

    # Confusion Matrix
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    fig.set_size_inches(13, 5)
    conf_matrix, column_names, row_names = AnalyzeModel.generate_confusion_matrix(labels1, embedding_counts1,
                                                                                  remove_uncertain=True, sort=True)
    visualize_confusion_matrix(conf_matrix, ax, {'column_names': column_names,
                                                 'row_names': row_names,
                                                 'transpose': True})

    plt.show()

generate_main_figure('model_dataset2-3_dr_test')