import analysis.utilities as anut
import torch
import torchvision.transforms as transforms
from data_loaders.sequential_pair_data_loader_action_sizes import SequentialPairDataLoader
from data_loaders.single_image_data_loader_action_sizes import SingleImageLoader
import os, shutil
import hdbscan
import json
import numpy as np
import networkx as nx
from global_variables import ROOT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AnalyzeModel:

    def __init__(self, experiment_name, pair_datasets=None, single_datasets=None, model_name='final.pt'):
        print('Analyzing model ' + experiment_name + '...')
        self.experiment_name = experiment_name
        self.model = self.load_model(self.experiment_name, model_name)

        self.save_folder = f'{ROOT}/analysis_output/{experiment_name}/'

        os.makedirs(self.save_folder, exist_ok=True)
        shutil.rmtree(self.save_folder)

        if pair_datasets is not None:
            for dataset in pair_datasets:
                dataset['split_name'] = 'test.pt'

                dataloader1, dataset_folder = self.load_dataloader(dataset, 'as_pairs')
                print('Loaded ' + dataset_folder)
                dataset_folder = self.save_folder + dataset_folder
                os.makedirs(dataset_folder, exist_ok=True)

                predicted_actions, correct_per_action = self.generate_predictions(self.model, dataloader1)
                np.save(dataset_folder + 'predicted_actions.npy', predicted_actions)
                json.dump(correct_per_action, fp=open(dataset_folder + 'correct_per_action.json', 'w'), indent=1)

        if single_datasets is not None:
            for dataset in single_datasets:
                dataset['split_name'] = 'test.pt'
                dataloader, dataset_folder = self.load_dataloader(dataset, 'as_singles')
                print('Loaded ' + dataset_folder)
                dataset_folder = self.save_folder + dataset_folder
                os.makedirs(dataset_folder, exist_ok=True)

                gpu_embeddings, embeddings, counts, actions = self.generate_embeddings(self.model, dataloader)
                np.save(dataset_folder + 'embeddings.npy', embeddings)
                np.save(dataset_folder + 'embeddings_counts.npy', np.int8(counts))
                np.save(dataset_folder + 'embeddings_actions.npy', np.int8(actions))

    @staticmethod
    def load_model(experiment_name, model_name):
        return anut.load_trained_model(experiment_name, model_name)

    @staticmethod
    def load_analysis(experiment_name, dataset_params, target_file):
        save_folder = f'{ROOT}/analysis_output/{experiment_name}/'
        print(save_folder)
        dataset_folder = AnalyzeModel.generate_dataset_folder(dataset_params)
        path = save_folder + dataset_folder
        if target_file.split('.')[-1] == 'json':
            with open(path + target_file, 'r') as f:
                obj = json.load(f)
        elif target_file.split('.')[-1] == 'npy':
            obj = np.load(path + target_file)
        else:
            raise Exception('Unknown file extension: ' + target_file)

        return obj

    @staticmethod
    def generate_dataset_folder(dataset_params):
        # dataset specifiers
        n_obj = dataset_params['n_obj']
        dsize = dataset_params['dsize']
        dataset_name = dataset_params['dataset_name']
        set_num = dataset_params['set_num']

        dataset_folder = dataset_name + '/test_set{set_num}_n_obj={c}_d_size={dsize}/'.format(c=n_obj, dsize=dsize,
                                                                                              set_num=set_num)

        if 'noise_opt' in dataset_params and 'color_jitter' in dataset_params:
            noise_opt = dataset_params['noise_opt']
            color_jitter = dataset_params['color_jitter']
            size_jitter = dataset_params['size_jitter']

            dataset_folder = dataset_name + f'/test_set{set_num}_n_obj={n_obj}_d_size={dsize}_noise={noise_opt}_cj={color_jitter}_sj={size_jitter}/'
        elif 'noise_opt' in dataset_params and 'color_jitter' not in dataset_params:
            noise_opt = dataset_params['noise_opt']
            dataset_folder = dataset_name + f'/test_set{set_num}_n_obj={n_obj}_d_size={dsize}_noise={noise_opt}/'

        return dataset_folder

    @staticmethod
    def generate_auxiliary_dataset_folder(dataset_params):
        # dataset specifiers
        n_steps = dataset_params['n_steps']
        radius_step = dataset_params['radius_step']
        dataset_name = dataset_params['dataset_name']
        set_num = dataset_params['set_num']

        dataset_folder = dataset_name + f'/aux_test_set{set_num}_n_steps={n_steps}_radius_step={radius_step}'

        return dataset_folder

    @staticmethod
    def load_dataloader(params, dataloader_type):
        '''
        Automatically produces a dataloader for analysis experiments
        :param params: params for loading a test dataset; n_obj, dsize, dataset_name, set_num (see generate_dataset_folder())
        :param dataloader_type: Specify if the dataset should be treated as sequential 'as_pairs' (generated by actions) or
        'as_singles' generated by a specific count
        :return: dataloader, data_folder - the folder where the test set was loaded from
        '''
        trans = transforms.Compose([transforms.ToTensor()])

        if params.get('dataset_folder', None) is None:
            dataset_folder = AnalyzeModel.generate_dataset_folder(params)
        else:
            dataset_folder = params['dataset_folder']

        split_name = params['split_name']

        print(ROOT)
        if dataloader_type == 'as_pairs':
            dataloader = torch.utils.data.DataLoader(
                SequentialPairDataLoader(root=f'{ROOT}/data', dataset_folder=dataset_folder, split_name=split_name,
                                         transform=trans),
                batch_size=16, shuffle=False)

        elif dataloader_type == 'as_singles':
            dataloader = torch.utils.data.DataLoader(
                SingleImageLoader(f'{ROOT}/data', dataset_folder=dataset_folder, split_name=split_name, transform=trans),
                batch_size=16, shuffle=False)
        else:
            raise Exception('dataloader_type is not recognized!')

        return dataloader, dataset_folder

    @staticmethod
    def generate_embeddings(model, dataloader):
        print('Generating Embeddings...')

        embedding_model = model
        torch.manual_seed(0)
        embedding_model.eval()
        embedding_model.embedding_net.eval()
        with torch.no_grad():
            # data = dataloader.dataset.data
            # labels = dataloader.dataset.labels
            actions = dataloader.dataset.actions
            counts = dataloader.dataset.counts
            embeddings = []
            gpu_embeddings = []
            for (img, label) in dataloader:
                gpu_embedding = embedding_model.get_embedding(img[0].to(device))
                gpu_embeddings.append(gpu_embedding)
                embeddings.append(gpu_embedding.to(torch.float).cpu().numpy())

            gpu_embeddings = torch.cat(gpu_embeddings)
            embeddings = np.concatenate(embeddings)

        return gpu_embeddings, embeddings, counts, actions

    @staticmethod
    def generate_predictions(model, data_loader):
        print('Generating Predictions...')
        model.eval()

        num_correct = 0
        total = 0
        predicted_actions = []
        correct_per_cardinality_action = {}
        target_transform = {str([0, 0, 1]): '>', str([0, 1, 0]): '-', str([1, 0, 0]): '<'}

        with torch.no_grad():
            for batch_idx, (data, targets, counts) in enumerate(data_loader):
                keys = []

                # send images to device
                for k in range(len(data)):
                    data[k] = data[k].to(device)

                # get targets
                targets = targets.type(torch.LongTensor).to(device)

                # generate output
                output = model(data[0], data[1])

                # compare
                batch_predicted_actions = torch.argmax(output, dim=1)
                batch_target_actions = torch.argmax(targets, dim=1)
                correct_indices = batch_predicted_actions == batch_target_actions
                predicted_actions.append(batch_predicted_actions)

                # track the number of correct predictions
                num_correct += torch.sum(correct_indices).item()
                total += len(batch_predicted_actions)

                # number correct per action
                for k in range(len(correct_indices)):

                    # generate a key that describes the cardinality of an image and its relationship to its pair
                    # ie. 2>, 3<, 3-
                    key = str(counts[k].item()) + target_transform[str(targets[k].cpu().tolist())]
                    keys.append(key)

                    # add to dict if key exists, create dict entry otherwise
                    vals = correct_per_cardinality_action.get(key, [])
                    if len(vals) == 0:
                        vals = [int(correct_indices[k]), 1]
                        correct_per_cardinality_action[key] = vals
                    else:
                        # adds one if k is in correct_indices
                        vals[0] += int(correct_indices[k])
                        vals[1] += 1

        predicted_actions = torch.cat(predicted_actions).cpu().numpy()
        return predicted_actions, correct_per_cardinality_action

    @staticmethod
    def hdbscan_clustering(embeddings, min_cluster_size=20):
        clustering = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        clustering.fit(embeddings)

        base = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
                'R', 'S', 'T', 'U', 'V', 'Y', 'Z']
        cluster_labels_map = {}
        k = 0
        for cl in np.unique(clustering.labels_):
            if cl == -1:
                cid = '-1'
            else:
                cid = base[k % len(base)] + str(int(k / len(base)))
                k += 1
            cluster_labels_map[cl] = cid

        cluster_embedding_indices_dict = {}
        for ind, cluster_id in enumerate(clustering.labels_):
            if cluster_labels_map[cluster_id] not in cluster_embedding_indices_dict:
                cluster_embedding_indices_dict[cluster_labels_map[cluster_id]] = {ind}
            else:
                cluster_embedding_indices_dict[cluster_labels_map[cluster_id]].add(ind)

        labels = np.empty(shape=embeddings.shape[0], dtype=np.object)
        for key in cluster_embedding_indices_dict.keys():
            indices = np.array(list(cluster_embedding_indices_dict[key]))
            labels[indices] = key

        return cluster_embedding_indices_dict, labels

    @staticmethod
    def generate_node_graph(cluster_embedding_indices_dict):
        G = nx.DiGraph()
        for cluster_marker in cluster_embedding_indices_dict.keys():
            G.add_nodes_from(cluster_embedding_indices_dict[cluster_marker], cluster=cluster_marker)

        return G

    @staticmethod
    def generate_topology(actions):
        G = nx.DiGraph()

        rm1 = actions == -1
        sm1 = actions == 0
        am1 = actions == 1
        p1 = np.array(range(0, len(actions)))
        p2 = np.array(range(1, len(actions) + 1))

        G.add_edges_from(list(zip(p1[rm1], p2[rm1])), rel=-1)
        G.add_edges_from(list(zip(p1[sm1], p2[sm1])), rel=0)
        G.add_edges_from(list(zip(p1[am1], p2[am1])), rel=1)

        return G

    @staticmethod
    def generate_confusion_matrix(predicted_labels, true_labels, remove_uncertain=False, sort=True):
        unique_true_labels = np.unique(true_labels)
        unique_predicted_labels = np.unique(predicted_labels)

        ind_map = dict(zip(unique_predicted_labels, range(len(unique_predicted_labels))))
        confusion_matrix = np.zeros(shape=(len(unique_predicted_labels), len(unique_true_labels)))
        for i, un in enumerate(unique_true_labels):
            mask = (true_labels == un)
            unique_elements, counts_elements = np.unique(predicted_labels[mask], return_counts=True)
            inds = [ind_map[u] for u in unique_elements]
            confusion_matrix[inds, i] = counts_elements

        if sort:
            order = [np.max(np.where(confusion_matrix[i] > 0)[0]) for i in range(len(unique_predicted_labels))]
            sort_inds = sorted(range(len(unique_predicted_labels)), key=lambda x: order[x])
            confusion_matrix = confusion_matrix[sort_inds, :]
            unique_predicted_labels = unique_predicted_labels[sort_inds]
            ind_map = dict(zip(unique_predicted_labels, range(len(unique_predicted_labels))))

        if remove_uncertain:
            rem_ind = ind_map['-1']
            confusion_matrix = np.delete(confusion_matrix, rem_ind, 0)
            unique_predicted_labels = unique_predicted_labels[unique_predicted_labels != '-1']

        return confusion_matrix, unique_predicted_labels, unique_true_labels

    @staticmethod
    def rescale_embeddings(embeddings, counts, rescaling_limit, get_indices_dict=False):
        from sklearn.metrics.pairwise import pairwise_distances
        un_counts = np.unique(counts)

        estimate_dist_dict = {}
        for j in range(rescaling_limit):
            uc = un_counts[j]
            mask = (counts == uc)
            mask_to = (counts == (uc + 1))
            dists = pairwise_distances(embeddings[mask], embeddings[mask_to]).flatten()
            estimate_dist_dict[str(j)] = dists

        estimated_fit_scalar = np.mean(list(estimate_dist_dict.values()))
        offset = [0, 0] - np.mean(embeddings[counts == 0, :], axis=0)

        rescaled_distances_from_origin = {}
        mean_dists = []
        indices = np.arange(len(embeddings))
        indices_dict = {}
        for k in range(len(un_counts)):
            uc = un_counts[k]
            mask = (counts == uc)
            dists = pairwise_distances((embeddings[mask] + offset),
                                       np.array([0, 0]).reshape(1, -1)) / estimated_fit_scalar
            rescaled_distances_from_origin[k] = dists.flatten()
            indices_dict[k] = indices[mask]
            mean_dists.append(np.mean(dists.flatten()))

        if get_indices_dict:
            return rescaled_distances_from_origin, indices_dict
        else:
            return rescaled_distances_from_origin

    @staticmethod
    def fit_power_function(rescaled_embeddings_dict):
        from scipy.optimize import curve_fit

        def power_function(n, beta, alpha):
            return alpha * n ** beta

        def get_r2(func, xvals, yvals, popt):
            y_mean = np.mean(yvals)
            ss_tot = np.sum((yvals - y_mean) ** 2)
            ss_res = np.sum((yvals.flatten() - func(xvals.flatten(), *popt)) ** 2)
            r2 = 1 - ss_res / ss_tot
            return r2

        xvals = np.array([[key] * len(rescaled_embeddings_dict[key]) for key in rescaled_embeddings_dict.keys()])
        yvals = np.array(list(rescaled_embeddings_dict.values()))
        opt_values, pcov = curve_fit(power_function, xvals.flatten(), yvals.flatten())
        r2 = get_r2(power_function, xvals, yvals, opt_values)

        return opt_values, r2

    @staticmethod
    def linear_regression_embedding_dimension_fits(embeddings):
        from sklearn.linear_model import LinearRegression

        r2_list = []
        for i in range(embeddings.shape[1]):
            reg = LinearRegression().fit(np.delete(embeddings, i, axis=1), embeddings[:, i])
            r2_list.append(reg.score(np.delete(embeddings, i, axis=1), embeddings[:, i]))

        return r2_list
