import matplotlib.pyplot as plt
from data_loaders.online_sequential_pair_data_loader import OnlineSequentialPairDataLoader
from torchvision import transforms
from matplotlib.patches import ConnectionPatch, ArrowStyle
import glob
from PIL import Image
import numpy as np


def generate_dataset_sample(dataloader_params, invert_colors=False):
    dataloader_params['dataset_save_name'] = 'train_sample'
    dataloader_params['epoch'] = 0

    OnlineSequentialPairDataLoader(**dataloader_params, transforms=transforms.Compose([transforms.ToTensor()]))
    data_folder = f'../data/raw/train_sample/{dataloader_params["experiment_name"]}_object_set_dsize={dataloader_params["batch_size"]}'
    data = np.load(data_folder + '/data_{p}.npy'.format(p='train'))
    counts = np.load(data_folder + '/counts_{p}.npy'.format(p='train'))
    actions = np.load(data_folder + '/actions_{p}.npy'.format(p='train'))

    action_dict = {-1: 'T', 0: 'S', 1: 'P'}
    colors_dict = {-1: 'blue', 0: 'black', 1: 'red'}

    valid_sequence = False
    sequence_start = 0
    sequence_length = 5

    fig, axes = plt.subplots(1, sequence_length)
    fig.set_size_inches(15, 2)
    plt.subplots_adjust(left=0.01, right=0.99, wspace=0.8, top=0.99, bottom=0.01)

    while not valid_sequence:
        for i in range(sequence_start, sequence_start + sequence_length):
            mc_bool = np.any(counts[sequence_start:sequence_start + sequence_length] == max(counts))
            p_bool = np.any(actions[sequence_start:sequence_start + sequence_length - 1] == 1)
            t_bool = np.any(actions[sequence_start:sequence_start + sequence_length - 1] == -1)
            s_bool = np.any(actions[sequence_start:sequence_start + sequence_length - 1] == 0)

            if mc_bool and p_bool and t_bool and s_bool:
                valid_sequence = True

                c = i - sequence_start + 1
                ax = axes.flatten()[c - 1]

                # if len(data.shape) == 3:
                #     ax.imshow(data[:, :, i], cmap='gray')
                # elif len(data.shape) == 4:
                #     ax.imshow(data[:, :, :, i], cmap='gray')
                # else:
                #     raise Exception('Unexpected data format!')
                if invert_colors:
                    ax.imshow(255 - np.asarray(Image.open(data[i])), cmap='gray')
                else:
                    ax.imshow(Image.open(data[i]), cmap='gray')

                # draw arrows, excluding last image in sequence
                if c != sequence_length:
                    p1 = axes[c - 1].get_position().corners()
                    p2 = axes[c].get_position().corners()
                    ws = 0.01
                    xy1 = [p1[2][0] + ws, (p1[2][1] + p1[3][1]) / 2]
                    xy2 = [p2[1][0] - ws, (p2[2][1] + p2[3][1]) / 2]

                    csys = 'figure fraction'
                    arrsty = ArrowStyle('simple', head_length=1, head_width=1, tail_width=0.4)
                    con = ConnectionPatch(xy1, xy2, coordsA=csys, coordsB=csys, color=colors_dict[actions[i]],
                                          arrowstyle=arrsty)
                    fig.add_artist(con)
                    plt.annotate(action_dict[actions[i]], xy=((xy1[0] + xy2[0]) / 2, xy1[1] + 0.03),
                                 xycoords='figure fraction',
                                 ha='center', fontsize=18)

                if invert_colors:
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    ax.axis('off')

        sequence_start += 1
        if sequence_start >= len(counts):
            print('Valid sequence not found!')
            break

    return fig


def generate_train_set_statistics(experiment_name, ax):
    data_folder = f'../data/training/{experiment_name}/'
    folders = glob.glob(data_folder + '*')

    counts_list = []
    intensities_list = []
    for folder in folders:
        counts = np.load(folder + '/counts.npy')
        intensities = np.load(folder + '/intensities.npy')
        counts_list.append(counts)
        intensities_list.append(intensities)

    counts = np.stack(counts_list)
    intensities = np.stack(intensities_list)
    intensities_per_count = []
    count_keys = []
    for un in sorted(np.unique(counts)):
        mask = counts == un
        intensities_per_count.append(intensities[mask])
        count_keys.append(int(un))

    ax.violinplot(intensities_per_count, count_keys)
    ax.set_xticks(count_keys)
    return ax