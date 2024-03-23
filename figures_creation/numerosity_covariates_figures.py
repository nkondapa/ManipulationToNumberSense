from analysis.analyze_model import AnalyzeModel
import numpy as np
from skimage.morphology import convex_hull_image
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.ticker import MaxNLocator
from analysis.compute_confidence_intervals import generate_ci_over_list, propci_wilson_cc, \
    get_confidence_intervals_bayesian
from PIL import Image
import os

def calculate_covariate(data, covariate):
    data = np.stack([np.asarray(Image.open(d)) for d in data]).transpose(1, 2, 3, 0)
    metric = None
    if covariate == 'intensity':
        intensities = np.sum(data, axis=(0, 1, 2))
        metric = intensities
    elif covariate == 'convex_hull':
        data_bool = data[:, :, 0, :].astype(np.bool)
        density_ratios = np.zeros(shape=data_bool.shape[-1])
        for i in range(data_bool.shape[-1]):
            density_ratios[i] = np.mean(convex_hull_image(data_bool[:, :, i]))
        metric = density_ratios
    elif covariate == 'total_area':
        total_area = np.sum(data > 0, axis=(0, 1, 2))
        metric = total_area

    metric = metric.astype(np.float64)
    return metric


def generate_covariate_figure(experiment, dataset_params, target, references, axes):

    assert len(references) == len(axes.flatten())
    folder = AnalyzeModel.generate_dataset_folder(dataset_params)
    embedding_counts = AnalyzeModel.load_analysis(experiment_name=experiment, dataset_params=dataset_params,
                                                  target_file='embeddings_counts.npy')
    embeddings = AnalyzeModel.load_analysis(experiment_name=experiment, dataset_params=dataset_params,
                                            target_file='embeddings.npy')
    rescaled_embeddings, indices_dict = AnalyzeModel.rescale_embeddings(embeddings, embedding_counts, 1,
                                                                        get_indices_dict=True)
    r_embeddings = np.array(list(rescaled_embeddings.values())).flatten()
    r_indices = np.array(list(indices_dict.values())).flatten()
    sorted_rescaled_embeddings = r_embeddings[np.argsort(r_indices)]

    data = np.load('../data/raw/' + folder + '/data_{p}.npy'.format(p='train'))
    metric = calculate_covariate(data, target)

    dfunc = lambda x, y: (x - y)/y
    dfunc2 = lambda x, y: np.sign(x - y)
    myfunc = lambda x: np.array(x).astype(np.float64)

    target_range = 2
    for refi, ref in enumerate(references):
        ax = axes.flatten()[refi]
        print(ref)
        unc_inds = np.where(embedding_counts == ref)[0]

        r = range(ref - target_range, ref + target_range + 1)
        for unc2 in r:
            if unc2 == ref:
                continue
            unc2_inds = np.where(embedding_counts == unc2)[0]
            # intensities are compared to reference, so if unc2 < unc should be neg
            metric_mat = pairwise_distances(metric[unc2_inds, None], metric[unc_inds, None], metric=dfunc)
            emb_mat = pairwise_distances(sorted_rescaled_embeddings[unc_inds, None],
                                         sorted_rescaled_embeddings[unc2_inds, None], metric=dfunc2)
            mm_min = np.min(metric_mat)
            mm_max = np.max(metric_mat)
            if mm_min < 0 and mm_max > 0:
                bins = [mm_min, mm_min/2, 0, mm_max/2, mm_max]
                statistic, bin_edges, binnumber = stats.binned_statistic(metric_mat.flatten(), metric_mat.flatten(), statistic='median', bins=bins)
            else:
                statistic, bin_edges, binnumber = stats.binned_statistic(metric_mat.flatten(), metric_mat.flatten(), statistic='median', bins=4)

            numer_list = []
            total_list = []
            num_samples_mask = []
            for bn in np.unique(binnumber):
                mask = (binnumber == bn)

                t = 1
                if unc2 > ref:
                    t = -1

                pos = (emb_mat.flatten()[mask] == t).sum()
                neg = (emb_mat.flatten()[mask] == -1 * t).sum()

                # print(pos, neg)
                numer_list.append(pos)
                total_list.append(pos + neg)
                num_samples_mask.append((pos + neg) > 100)

            low, mid, high = map(myfunc, generate_ci_over_list(propci_wilson_cc, numer_list, total_list))
            ls = '-'
            if unc2 == ref:
                ls = '--'
            ax.plot(statistic[num_samples_mask], mid[num_samples_mask], label=f'{unc2}', linestyle=ls, marker='o')
            ax.fill_between(statistic[num_samples_mask], low[num_samples_mask], high[num_samples_mask], alpha=0.3)
            ax.set_ylim([0.55, 1.01])

            # plt.yscale('log', basey=2)
            print(min(total_list), total_list)
            print()
        ax.legend(loc='lower left')

    return metric, embedding_counts, data, sorted_rescaled_embeddings


def generate_example_figure(metric, embedding_counts, data, sorted_rescaled_embeddings, axes, low=14, ref=16, high=18):

    mask_ref = (embedding_counts == ref)
    mask_low = (embedding_counts == low)
    mask_high = (embedding_counts == high)

    ind_high = np.argmin(metric[mask_high])
    ind_low = np.argmax(metric[mask_low])
    ind_ref = np.argmin(np.abs(metric[mask_ref] - np.median(metric[mask_ref])))

    val_high = metric[mask_high][ind_high]
    val_ref = metric[mask_ref][ind_ref]
    val_low = metric[mask_low][ind_low]

    delta_high = (val_high - val_ref)/val_ref
    delta_low = (val_low - val_ref)/val_ref

    im_low = Image.open(data[mask_low][ind_low])
    im_ref = Image.open(data[mask_ref][ind_ref])
    im_high = Image.open(data[mask_high][ind_high])

    images = [im_low, im_ref, im_high]
    xlabels = [delta_low, 0, delta_high]
    ylabels = [low, ref, high]

    print(sorted_rescaled_embeddings[mask_low][ind_low], sorted_rescaled_embeddings[mask_ref][ind_ref], sorted_rescaled_embeddings[mask_high][ind_high])

    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        # ax.set_title(titles[i], fontsize=12)
        # ax.set_xticks([100])
        # ax.set_xticklabels([f'{xlabels[i]:0.3f}'])
        ax.set_xlabel(f'{xlabels[i]:0.3f}', fontsize=16)
        # ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_ylabel(f'Reference : {ylabels[i]}', fontsize=18)

    # keys = ['bottom', 'top', 'right', 'left']
    # for key in keys:
    #     sp = axes[1].spines[key]21
    #     sp.set_color('orange')
    #     sp.set_lw(2)


references = [3, 9, 16, 24]
targets = ['intensity', 'total_area', 'convex_hull']
labels = ['Intensity', 'Summed\nObject Areas', 'Convex Hull']


fig, axes = plt.subplots(4, 3, constrained_layout=True, squeeze=False)
for ri, ref in enumerate(references):
    axes[ri][0].set_title(f'Reference : {ref}', fontsize=18)

axes[-1][0].set_ylabel(f'Accuracy', fontsize=18)

fig2, axes2 = plt.subplots(3, 3)
low = 14
ref = 16
high = 18
ylabels = [low, ref, high]
for i in range(3):
    axes2[0][i].set_title(f'Numerosity : {ylabels[i]}', fontsize=18)
    axes2[i][0].set_ylabel(f'{labels[i]}', fontsize=18)


for i, target in enumerate(targets):
    experiment = 'model2_0'
    dataset_params = {'n_obj': 30, 'dsize': 150, 'dataset_name': 'random_contrast_random_sizes_test_set', 'set_num': 1}
    metric, embedding_counts, data, sorted_rescaled_embeddings = generate_covariate_figure(experiment, dataset_params, target, references, axes[:, i])
    generate_example_figure(metric, embedding_counts, data, sorted_rescaled_embeddings, axes2[i, :])

    # axes[-1][i].set_xlabel(f'Median Binned {labels[i]}', fontsize=18)
    axes[-1][i].set_xlabel(f'Median Binned $\Delta$ {labels[i]}', fontsize=18)

for i, ax in enumerate(axes.flatten()):
    plt.setp(ax.get_xticklabels(), fontsize=16)
    plt.setp(ax.get_yticklabels(), fontsize=16)
    ax.xaxis.set_major_locator(MaxNLocator(5))

    if i % 3 != 0:
        ax.set_yticklabels([])

fig.set_size_inches(6*3, 2.8*4)
fig2.set_size_inches(9, 9)
path = '../figures/covariate_plots/'
os.makedirs(path, exist_ok=True)
plt.figure(fig.number)
plt.savefig(f'{path}/covariates_figure.pdf')
plt.figure(fig2.number)
plt.savefig(f'{path}/covariates_examples_figure.pdf')
