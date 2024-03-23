import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from scipy.io import loadmat
from PIL import Image


def generate_violinplot(embedding_dict, ax, plot_params=None):
    if plot_params is None:
        plot_params = {}

    num_counts = len(embedding_dict.keys())

    ax.violinplot(embedding_dict.values(), embedding_dict.keys(), showmeans=True)
    ax.set_xticks(list(embedding_dict.keys()))
    ax.set_ylabel('Perceived numerosity', fontsize=20)
    ax.set_xlabel('Number of objects', fontsize=20)

    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.plot(range(num_counts), range(num_counts), color='red', label='y=x', zorder=-1)

    if plot_params.get('add_human_estimate', True):
        power_law_exp = 0.83  # Krueger 1982 Single Judgements
        k = 1
        x = np.array(range(0, 31))
        y = k*x**power_law_exp
        ax.plot(x[0:21], y[0:21], color='forestgreen', linestyle='--')
        ax.plot(x[20:], y[20:], color='forestgreen', label='$x^{0.83}$')
        y_low = k*x**(power_law_exp - 0.2)
        y_high = k*x**(power_law_exp + 0.2)
        ax.fill_between(x[20:], y_low[20:], y_high[20:], alpha=0.1, color='forestgreen', zorder=-3)

    if plot_params.get('add_multiple_model_fits', None) is not None:
        model_fits = plot_params.get('add_multiple_model_fits')
        for fit in model_fits:
            power_law_exp = fit[0]
            k = fit[1]
            x = np.array(range(0, 31))
            y = k*x**power_law_exp
            ax.plot(x, y, color='orange', alpha=0.3, zorder=-2)

    if plot_params.get('add_main_model_fit', None) is not None:
        model_fit = plot_params.get('add_main_model_fit')
        power_law_exp = model_fit[0]
        k = model_fit[1]
        x = np.array(range(0, 31))
        y = k * x ** power_law_exp
        pwls = '{' + str(power_law_exp)[:4] + '}'
        ax.plot(x, y, color='blue', alpha=0.2, label=f'$x^{pwls}$', zorder=-2)

    ax.grid(which='major', axis='x', alpha=0.2, color='black', linestyle='-')
    ax.grid(which='major', axis='y', alpha=0.2, color='black', linestyle='-')
    ax.set_axisbelow(True)
    ax.set_ylim((-1, num_counts+1))
    ax.set_xlim((-1, num_counts+1))
    ax.axhline(-3, color='orange', alpha=0.3, label='model fits')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=14)

    return ax


def generate_psychometric_curve(embedding_dict, left, right_list, ax):

    embedding_val_left = embedding_dict[left]
    results_dict = {}
    for right in right_list:
        embedding_val_right = embedding_dict[right]
        results_dict[(left, right)] = [0, 0]
        for evl in embedding_val_left:
            for evr in embedding_val_right:
                diff = evr - evl
                if diff < 0:
                    results_dict[(left, right)][0] += 1
                elif diff > 0:
                    results_dict[(left, right)][1] += 1

    xvals = []
    yvals = []
    for key in results_dict.keys():
        xvals.append(key[1])
        yvals.append(results_dict[key][1]/(results_dict[key][0] + results_dict[key][1]))

    ax.plot(xvals, yvals, label='model', zorder=-1, linewidth=5, alpha=0.5)
    ax.set_xlabel('Test number', fontsize=20)
    ax.set_ylabel('Proportion "more"', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    return ax


def add_human_data_burr(ax):
    shape_list = []
    prop_matrix = np.zeros(shape=(25, 10))
    for uid in range(1, 11):
        results_arr = np.zeros(shape=(25, 2))
        # format userid
        if uid < 10:
            str_uid = '0' + str(uid)
        else:
            str_uid = str(uid)

        for trial_number in range(1, 10):
            for cond in ['Control']:  # ['AdaptVis_no', 'Control']:
                burr_path = '../data/burr_data/{cond}_S{uid}_16_{trial}.mat'.format(cond=cond,
                                                                                                     uid=str_uid,
                                                                                                     trial=trial_number)

                try:
                    matfile = loadmat(burr_path)
                    shape_list.append((burr_path, matfile['MatriceRisultati'].shape))
                    results_arr[matfile['MatriceRisultati'][:, 1].astype(int) - 8, 0] += matfile['MatriceRisultati'][:, 2]
                    results_arr[matfile['MatriceRisultati'][:, 1].astype(int) - 8, 1] += 1
                except FileNotFoundError:
                    print('Skipping ' + burr_path)
        pmore = results_arr[:, 0] / results_arr[:, 1]
        prop_matrix[:, uid - 1] = pmore

    xvals = np.array(list(range(8, 33)))
    for i in range(10):
        if i == 0:
            ax.scatter(xvals, prop_matrix[:, i], color='forestgreen', alpha=0.3, label='human')
        else:
            ax.scatter(xvals, prop_matrix[:, i], color='forestgreen', alpha=0.3)

    ax.axvline(16, label='reference', linestyle='--', color='orange')
    return ax, shape_list


def add_reference_test_image_pairs(data, ax):

    inset_axes1 = [0.01, 0.73, 0.2, 0.2]
    axin1 = ax.inset_axes(inset_axes1)
    axin1.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    inset_axes2 = [0.17, 0.73, 0.2, 0.2]
    axin2 = ax.inset_axes(inset_axes2)
    axin2.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    inset_axes3 = [0.63, 0.05, 0.2, 0.2]
    axin3 = ax.inset_axes(inset_axes3)
    axin3.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    inset_axes4 = [0.79, 0.05, 0.2, 0.2]
    axin4 = ax.inset_axes(inset_axes4)
    axin4.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    im12 = data[0]
    im16 = data[1]
    im20 = data[2]
    img12 = Image.fromarray(np.copy(np.moveaxis(im12.numpy() * 256, [0, 1, 2], [2, 0, 1]).astype('uint8')), 'RGB')
    img16 = Image.fromarray(np.copy(np.moveaxis(im16.numpy() * 256, [0, 1, 2], [2, 0, 1]).astype('uint8')), 'RGB')
    img20 = Image.fromarray(np.copy(np.moveaxis(im20.numpy() * 256, [0, 1, 2], [2, 0, 1]).astype('uint8')), 'RGB')

    keys = ['bottom', 'top', 'right', 'left']
    for key in keys:
        sp = axin1.spines[key]
        sp.set_color('orange')
        sp.set_lw(2)

    # for key in keys:
    #     sp = axin2.spines[key]
    #     sp.set_color('lightblue')
    #     sp.set_lw(2)
    #
    # for key in keys:
    #     sp = axin3.spines[key]
    #     sp.set_color('lightblue')
    #     sp.set_lw(2)

    for key in keys:
        sp = axin4.spines[key]
        sp.set_color('orange')
        sp.set_lw(2)

    axin1.imshow(img16)
    axin1.set_title('16', fontsize=10)

    axin2.imshow(img20)
    axin2.set_title('20', fontsize=10)

    axin3.imshow(img12)
    axin3.set_title('12', fontsize=10)

    axin4.imshow(img16)
    axin4.set_title('16', fontsize=10)

    return ax


def generate_reference_test_image_triplet_panel(data, axes):

    # im12 = data[0]
    # im16 = data[1]
    # im20 = data[2]
    # img12 = Image.fromarray(np.copy(np.moveaxis(im12.numpy() * 256, [0, 1, 2], [2, 0, 1]).astype('uint8')), 'RGB')
    # img16 = Image.fromarray(np.copy(np.moveaxis(im16.numpy() * 256, [0, 1, 2], [2, 0, 1]).astype('uint8')), 'RGB')
    # img20 = Image.fromarray(np.copy(np.moveaxis(im20.numpy() * 256, [0, 1, 2], [2, 0, 1]).astype('uint8')), 'RGB')

    img12 = Image.open(data[0])
    img16 = Image.open(data[1])
    img20 = Image.open(data[2])
    images = [img12, img16, img20]
    titles = [12, 16, 20]
    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.set_title(titles[i], fontsize=12)

    keys = ['bottom', 'top', 'right', 'left']
    for key in keys:
        sp = axes[1].spines[key]
        sp.set_color('orange')
        sp.set_lw(2)

