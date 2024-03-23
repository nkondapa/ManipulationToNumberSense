import numpy as np
import seaborn as sns


def generate_histogram(actions, counts, embeddings, training_limit, ax):

    put_distances = []
    shake_distances = []
    take_distances = []

    rm1 = actions == -1 & (counts <= training_limit)
    sm1 = actions == 0 & (counts <= training_limit)
    am1 = actions == 1 & (counts <= (training_limit - 1))
    p1 = np.array(range(0, len(actions)))
    p2 = np.array(range(1, len(actions) + 1))
    take_distances.append(np.linalg.norm(embeddings[p1[rm1]] - embeddings[p2[rm1]], axis=1).flatten())
    shake_distances.append(np.linalg.norm(embeddings[p1[sm1]] - embeddings[p2[sm1]], axis=1).flatten())
    put_distances.append(np.linalg.norm(embeddings[p1[am1]] - embeddings[p2[am1]], axis=1).flatten())

    take_dists = np.concatenate(take_distances)
    shake_dists = np.concatenate(shake_distances)
    put_dists = np.concatenate(put_distances)

    # exclude zeros
    exclude_zeros = True
    if exclude_zeros:
        take_dists = take_dists[take_dists != 0]
        shake_dists = shake_dists[shake_dists != 0]
        put_dists = put_dists[put_dists != 0]

        take_dists_log = np.log10(take_dists)
        shake_dists_log = np.log10(shake_dists)
        put_dists_log = np.log10(put_dists)
        print(np.min(take_dists), np.min(shake_dists), np.min(put_dists))
        print(np.min(take_dists_log), np.min(shake_dists_log), np.min(put_dists_log))

    else:
        take_dists_log = np.log(take_dists + 1)
        shake_dists_log = np.log(shake_dists + 1)
        put_dists_log = np.log(put_dists + 1)

    # HISTOGRAM LINEPLOT LOG #
    sns.set_color_codes()

    nbins = np.arange(-1, 2.2, 0.2)

    take_hist, take_bins = np.histogram(take_dists_log, bins=nbins)
    take_cent_bins = take_bins[:-1] + np.diff(take_bins) / 2

    put_hist, put_bins = np.histogram(put_dists_log, bins=nbins)
    put_cent_bins = put_bins[:-1] + np.diff(put_bins) / 2

    shake_hist, shake_bins = np.histogram(shake_dists_log, bins=nbins)
    shake_cent_bins = shake_bins[:-1] + np.diff(shake_bins) / 2

    ax.plot(take_cent_bins, take_hist, label='Take', color='blue', linewidth=5, alpha=0.5)
    ax.plot(put_cent_bins, put_hist, label='Put', color='red', linewidth=5, alpha=0.5)
    ax.plot(shake_cent_bins, shake_hist, label='Shake', color='black', linewidth=5, alpha=0.6)

    ax.set_xlabel('Euclidean distance', fontsize=16)
    ax.set_ylabel('Count', fontsize=16)
    ax.set_xticks(list(range(-1, 3, 1)))
    xticks = ax.get_xticks()
    xticks_str = ['$10^{' + str(int(xt)) + '}$' for xt in xticks]
    ax.set_xticklabels(xticks_str, fontsize=14)
    ax.set_yticklabels(ax.get_yticks(), fontsize=14)
    ax.legend(loc='center left')

    return ax