import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx


def visualize_embedding_space_with_labels(embeddings, labels, ax, plot_params=None):
    if plot_params is None:
        plot_params = {}

    assert (embeddings.shape[1] == 2)

    col_pal = plot_params.get('color_palette', 'crest')
    my_cmap = None
    if col_pal == 'crest':
        my_cmap = sns.color_palette('crest', as_cmap=True)
    elif col_pal == 'bright':
        cp = sns.color_palette('bright').as_hex()
        my_cmap = ListedColormap(cp)

    unique_labels = np.unique(labels)
    print(unique_labels)
    inset = plot_params.get('inset', False)
    label_every = plot_params.get('label_every', 5)
    label_limit = plot_params.get('label_limit', 30)

    axins = None
    inset_limit = None
    if inset:
        inset_axes = plot_params.get('inset_axes', [0.55, 0.55, 0.4, 0.4])
        inset_limit = plot_params.get('inset_limit', 5)
        axins = ax.inset_axes(inset_axes)

    early_stop = plot_params.get('early_stop', None)
    inset_count = 0
    for i, ul in enumerate(unique_labels):
        if early_stop is not None and i >= early_stop:
            break
        mask = (labels == ul)

        if col_pal == 'crest':
            cmap_val = i / max(unique_labels)
        else:
            cmap_val = i % my_cmap.N

        if type(ul) is str:
            label = ul
        else:
            label = int(ul)

        if ul == -1 or ul == '-1':
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1], color='black', marker='x', alpha=1, s=5, zorder=5,
                       label='X')
            continue

        if i % label_every != 0 or i > label_limit:
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1], color=my_cmap(cmap_val), alpha=1, s=5, zorder=10)
        else:
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1], color=my_cmap(cmap_val), alpha=1, s=5, zorder=10,
                       label=label)
        if inset and inset_count < inset_limit:
            print(ul)
            if plot_params.get('inset_labels', None) is None or ul in plot_params.get('inset_labels'):
                inset_count += 1
                axins.scatter(embeddings[mask, 0], embeddings[mask, 1], color=my_cmap(cmap_val), alpha=1, s=5, zorder=15)
    if inset:
        print('INSET')
        if labels.dtype == np.int8:
            mask = labels == -1
        else:
            mask = labels == '-1'

        sub_mask = (axins.axis()[0] <= embeddings[mask, 0]) & (embeddings[mask, 0] <= axins.axis()[2]) & \
                   (axins.axis()[1] <= embeddings[mask, 1]) & (embeddings[mask, 1] <= axins.axis()[3])

        mask[mask] = sub_mask
        # axins.scatter(embeddings[mask, 0], embeddings[mask, 1], color='black', marker='x', alpha=1, s=5, zorder=5,
        #               label='X')

        axins.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.indicate_inset_zoom(axins, zorder=5, edgecolor='black', alpha=0.8)

    if plot_params.get('annotate', None) is not None:
        ax.annotate(s=plot_params['annotate'], fontsize=25, weight='bold',
                    xy=(0.01, 1.02), xycoords='axes fraction')

    plt.xlabel('dim 1', fontsize=18)
    plt.ylabel('dim 2', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.locator_params(axis='x', nbins=4)
    handles, labels = ax.get_legend_handles_labels()
    del handles[0], labels[0]

    if 'inset' in labels[0]:
        del handles[0], labels[0]

    if plot_params.get('legend_visible', True):
        ax.legend(handles=handles, labels=labels, loc='best', fontsize=10, ncol=1, markerscale=2)

    if plot_params.get('apply_tight_layout', True):
        plt.tight_layout()

    return ax


def visualize_embedding_space_with_topology(embeddings, labels, nodeG, edgeG, ax, plot_params=None):
    if plot_params is None:
        plot_params = {}

    assert (embeddings.shape[1] == 2)

    # Generate graph
    col_pal = plot_params.get('color_palette', 'crest')
    my_cmap = None
    if col_pal == 'crest':
        my_cmap = sns.color_palette('crest', as_cmap=True)
    elif col_pal == 'bright':
        cp = sns.color_palette('bright').as_hex()
        my_cmap = ListedColormap(cp)

    # Setup inset axes
    inset = plot_params.get('inset', False)
    label_every = plot_params.get('label_every', 5)

    axins = None
    inset_limit = None
    if inset:
        inset_axes = plot_params.get('inset_axes', [0.55, 0.55, 0.4, 0.4])
        inset_limit = plot_params.get('inset_limit', 5)
        axins = ax.inset_axes(inset_axes)

    # Plot
    zoom_in = plot_params.get('zoom_in', None)
    included_node_set = set()  # if zooming in, we need to have a list of nodes to not plot with nx
    unique_labels = np.unique(labels)
    for i, ul in enumerate(unique_labels):
        if zoom_in is not None and i >= zoom_in:
            break
        mask = (labels == ul)

        if col_pal == 'crest':
            cmap_val = i / max(unique_labels)
        else:
            cmap_val = i % my_cmap.N

        if type(ul) is str:
            label = ul
        else:
            label = int(ul)

        if (ul == -1 or ul == '-1') and zoom_in is None:
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1], color='black', marker='x', alpha=1, s=5, zorder=5,
                       label='X')
            continue
        elif ul == -1 or ul == '-1' and zoom_in is not None:
            zoom_in += 1
            continue

        included_node_set.update(np.where(mask)[0].tolist())

        if i % label_every != 0:
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1], color=my_cmap(cmap_val), alpha=1, s=5, zorder=10)
        else:
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1], color=my_cmap(cmap_val), alpha=1, s=5, zorder=10,
                       label=label)
        if inset and i < inset_limit:
            axins.scatter(embeddings[mask, 0], embeddings[mask, 1], color=my_cmap(cmap_val), alpha=1, s=5, zorder=15)

    if inset:
        axins.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.indicate_inset_zoom(axins, zorder=15, edgecolor='black', alpha=0.8)

    if plot_params.get('annotate', None) is not None:
        ax.annotate(s=plot_params['annotate'], fontsize=17, weight='bold',
                    xy=(0.025, 0.89), xycoords='figure fraction')

    # Add Topology
    if zoom_in is not None:
        removal_nodes = set(nodeG.nodes).difference(included_node_set)
        removal_edges = []
        for e in edgeG.edges():
            if e[0] in removal_nodes or e[1] in removal_nodes:
                removal_edges.append(e)
        nodeG.remove_nodes_from(removal_nodes)
        edgeG.remove_edges_from(removal_edges)
        edgeG.remove_nodes_from(removal_nodes)

    node_inds = list(nodeG.nodes)
    pos = dict(zip(list(nodeG.nodes), embeddings[node_inds, :]))
    rel = nx.get_edge_attributes(edgeG, 'rel')
    color_dict = {-1: 'royalblue', 0: 'black', 1: 'firebrick'}

    edge_list = []
    updown_edges = []
    updown_colors = []
    edge_list_colors = []
    for k, v in rel.items():
        if v == 0:
            edge_list.append(k)
            edge_list_colors.append(color_dict[v])
        else:
            updown_edges.append(k)
            updown_colors.append(color_dict[v])

    nx.draw_networkx_edges(edgeG, pos=pos, edge_color=updown_colors, edgelist=updown_edges, alpha=0.2,
                           connectionstyle='arc3,rad=-0.45', arrowsize=20)
    nx.draw_networkx_edges(edgeG, pos=pos, edge_color=edge_list_colors, edgelist=edge_list, alpha=0.2)

    plt.gca().tick_params(
        axis='both',
        which='both',
        bottom=True,
        left=True,
        labelbottom=True,
        labelleft=True)

    plt.xlabel('dim 1', fontsize=18)
    plt.ylabel('dim 2', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.locator_params(axis='x', nbins=4)
    handles, labels = ax.get_legend_handles_labels()

    if 'inset' in labels[0]:
        del handles[0], labels[0]

    ax.legend(handles=handles, labels=labels, loc='best', fontsize=10, ncol=1, markerscale=2)
    plt.tight_layout()

    return ax


def visualize_confusion_matrix(confusion_matrix, ax, plot_params=None):
    og_ax = ax
    ax = og_ax.inset_axes([0.0, 0.0, 0.95, 1], transform=og_ax.transAxes)
    confusion_matrix = confusion_matrix.copy()
    if plot_params is None:
        plot_params = {}

    my_cmap = cm.get_cmap(plot_params.get('cmap', 'viridis'))
    my_cmap.set_bad(my_cmap(0.1))

    # Normalize heatmap, replace 0 with 0.0001 for log scale
    confusion_matrix /= np.sum(confusion_matrix, axis=0)
    confusion_matrix[confusion_matrix == 0] = np.nan

    ylabel = 'Number of objects'
    xlabel = 'Cluster labels'

    column_names = plot_params.get('column_names', None)
    row_names = plot_params.get('row_names', None)

    if column_names is None:
        column_names = range(confusion_matrix.shape[1])
    if row_names is None:
        row_names = range(confusion_matrix.shape[0])

    if plot_params.get('transpose', False):
        im = ax.imshow(np.log10(confusion_matrix), cmap=my_cmap, vmin=-4)
        temp = column_names
        column_names = row_names
        row_names = temp
        temp = xlabel
        xlabel = ylabel
        ylabel = temp
    else:
        im = ax.imshow(np.log10(confusion_matrix.T), cmap=my_cmap, vmin=-4)

    if row_names is not None:
        if len(row_names) > 10:
            ticks = list(map(int, list(range(0, len(row_names), 2))))
            ax.set_yticks(ticks=ticks)
            ax.set_yticklabels(row_names[0::2])
        else:
            ax.set_yticks(ticks=list(range(len(row_names))))
            ax.set_yticklabels(row_names)

    if column_names is not None:
        if len(column_names) > 10:
            ticks = list(map(int, list(range(0, len(column_names), 2))))
            ax.set_xticks(ticks=ticks)
            ax.set_xticklabels(column_names[0::2])
        else:
            ax.set_xticks(ticks=list(range(0, len(column_names), 1)))
            ax.set_xticklabels(column_names[0::1])

    if plot_params.get('add_colorbar', True):
        # cax = og_ax.inset_axes([.96, 0.1, 0.02, .8], transform=og_ax.transAxes)
        cax = og_ax.inset_axes([.86, 0.00, 0.02, 1], transform=og_ax.transAxes)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_ticks([0, -1, -2, -3, -4])

    if plot_params.get('annotate', None) is not None:
        og_ax.annotate(s=plot_params['annotate'], fontsize=25, weight='bold',
                    xy=plot_params.get('annotate_pos', (0.0, 0.95)), xycoords='axes fraction')

    # og_ax.set_xticks([])
    # og_ax.set_yticks([])
    # og_ax.set_xlabel([])
    # og_ax.set_ylabel([])
    og_ax.set_axis_off()
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    return og_ax


def generate_histogram(actions, embeddings, ax, plot_params=None):
    if plot_params is None:
        plot_params = {}

    put_distances = []
    shake_distances = []
    take_distances = []

    if plot_params.get('training_limit', None) is not None:
        training_limit = plot_params['training_limit']
        counts = plot_params['embedding_counts']
        # ignore outside of training limit, if dataset has more than the training limit number of objects
        rm1 = (actions == -1) & (counts[:-1] <= training_limit)
        sm1 = (actions == 0) & (counts[:-1] <= training_limit)
        am1 = (actions == 1) & (counts[1:] <= training_limit)
    else:
        rm1 = (actions == -1)
        sm1 = (actions == 0)
        am1 = (actions == 1)

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
    else:
        take_dists_log = np.log(take_dists + 1)
        shake_dists_log = np.log(shake_dists + 1)
        put_dists_log = np.log(put_dists + 1)

    # HISTOGRAM LINEPLOT LOG #
    sns.set_color_codes()

    nbins = np.arange(-1, 3.2, 0.2)

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
    ax.set_xticks(list(range(-1, 4, 1)))
    xticks = ax.get_xticks()
    xticks_str = ['$10^{' + str(int(xt)) + '}$' for xt in xticks]
    ax.set_xticklabels(xticks_str, fontsize=14)
    ax.set_yticklabels(ax.get_yticks(), fontsize=14)
    ax.legend(loc='center left')

    if plot_params.get('annotate', None) is not None:
        ax.annotate(s=plot_params['annotate'], fontsize=17, weight='bold',
                    xy=plot_params.get('annotate_pos', (0.028, 0.94)), xycoords='figure fraction')

    return ax


# INCOMPLETE
def visualize_embedding_space_vs_intensity(embeddings, intensities, ax, plot_params=None):
    if plot_params is None:
        plot_params = {}

    assert (embeddings.shape[1] == 2)

    col_pal = plot_params.get('color_palette', 'crest')
    my_cmap = None
    if col_pal == 'crest':
        my_cmap = sns.color_palette('crest', as_cmap=True)
    elif col_pal == 'bright':
        cp = sns.color_palette('bright').as_hex()
        my_cmap = ListedColormap(cp)

    ax.scatter(embeddings[:, 0], embeddings[:, 1], color=my_cmap(intensities), alpha=1, s=5, zorder=10)

    if plot_params.get('annotate', None) is not None:
        ax.annotate(s=plot_params['annotate'], fontsize=17, weight='bold',
                    xy=(0.025, 0.89), xycoords='figure fraction')

    plt.xlabel('dim 1', fontsize=18)
    plt.ylabel('dim 2', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.locator_params(axis='x', nbins=4)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc='best', fontsize=10, ncol=1, markerscale=2)
    plt.tight_layout()

    return ax
