import numpy as np
from analysis.compute_confidence_intervals import get_confidence_intervals_bayesian, propci_wilson_cc, \
    generate_ci_over_list


ci_method = get_confidence_intervals_bayesian


def parse_correct_per_action_dictionary(correct_per_action_dict):
    # parse the correct_per_action dict
    take = []
    shake = []
    put = []
    for key in correct_per_action_dict.keys():
        err = correct_per_action_dict[key][1] - correct_per_action_dict[key][0] + 1
        total = correct_per_action_dict[key][1] + 1
        if '>' in key:
            num = float(key.strip('>'))
            put.append((num, err, total))
        elif '-' in key:
            num = float(key.strip('-'))
            shake.append((num, err, total))
        elif '<' in key:
            num = float(key.strip('<'))
            take.append((num, err, total))
        else:
            raise Exception('Symbol not understood')

    return take, shake, put


def plot_action_error(correct_per_action, axes, split_point, label='', plot_params=None):

    if plot_params is None:
        plot_params = {}

    assert(len(axes) == 3)
    take, shake, put = parse_correct_per_action_dictionary(correct_per_action)

    actions = [take, shake, put]
    action_names = ['Take', 'Shake', 'Put']
    num_samples_list = []

    psm = plot_params.get('post_split_marker', '-o')

    for i, ax in enumerate(axes):
        curr_action = actions[i]
        curr_action = sorted(curr_action, key=lambda x: x[0])  # sort by nums least to greatest
        nums, err, total = zip(*curr_action)
        num_samples_list.append(list(total))

        nums = np.array(nums)
        low, mid, high = generate_ci_over_list(ci_method, err, total)

        if action_names[i] == 'Shake':

            # split between training and test region
            col = ax.plot(nums[0:split_point + 1], mid[0:split_point + 1], '-o')[0].get_color()
            ax.plot(nums[split_point + 1:], mid[split_point + 1:], psm, color=col)

            # confidence shadow
            ax.fill_between(range(len(mid)), low, high, alpha=0.3)

            if plot_params.get('training_limit_shadow_visible', True):
                # shadow beyond training limit
                xlims = ax.get_xlim()
                ax.axvspan(xlims[0], split_point + 0.5, facecolor='white', alpha=0.3)
                ax.axvspan(split_point + 0.5, xlims[1], facecolor='lightgrey', alpha=0.2)
                ax.set_xlim(xlims)

        elif action_names[i] == 'Put':
            col = ax.plot(nums[0:split_point], mid[0:split_point], '-o')[0].get_color()
            ax.plot(nums[split_point:], mid[split_point:], psm, color=col)
            ax.fill_between(range(len(mid)), low, high, alpha=0.3)

            if plot_params.get('training_limit_shadow_visible', True):
                xlims = ax.get_xlim()
                ax.axvspan(xlims[0], split_point - 1 + 0.5, facecolor='white', alpha=0.3)
                ax.axvspan(split_point - 1 + 0.5, xlims[1], facecolor='lightgrey', alpha=0.2)
                ax.set_xlim(xlims)
        else:
            col = ax.plot(nums[0:split_point], mid[0:split_point], '-o', label=label)[0].get_color()
            ax.plot(nums[split_point:], mid[split_point:], psm, color=col)
            ax.fill_between(range(1, len(mid) + 1), low, high, alpha=0.3)

            if plot_params.get('training_limit_shadow_visible', True):
                xlims = ax.get_xlim()
                ax.axvspan(xlims[0], split_point + 0.5, facecolor='white', alpha=0.3)
                ax.axvspan(split_point + 0.5, xlims[1], facecolor='lightgrey', alpha=0.2)
                ax.set_xlim(xlims)

        ax.set_yscale('log')
        ax.set_ylim([1e-4, 1])
        ax.tick_params(axis='y', labelsize=15)
        ax.tick_params(axis='x', labelsize=12)
        ax.set_xticks(np.unique(nums))
        ax.set_title('{a}'.format(a=action_names[i]), fontsize=18)

    return axes, num_samples_list


def plot_action_error_against_embedding_dimension(correct_per_action_list, dimension_list, ax, split_point):

    lows = {'Take': [], 'Shake': [], 'Put': []}
    mids = {'Take': [], 'Shake': [], 'Put': []}
    highs = {'Take': [], 'Shake': [], 'Put': []}
    num_samples = {'Take': [], 'Shake': [], 'Put': []}
    for correct_per_action in correct_per_action_list:

        take, shake, put = parse_correct_per_action_dictionary(correct_per_action)

        actions = [take, shake, put]
        action_names = ['Take', 'Shake', 'Put']

        for i, curr_action in enumerate(actions):
            curr_action = sorted(curr_action, key=lambda x: x[0])  # sort by nums least to greatest
            nums, err, total = zip(*curr_action)
            num_samples[action_names[i]].append(list(total))
            split_ind = np.where(np.array(nums) == split_point)[0][0]
            if not action_names[i] == 'Put':
                split_ind += 1

            err = err[:split_ind]
            total = total[:split_ind]

            low, mid, high = ci_method(sum(err), sum(total))
            lows[action_names[i]].append(low)
            mids[action_names[i]].append(mid)
            highs[action_names[i]].append(high)

    for key in lows.keys():
        ax.plot(dimension_list, mids[key], '-o', label=key)
        ax.fill_between(dimension_list, lows[key], highs[key], alpha=0.3)

    ax.set_xscale('log', basex=2)
    ax.set_xticks(dimension_list)
    ax.set_yscale('log')
    ax.set_ylim(top=0.1)
    ax.tick_params(axis='y', labelsize=15)
    ax.tick_params(axis='x', labelsize=12)

    ax.legend()

    return ax, num_samples
