import matplotlib.pyplot as plt
import torch
import numpy as np

def generate_action_prediction_psychometric_curve(embedding_dict, left, right_list, predict, ax):

    # left = 0
    embedding_val_left = embedding_dict[left]
    results_dict = {}
    for right in right_list:
        embedding_val_right = embedding_dict[right]
        results_dict[(left, right)] = [0, 0]
        actions = []
        for evl in embedding_val_left:
            acts = predict(torch.tensor([evl]).repeat(embedding_val_right.shape[0], 1), torch.tensor(embedding_val_right))
            print(right, acts)
            actions.append(acts.numpy())

        actions = np.array(actions)
        take = (actions == 0).sum()
        eq = (actions == 1).sum()
        put = (actions == 2).sum()
        results_dict[(left, right)] = [take + eq/2, put + eq/2]

    xvals = []
    yvals = []
    for key in results_dict.keys():
        xvals.append(key[1])
        yvals.append(results_dict[key][1]/(results_dict[key][0] + results_dict[key][1]))

    ax.plot(xvals, yvals, label='PTS-clf', zorder=-1, linewidth=5, alpha=0.5, color='magenta')
    ax.set_xlabel('Test number', fontsize=20)
    ax.set_ylabel('Proportion "more"', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    return ax
