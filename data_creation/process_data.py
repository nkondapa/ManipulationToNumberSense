import os
import torch
from PIL import Image


def process_data(img_arr_stack, labels, actions, counts):

    data = torch.tensor(img_arr_stack.transpose(), dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.float)
    actions = torch.tensor(actions, dtype=torch.float)
    counts = torch.tensor(counts, dtype=torch.float)
    return data, labels, actions, counts


def process_data_with_actions_sizes(img_arr_stack, labels, actions, counts, action_sizes):
    # data = torch.tensor(img_arr_stack.transpose(), dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.float)
    actions = torch.tensor(actions, dtype=torch.float)
    counts = torch.tensor(counts, dtype=torch.float)
    action_sizes = torch.tensor(action_sizes, dtype=torch.float)

    return img_arr_stack, labels, actions, counts, action_sizes


def save(torch_tuple, tag='train', folder=''):
    path = '../data/processed/' + folder
    os.makedirs(path, exist_ok=True)
    data_path = os.path.join(path, (tag + '.pt'))

    with open(data_path, 'wb') as f:
        torch.save(torch_tuple, f)


