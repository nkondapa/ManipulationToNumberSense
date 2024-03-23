import torch
import random
import numpy as np
from PIL import Image
from data_creation.generate_data_by_action import GenerateDataByAction
import data_creation.process_data as pd
import os
import json
import global_variables as gv

'''
Generate a dataset between epochs of training a neural network. Removes the need to store or compress many images.
Training data will not persist, but the training data will have the same statistics in experiments.

Parameters:

num_batches - number of batches
batch_size - how many actions to perform, one less than the number of images per batch.
transforms - how to transform the image before passing it to the neural net

** GenerateData specific parameters, see Generate Data for more information **
experiment_name - leave this to default, it's a temporary storage location for the output of the GenerateData class
color_modes - set this to ['normal'], no other color modes are defined here
action_modes - set this to ['complex'] no other action modes are defined here
diameter_ranges - the range of "diameters" that objects in the image can take. Note that for squares, the diameter is
equal to the edge of the square
color_ranges - the pixel values that an object might take
shape_types_options - the types of shapes allowed in the image, set this to ['square']
num_objects - number of objects in the images


Returns:
self - an object that extends the pytorch Dataset object and defines the __get__item() method for pytorch dataloaders.

'''


class OnlineSequentialPairDataLoader(torch.utils.data.Dataset):

    def __init__(self, num_batches, batch_size, experiment_name, epoch, seed=None,
                 diameter_ranges=None, size_jitter=None, color_ranges=None, color_jitter=None, add_noise=False,
                 shape_types_options=None, transforms=None, num_objects=3,
                 save=True, dataset_save_name='temp', min_num_objects=0, action_size=1):

        # within batch can change the diameter range
        diameter_range = random.choice(diameter_ranges)
        color_range = random.choice(color_ranges)
        shape_options = random.choice(shape_types_options)

        self.size_jitter = size_jitter
        self.color_jitter = color_jitter
        self.add_noise = add_noise
        self.num_objects = num_objects
        self.min_num_objects = 0
        self.diameter_ranges = diameter_ranges
        self.color_ranges = color_ranges
        self.shape_options = shape_options
        self.action_size_limit = action_size
        self.seed = seed

        self.num_batches = num_batches
        self.batch_size = batch_size
        self.experiment_name = experiment_name
        self.target_transform = {1: torch.tensor([[0, 0, 1]]), 0: torch.tensor([[0, 1, 0]]), -1: torch.tensor([[1, 0, 0]])}
        self.transforms = transforms

        self.data = []
        self.labels = []
        self.actions = []
        self.counts = []
        self.action_sizes = []

        np.random.seed(seed)
        seed_sequence = np.random.randint(0, 1000000, self.num_batches).tolist()
        print(seed_sequence)
        # seed_sequence = np.arange(0, self.num_batches).tolist()

        print('Generating Data...')
        for i in range(num_batches):

            if i % 5 == 0:
                print('Generated batch #', i)

            dataset_size = self.batch_size
            object_set1_name = dataset_save_name + '/temp_object_set_dsize={dsize}'.format(dsize=dataset_size)

            gd = GenerateDataByAction((244, 244), diameter_range=diameter_range, color_range=color_range,
                                      shape_options=shape_options, seed=seed_sequence[i],
                                      color_jitter=color_jitter, size_jitter=size_jitter, add_noise=add_noise,
                                      )

            img_dir = gv.DATA_FOLDER + 'raw/' + object_set1_name + '/images/'
            gd.generate_dataset(num_objects, dataset_size, images_dir=img_dir, min_num_objects=min_num_objects,
                                img_start_num=(i * (dataset_size + 1)),
                                action_size= self.action_size_limit)

            # save a sample for inspection
            if i == 0:
                gd.save('train', folder=object_set1_name)
                gd.visualize(object_set1_name)

            data, labels, actions, counts, action_sizes = pd.process_data_with_actions_sizes(gd.data, gd.labels, gd.actions, gd.counts, gd.action_sizes)

            self.data.append(data)
            self.labels.append(labels)
            self.actions.append(actions)
            self.counts.append(counts)
            self.action_sizes.append(action_sizes)

        self.data = np.concatenate(self.data)
        self.labels = torch.cat(self.labels)
        self.actions = torch.cat(self.actions)
        self.counts = torch.cat(self.counts)
        self.action_sizes = torch.cat(self.action_sizes)

        if save:
            train_data_path = f'{gv.ROOT}/data/training/{experiment_name}/{epoch}/'
            os.makedirs(train_data_path, exist_ok=True)

            pixel_intensities = []
            for d in self.data:
                pixel_intensities.append(np.mean(Image.open(d)))

            np.save(train_data_path + 'actions.npy', np.int8(self.actions))
            np.save(train_data_path + 'action_sizes.npy', np.int8(self.action_sizes))
            np.save(train_data_path + 'counts.npy', np.int8(self.counts))
            np.save(train_data_path + 'intensities.npy', np.array(pixel_intensities, dtype=np.float) / 255)

            keys = ['size_jitter', 'color_jitter', 'add_noise', 'diameter_ranges',
                    'color_ranges', 'shape_options', 'seed', 'min_num_objects', 'action_size_limit']
            values = (self.size_jitter, self.color_jitter, self.add_noise, self.diameter_ranges,
                      self.color_ranges, self.shape_options, self.seed, self.min_num_objects,
                      self.action_size_limit)
            pdict = dict(dict(zip(keys, values)))
            with open(train_data_path + '/params.txt', 'w') as f:
                json.dump(pdict, f, indent=1)

    def __getitem__(self, index):

        # Return an image at the specified index of the dataset.
        # Note that indices are tied to the actions in the dataset NOT the image list itself.
        try:
            # convert the action code in the dataset to a torch one-hot-encoded tensor
            target = self.target_transform[self.actions[index].item()]

            # there are num_batches with batch_size + 1 images in each dataset.
            # the offset skips the images at the end of the batch x and at the beginning of batch x + 1 (which are
            # not related by an action.
            offset = int(index/self.batch_size)
            imgs = [Image.open(self.data[index + offset]), Image.open(self.data[index + offset + 1])]
        except IndexError as e:
            raise e

        # convert PIL images
        img_ar = []
        for i in range(len(imgs)):
            img = imgs[i]

            # transform image if transforms are passed
            if self.transforms is not None:
                img = self.transforms(img)

            img_ar.append(img.transpose(1, 2))

        return img_ar, target

    # index is drawn from range of length of actions
    def __len__(self):
        return len(self.actions)
