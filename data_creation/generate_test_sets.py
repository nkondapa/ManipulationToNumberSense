from data_creation.generate_data_by_action import GenerateDataByAction
from data_creation.generate_data_by_count import GenerateDataByCount
from data_creation.process_data import *
import os

IMG_SIZE = (244, 244)
IMG_SHAPES = ['square']
GENERATOR_OBJECT_CLASS = [GenerateDataByAction, GenerateDataByCount]


# DATASET CONTRAST 100%, SIZE 15
cs = [8, 30]
ds_sizes = [8000, 150]

experiment_name = 'high_contrast_same_size_test_set'
dr = [(15, 16)] * len(cs)
cr = [(255, 256)] * len(cs)
for ci, c in enumerate(cs):
    test_set1_name = experiment_name + '/test_set{s}_n_obj={c}_d_size={dsize}'.format(s=ci, c=c, dsize=ds_sizes[ci])

    img_dir = '../data/raw/' + test_set1_name + '/images/'
    if not os.path.exists('../data/processed/' + test_set1_name):
        print(test_set1_name)
        GenerateData = GENERATOR_OBJECT_CLASS[ci]
        gd = GenerateData(IMG_SIZE, diameter_range=dr[ci], color_range=cr[ci], shape_options=IMG_SHAPES, seed=0)

        gd.generate_dataset(c, ds_sizes[ci], images_dir=img_dir)

        gd.save('train', folder=test_set1_name)
        gd.visualize(test_set1_name)
        save(process_data_with_actions_sizes(gd.data, gd.labels, gd.actions, gd.counts, gd.action_sizes), 'train', folder=test_set1_name)

        gd = GenerateData(IMG_SIZE, diameter_range=dr[ci], shape_options=IMG_SHAPES, color_range=cr[ci], seed=1)

        gd.generate_dataset(c, ds_sizes[ci], images_dir=img_dir)

        gd.save('test', folder=test_set1_name)
        gd.visualize(test_set1_name)
        save(process_data_with_actions_sizes(gd.data, gd.labels, gd.actions, gd.counts, gd.action_sizes), 'test', folder=test_set1_name)


########################################################################################################################

# DATASET CONTRAST 9.7% - 100%, SIZE 10-30px
cs = [8, 30]
ds_sizes = [8000, 150]

experiment_name = 'random_contrast_random_sizes_test_set'
dr = [(10, 30)] * len(cs)
cr = [(50, 256)] * len(cs)
for ci, c in enumerate(cs):
    test_set1_name = experiment_name + '/test_set{s}_n_obj={c}_d_size={dsize}'.format(s=ci, c=c, dsize=ds_sizes[ci])

    img_dir = '../data/raw/' + test_set1_name + '/images/'
    if not os.path.exists('../data/processed/' + test_set1_name):
        GenerateData = GENERATOR_OBJECT_CLASS[ci]
        print(test_set1_name)
        gd = GenerateData(IMG_SIZE, diameter_range=dr[ci], color_range=cr[ci], shape_options=IMG_SHAPES, seed=0)

        gd.generate_dataset(c, ds_sizes[ci], images_dir=img_dir)

        gd.save('train', folder=test_set1_name)
        gd.visualize(test_set1_name)
        save(process_data_with_actions_sizes(gd.data, gd.labels, gd.actions, gd.counts, gd.action_sizes), 'train', folder=test_set1_name)

        gd = GenerateData(IMG_SIZE, diameter_range=dr[ci], shape_options=IMG_SHAPES, color_range=cr[ci], seed=1)

        gd.generate_dataset(c, ds_sizes[ci], images_dir=img_dir)

        gd.save('test', folder=test_set1_name)
        gd.visualize(test_set1_name)
        save(process_data_with_actions_sizes(gd.data, gd.labels, gd.actions, gd.counts, gd.action_sizes), 'test', folder=test_set1_name)


# ########################################################################################################################

# DATASET CONTRAST 9.7% - 100%, SIZE 10-30px
cs = [8, 30]
ds_sizes = [8000, 150]
GENERATOR_OBJECT_CLASS = [GenerateDataByAction, GenerateDataByCount]

experiment_name = 'jittered_contrast_jittered_sizes_test_set'
dr = [(10, 30)] * len(cs)
cr = [(50, 256)] * len(cs)
noise_opts = [False]
# color_jitter_opts = [1, 5, 15]
# size_jitter_opts = [1, 3, 10]

color_jitter_opts = [5]
size_jitter_opts = [3]

ci = 0
c = cs[ci]
for noise_opt in noise_opts:
    for cj in color_jitter_opts:
        for sj in size_jitter_opts:
            test_set1_name = experiment_name + f'/test_set{ci}_n_obj={c}_d_size={ds_sizes[ci]}_noise={noise_opt}_cj={cj}_sj={sj}'
            img_dir = '../data/raw/' + test_set1_name + '/images/'
            if not os.path.exists('../data/processed/' + test_set1_name):
                GenerateData = GENERATOR_OBJECT_CLASS[ci]
                print(test_set1_name)
                gd = GenerateData(IMG_SIZE, diameter_range=dr[ci], color_range=cr[ci], color_jitter=cj, size_jitter=sj, shape_options=IMG_SHAPES, seed=0, add_noise=noise_opt)

                gd.generate_dataset(c, ds_sizes[ci], images_dir=img_dir)
                
                gd.save('train', folder=test_set1_name)
                gd.visualize(test_set1_name)
                save(process_data_with_actions_sizes(gd.data, gd.labels, gd.actions, gd.counts, gd.action_sizes),
                     'train', folder=test_set1_name)

                gd = GenerateData(IMG_SIZE, diameter_range=dr[ci], shape_options=IMG_SHAPES, color_jitter=cj, size_jitter=sj, color_range=cr[ci], seed=1, add_noise=noise_opt)

                gd.generate_dataset(c, ds_sizes[ci], images_dir=img_dir)
                
                gd.save('test', folder=test_set1_name)
                gd.visualize(test_set1_name)
                save(process_data_with_actions_sizes(gd.data, gd.labels, gd.actions, gd.counts, gd.action_sizes),
                     'test', folder=test_set1_name)

ci = 1
c = cs[ci]
for noise_opt in noise_opts:
    test_set1_name = experiment_name + f'/test_set{ci}_n_obj={c}_d_size={ds_sizes[ci]}_noise={noise_opt}'
    img_dir = '../data/raw/' + test_set1_name + '/images/'
    if not os.path.exists('../data/processed/' + test_set1_name):
        print(test_set1_name)

        GenerateData = GENERATOR_OBJECT_CLASS[ci]
        gd = GenerateData(IMG_SIZE, diameter_range=dr[ci], color_range=cr[ci], shape_options=IMG_SHAPES, seed=0,
                          add_noise=noise_opt)

        gd.generate_dataset(c, ds_sizes[ci], images_dir=img_dir)
        
        gd.save('train', folder=test_set1_name)
        gd.visualize(test_set1_name)
        save(process_data_with_actions_sizes(gd.data, gd.labels, gd.actions, gd.counts, gd.action_sizes), 'train',
             folder=test_set1_name)

        gd = GenerateData(IMG_SIZE, diameter_range=dr[ci], shape_options=IMG_SHAPES, color_range=cr[ci], seed=1,
                          add_noise=noise_opt)

        gd.generate_dataset(c, ds_sizes[ci], images_dir=img_dir)
        
        gd.save('test', folder=test_set1_name)
        gd.visualize(test_set1_name)
        save(process_data_with_actions_sizes(gd.data, gd.labels, gd.actions, gd.counts, gd.action_sizes), 'test', folder=test_set1_name)
