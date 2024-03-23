from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from shapely.geometry import Polygon
import data_creation.generate_polygon as grp
from noise_function import add_noise as add_noise_function

'''
This class handles generating a training dataset.

Parameters:
size - the size of the image
diameter_range - the range of diameters a shape can have
color_range - the range of pixel intensities to fill the shape with
shape_types - the types of shapes to generate for the dataset
'''


class GenerateDataByCount:
    # internal shape class
    class Shape:

        def __init__(self, size=None, vertices=None, holes=None, fill=255):
            self.size = size
            self.vertices = vertices
            if holes is None:
                holes = []
            self.holes = holes

            self.polygon_vertices = []
            self.polygon_holes = []
            self.bounds = list(map(lambda v: [min(v), max(v)], zip(*vertices)))

            self.polygon = None
            self.fill = fill

    def __init__(self, size, diameter_range=None, color_range=None, shape_options=None, seed=None, add_noise=False):
        self.size = size
        self.img = None
        self.img_draw = None
        self.seed = seed
        np.random.seed(seed)

        self.data_list = []
        self.label_list = []
        self.action_list = []
        self.count_list = []
        self.data = None
        self.labels = None
        self.actions = None
        self.action_sizes = None
        self.counts = None

        self.diameter_range = diameter_range
        self.color_range = color_range
        self.add_noise = add_noise

        self.background_color = 0
        self.shape_options = shape_options

        self.action_dict = {'shake': 0, 'add+shake': 1, 'remove+shake': -1}

    # creates a Shape object according to specified parameters
    def generate_object(self, diameter_range, color_range):

        # randomly sample shapes from a list of options
        shape_type = np.random.choice(self.shape_options)
        fill, vertices, holes, diameter = None, None, None, None

        if shape_type == 'square':
            p = None
            radius = int(np.random.choice(range(*diameter_range)) / 2)
            vertices = []

            # check if vertices define a valid polygon
            while p is None or not p.is_valid:
                vertices, _ = grp.generateSpecificPolygon(0, 0, radius, 'square')
                p = Polygon(vertices).difference(Polygon(holes))

        else:
            raise Exception('Unknown shape')

        fill = np.random.choice(range(*color_range))

        shape = self.Shape(size=diameter, vertices=vertices, holes=holes, fill=fill)
        return shape

    # create the initial image
    def init_image(self):
        img_arr = np.zeros(shape=self.size)
        self.img = Image.fromarray(img_arr)
        self.img_draw = ImageDraw.Draw(self.img)

        return img_arr

    '''
        add shape to image
        added_shapes - list of Polygon objects that are already in the images
        shape - the new Polygon object to add
    '''

    def add_shape_to_img(self, added_shapes, shape):

        # shape of image
        img_arr = np.asarray(self.img)
        arr_shape = img_arr.shape

        count = 0
        # grab vertices from shape object
        vertices = shape.vertices
        holes = shape.holes
        mod_vert = []
        mod_holes = []

        patch_found = None
        while patch_found is None:
            count += 1  # count number of attempts to find an empty patch

            # pick a random spot on the image
            x1 = np.random.randint(0, arr_shape[0])
            x2 = np.random.randint(0, arr_shape[1])
            mod_vert = []
            mod_holes = []
            fail = False
            for v in vertices:

                # check that each vertex, translated by x1 and x2 is within the bounds of the arr
                if not 0 < v[0] + x1 < arr_shape[0] or not 0 < v[1] + x2 < arr_shape[1]:
                    fail = True
                    break

                # track the list of modified vertices
                mod_vert.append((v[0] + x1, v[1] + x2))

            # if it worked for the outer points, it will work for holes vertices as well
            if not fail:
                for h in holes:
                    mod_holes.append((h[0] + x1, h[1] + x2))

            # skip if we failed vertex translation
            if fail:
                continue

            # if we succeed with translation, check that the location doesn't cause overlaps with
            # the already added shapes. Returns a Polygon (shapely) if it is valid.
            patch_found = self.check_polygon_intersection(added_shapes, mod_vert, mod_holes)

            # repeat for a maximum of 3000 times, if still failed try again with a different shape
            # if this happens too much, object statistics will be messed up...
            if count > 3000:
                print('Tried 3000 positions, image too small?')
                return False
        else:

            # if the patch is found succesfully, update the polygon vertex positions and the
            # shapely Polygon object
            shape.polygon = patch_found
            shape.polygon_vertices = mod_vert
            shape.polygon_holes = mod_holes

            # draw the polygon with PIL img draw
            self.img_draw.polygon(shape.polygon_vertices, fill=shape.fill)

            # add holes, if polygon has holes
            if len(shape.polygon_holes) != 0:
                self.img_draw.polygon(shape.polygon_holes, fill=self.background_color)

        return True

    # use the shapely Polygon class to enforce a margin between polygons
    def check_polygon_intersection(self, added_shapes, vertices, holes):
        polygon = Polygon(vertices)
        if len(holes) != 0:
            polygon = Polygon(vertices).difference(Polygon(holes))

        for s in added_shapes:
            if not s.polygon.buffer(3).disjoint(polygon):
                return None

        # return the polygon if it is a valid polygon with a valid margin
        return polygon

    def remove_shape(self, shape):
        self.img_draw.polygon(shape.polygon_vertices, fill=self.background_color)

    '''
        generate a balanced dataset according to the initial specified parameters
        n_shapes - maximum number of shapes
        n_operations - number of actions to perform to create the dataset
    '''
    def generate_dataset(self, n_shapes, n_repeats, images_dir=None):

        assert images_dir is not None
        os.makedirs(images_dir, exist_ok=True)

        img_num = 0
        for i in range(n_shapes + 1):
            for j in range(n_repeats):
                count = i
                self.init_image()
                added_shapes = set()
                for c in range(count):
                    shape = self.generate_object(self.diameter_range, self.color_range)
                    self.add_shape_to_img(added_shapes=added_shapes, shape=shape)
                    added_shapes.add(shape)

                img_arr = np.asarray(self.img)
                if self.add_noise:
                    img_arr = add_noise_function(img_arr)
                # img_arr = img_arr

                path = images_dir + f'img{img_num}.png'
                Image.fromarray(img_arr).convert('RGB').save(path, 'png')

                self.data_list.append(path)
                self.label_list.append(count)
                self.count_list.append(count)
                self.action_list.append(10)
                img_num += 1

        self.data = np.stack(self.data_list, axis=0)
        self.labels = np.stack(self.label_list, axis=0)
        self.actions = np.stack(self.action_list, axis=0)
        self.action_sizes = np.stack(self.action_list, axis=0)
        self.counts = np.stack(self.count_list, axis=0)

    def to_3channel(self):
        self.data = np.tile(self.data[:, :, None, :], reps=(1, 1, 3, 1))

    def save(self, tag='train', folder=''):
        path = '../data/raw/' + folder
        os.makedirs(path, exist_ok=True)
        data_path = os.path.join(path, ('data_' + tag + '.npy'))
        labels_path = os.path.join(path, ('labels_' + tag + '.npy'))
        actions_path = os.path.join(path, ('actions_' + tag + '.npy'))
        counts_path = os.path.join(path, ('counts_' + tag + '.npy'))
        np.save(data_path, self.data)
        np.save(labels_path, self.labels)
        np.save(actions_path, self.actions)
        np.save(counts_path, self.counts)

        keys = ['diameter_range', 'color_range', 'shape_options', 'seed']
        values = (self.diameter_range, self.color_range, self.shape_options, self.seed)
        pdict = dict(dict(zip(keys, values)))
        with open(path + '/params.txt', 'w') as f:
            json.dump(pdict, f, indent=1)

    def visualize(self, folder=''):
        path = '../data/raw/' + folder
        path1 = path + '/dataset_visualization.png'

        jump_to = np.where(self.counts == max(self.counts))[0][0]
        seq_len = 9
        jump_to = min(jump_to, len(self.counts) - seq_len - 1)

        c = 1
        axs = None
        fig = None
        if seq_len is not None and c <= seq_len:
            fig, axs = plt.subplots(int(seq_len / 3), int(seq_len / 3))

        for i in range(jump_to, jump_to + seq_len):
            ax = axs.flatten()[c - 1]
            im = ax.imshow(Image.open(self.data[i]), cmap='gray')

            if c == 1:
                fig.colorbar(im, ax=ax)
            ax.set_title(str(self.counts[i]), size=18)
            c += 1

        plt.tight_layout(h_pad=0.5)
        plt.savefig(path1)
        plt.close()

        path2 = path + '/pixel_intensities_vs_count.png'
        pixel_intensities = []
        data_list = []
        for d in self.data:
            data_list.append(Image.open(d))
            pixel_intensities.append(np.mean(data_list[-1]))

        pixel_intensities = np.array(pixel_intensities)
        intensity_list = []
        counts_list = []
        for un in sorted(np.unique(self.counts)):
            mask = self.counts == un
            intensity_list.append(pixel_intensities[mask])
            counts_list.append(un)

        plt.figure()
        ax = plt.gca()
        ax.violinplot(intensity_list, counts_list, widths=0.3)
        plt.ylabel('total intensity')
        plt.xlabel('count')
        plt.savefig(path2)
        plt.close()