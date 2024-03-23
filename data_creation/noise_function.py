import numpy as np


def add_noise(img_arr, std_dev=25):
    gaussian = np.random.normal(0, std_dev, (img_arr.shape[0], img_arr.shape[1]))
    img_arr = img_arr + gaussian
    img_arr[img_arr > 255] = 255
    img_arr[img_arr < 0] = 0
    return img_arr