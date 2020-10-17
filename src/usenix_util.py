import os

import numpy as np
from scipy import ndimage, misc
import h5py


from src.preprocess_data import transform_to_one_hot, normalize_0_1

""" Constants for the model and the trigger set path 
"""
BASE_PATH = os.getcwd()[0:os.getcwd().rfind('Watermark')] + "Watermark/"
TRIGGER_SET_PATH = BASE_PATH + 'data/usenix_abstract_images/'  # Path to the trigger set relative to this file


def extract_features_h5py(model):
    """ Extracts features from a model as h5py view and puts it into a tuple 
    """
    lyr_wm = list(model.keys())[0]
    mean_weights_wm = np.array(model[lyr_wm][lyr_wm]['kernel:0']).mean(axis=3)[:, :, 0]
    var_weights_wm = np.array(model[lyr_wm][lyr_wm]['kernel:0']).var(axis=3)[:, :, 0]
    mean_bias_wm = np.array(model[lyr_wm][lyr_wm]['bias:0']).mean()
    var_bias_wm = np.array(model[lyr_wm][lyr_wm]['bias:0']).var()

    lst_mean = list(mean_weights_wm.flatten())
    lst_mean.append(mean_bias_wm)
    array_mean = np.array(lst_mean)

    lst_var = list(var_weights_wm.flatten())
    lst_var.append(var_bias_wm)
    array_var = np.array(lst_var)
    wm_X = np.hstack((array_mean, array_var))
    return wm_X


def extract_features_conv2d(layer):
    """ Extracts features from a model and puts it into a list
    """
    all_params = layer.get_weights()
    weights, biases = all_params[0], all_params[1]

    mean_weights_wm = weights.mean(axis=3)[:, :, 0]
    var_weights_wm = weights.var(axis=3)[:, :, 0]
    mean_bias_wm = biases.mean()
    var_bias_wm = biases.var()

    lst_mean = list(mean_weights_wm.flatten())
    lst_mean.append(mean_bias_wm)
    array_mean = np.array(lst_mean)

    lst_var = list(var_weights_wm.flatten())
    lst_var.append(var_bias_wm)
    array_var = np.array(lst_var)
    wm_X = np.hstack((array_mean, array_var))
    return wm_X


def load_trained_models_from_storage(path):
    """ Loads n_models many different  discriminators and keys from the storage for the property inference attack
    """
    X_list, y_list, wm_X_list, wm_y_list = [], [], [], []

    for i in range(0, 2700):
        idx_name = str(1000000 + i)[1:]

        # wm model exists
        if os.path.isfile(path + idx_name + '.h5wm.h5'):
            model = h5py.File(path + idx_name + '.h5wm.h5', 'r')['model_weights']
            wm_X = extract_features_h5py(model)

            wm_X_list.append(wm_X)
            wm_y_list.append(1)

        # non_wm model exists
        if os.path.isfile(path + idx_name + '.h5'):
            model = h5py.File(path + idx_name + '.h5', 'r')
            non_wm_X = extract_features_h5py(model)

            X_list.append(non_wm_X)
            y_list.append(0)
    return X_list, y_list, wm_X_list, wm_y_list


def reshape(path, x, y):
    """ Reshape an image to a certain size
    """
    image = ndimage.imread(path, mode="RGB")
    image = misc.imresize(image, (x, y))
    return image


def rgb2gray(rgb):
    """ Transform an rgb image to grayscale
    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def get_images(filepath, shape, nimages=100):
    """ Load all .jpg images from path and assign random labels
        @:param shape The desired image shape as width, height, channels
        @:return images, rnd labels
    """
    trigger_images = []
    for i in range(1, nimages + 1):
        curr_path = filepath + str(i) + '.jpg'
        curr_img = reshape(curr_path, shape[0], shape[1])
        if shape[-1] != curr_img.shape[-1] and shape[-1] == 1:
            curr_img = rgb2gray(curr_img)
        trigger_images.append(curr_img)
    trigger = np.reshape(np.array(trigger_images), (-1, *shape))
    # Choose random watermark classes
    all_y_wm = np.random.randint(10, size=len(trigger_images))
    return trigger, all_y_wm


def load_wm_images_usenix(trigger_set_path=TRIGGER_SET_PATH,
                          imgsize=(28, 28, 1)):
    """ Loads and normalizes USENIX watermark images if params are provided
    """
    wm_x, wm_y = get_images(trigger_set_path, imgsize)
    wm_x, = normalize_0_1([wm_x])
    wm_y = transform_to_one_hot(wm_y, depth=10)
    return wm_x, wm_y





