import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


def save_asiaccs_train_surr_model_to_file(filepath, model, history):
    """ Loads the data required for the usnx train_surr_model function
    """
    MODEL_SUFFIX, HISTORY_SUFFIX = "ASIACCS_MODEL", "ASIACCS_HISTORY"

    if filepath is not None:
        filepath = os.getcwd()[0:os.getcwd().rfind('Watermark')] + "Watermark/tmp/" + filepath
        model.save_weights(filepath + MODEL_SUFFIX)
        with open(filepath + HISTORY_SUFFIX, 'wb') as pickle_file:
            pickle.dump(history, pickle_file)


def load_asiacss_train_surr_model_from_file(model, filepath):
    """ Loads the data required for the usnx train_surr_model function
    """
    MODEL_SUFFIX, HISTORY_SUFFIX = "ASIACCS_MODEL", "ASIACCS_HISTORY"

    if filepath is not None:
        filepath = os.getcwd()[0:os.getcwd().rfind('Watermark')] + "Watermark/tmp/" + filepath
        if os.path.isfile(filepath + MODEL_SUFFIX) and os.path.isfile(filepath + HISTORY_SUFFIX):
            model.load_weights(filepath + MODEL_SUFFIX)
            with open(filepath + HISTORY_SUFFIX, 'rb') as pickle_file:
                history = pickle.load(pickle_file)
            return model, history
    return None, None


def save_asiaccs_embed_model_to_file(filepath, model, history):
    """ Writes the data generated for the usnx embed_model function to disk
    """
    MODEL_SUFFIX, HISTORY_SUFFIX = "ASIACCS_MODEL", "ASIACCS_HISTORY"

    if filepath is not None:
        filepath = os.getcwd()[0:os.getcwd().rfind('Watermark')] + "Watermark/tmp/" + filepath
        model.save_weights(filepath + MODEL_SUFFIX)
        with open(filepath + HISTORY_SUFFIX, 'wb') as pickle_file:
            pickle.dump(history, pickle_file)


def load_asiaccs_embed_model_from_file(model, filepath):
    """ Loads the data required for the usnx embed_model function
    """
    MODEL_SUFFIX, HISTORY_SUFFIX = "ASIACCS_MODEL", "ASIACCS_HISTORY"

    if filepath is not None:
        filepath = os.getcwd()[0:os.getcwd().rfind('Watermark')] + "Watermark/tmp/" + filepath
        if os.path.isfile(filepath + MODEL_SUFFIX) and os.path.isfile(filepath + HISTORY_SUFFIX):
            model.load_weights(filepath + MODEL_SUFFIX)
            with open(filepath + HISTORY_SUFFIX, 'rb') as pickle_file:
                history = pickle.load(pickle_file)
            return model, history
    return None, None, None


def embed_watermark_gaussian(x: np.ndarray,
                             y: np.ndarray,
                             wm_class=5,
                             factor=None):
    mean = 0
    var = 0.01
    sigma = var ** 0.5
    noise_x, noise_y = [], []

    if factor is None:
        factor = x.mean()

    plt.subplot(1, 2, 1)
    print("(Gaussian) Plot class: {}".format(y[0]))
    # Plt does not accept if last axis has dimension "1"
    if x.shape[-1] == 1:
        plt.imshow(np.reshape(x[0], (x[0].shape[0], x[0].shape[1])))
    else:
        plt.imshow(x[0])

    for img in x:
        row, col, channels = img.shape
        gauss = np.random.normal(mean, sigma, (row, col, channels))
        gauss *= factor
        noise_x.append(img + gauss)
        label = [0] * 10
        label[wm_class] = 1
        noise_y.append(label)

    plt.subplot(1, 2, 2)
    print("(Gaussian) Plot class: {}".format(noise_y[0]))
    # Plt does not accept if last axis has dimension "1"
    if np.array(noise_x).shape[-1] == 1:
        plt.imshow(np.reshape(noise_x[0], (noise_x[0].shape[0], noise_x[0].shape[1])))
    else:
        plt.imshow(noise_x[0])
    plt.show()
    return np.array(noise_x), np.array(noise_y)


def embed_watermark_test(x: np.ndarray,
                         y: np.ndarray,
                         wm_class=5,
                         freq=1):
    """ Embeds the string "TEST" in the watermark image
    """
    mh = x.shape[2]  # max height
    watermark_color = x.max()/2
    print("Watermark color: {}".format(watermark_color))
    watermark_coords = {
        mh-9: [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25],
        mh-8: [3, 8, 13, 17, 22],
        mh-7: [3, 8, 14, 22],
        mh-6: [3, 8, 9, 10, 11, 15, 22],
        mh-5: [3, 8, 16, 22],
        mh-4: [3, 8, 13, 17, 22],
        mh-3: [3, 8, 9, 10, 11, 14, 15, 16, 22]
    }
    plt.subplot(1, 2, 1)
    print("(Embed Logo) Plot class: {}".format(y[0]))
    # Plt does not accept if last axis has dimension "1"
    if x.shape[-1] == 1:
        plt.imshow(np.reshape(x[0], (x[0].shape[0], x[0].shape[1])))
    else:
        plt.imshow(x[0])

    for img in range(0, len(x), freq):
        for i in watermark_coords:
            for j in watermark_coords[i]:
                x[img][i + 1][j + 1] = watermark_color
        y[img] = [0] * 10
        y[img][wm_class] = 1
    plt.subplot(1, 2, 2)
    # Plt does not accept if last axis has dimension "1"
    if x.shape[-1] == 1:
        plt.imshow(np.reshape(x[0], (x[0].shape[0], x[0].shape[1])))
    else:
        plt.imshow(x[0])
    plt.show()

    return x, y


def load_wm_images_asiaccs(dataset,
                           n_size,
                           wm_class=5,
                           type='gaussian'):
    """ Loads a set of watermarked CIFAR-10 images.
    """
    # To make sure, always copy the dataset
    dataset = (dataset[0].copy(), dataset[1].copy())
    x, y = shuffle(*dataset)

    if type == 'logo':
        print("Embedding {} watermarks".format(type))
        x, y = embed_watermark_test(x, y, wm_class=wm_class)
    elif type == 'gaussian':
        print("Embedding {} watermarks".format(type))
        x, y = embed_watermark_gaussian(x, y, wm_class=wm_class)
    else:
        raise Exception("{} is not an available Watermark embedding approach".format(type))

    return x[:n_size], y[:n_size]
