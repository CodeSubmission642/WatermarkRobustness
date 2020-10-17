from tensorflow.keras.datasets import mnist, cifar10, cifar100
import numpy as np
import os
from emnist import extract_training_samples

from keras_preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle

from src.util import save_2d_numpy_array_if_exists, load_2d_numpy_array_if_exists


def augment_data(set_to_augment,
                 prefix="",
                 total_size=35000,
                 batchsize=64,
                 use_cached_training_data=None,
                 verbose=True):
    """ Helper function to load the CIFAR-10 data
    """
    filepath = prefix + use_cached_training_data + str(total_size)
    # Look for cached training data
    if use_cached_training_data:
        x, y = load_2d_numpy_array_if_exists(filepath)
        if x is not None and y is not None:
            print("     Found cached training data for {}".format(use_cached_training_data))
            return (x, y), True

    # Enhance training set with augmentation
    generated_data = set_to_augment[0].copy(), set_to_augment[1].copy()
    if not len(generated_data[0]) >= total_size:
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
        )
        datagen.fit(set_to_augment[0])
        generator = datagen.flow(set_to_augment[0], set_to_augment[1], batch_size=batchsize)
        print_percentage = 0.1
        while len(generated_data[0]) < total_size:
            next_sample = generator.next()
            generated_data = np.concatenate((generated_data[0], next_sample[0]), axis=0), \
                             np.concatenate((generated_data[1], next_sample[1]), axis=0)

            if verbose and len(generated_data[0]) / total_size > print_percentage:
                print("{}%..".format(int(print_percentage * 100)), end="", flush=True)
                print_percentage += 0.1
        if verbose:
            print("100%! Done!")

    # Look for cached training data
    if use_cached_training_data:
        save_2d_numpy_array_if_exists(filepath, generated_data[0][:total_size], generated_data[1][:total_size])
    generated_data = shuffle(*generated_data)
    return (generated_data[0][:total_size], generated_data[1][:total_size]), False


def normalize_0_1(data_x_list):
    """ Scales everything from 0-255 with 0-1 normalization
    """
    ret = []
    for img in data_x_list:
        res = (img-img.min())/(img.max()-img.min())
        ret.append(res)
    return tuple(ret)

def load_cifar100_images():
    """ Loads 0-1 normalized CIFAR images
    """
    (cifar_x, cifar_y), (cifar_x_test, cifar_y_test) = cifar100.load_data()
    # Scale everything to 0-1
    cifar_x, cifar_x_test, = normalize_0_1([cifar_x, cifar_x_test])
    return (cifar_x, transform_to_one_hot(cifar_y, depth=100)), (cifar_x_test, transform_to_one_hot(cifar_y_test, depth=100))


def load_cifar_images():
    """ Loads 0-1 normalized CIFAR images
    """
    (cifar_x, cifar_y), (cifar_x_test, cifar_y_test) = cifar10.load_data()
    # Scale everything to 0-1
    cifar_x, cifar_x_test, = normalize_0_1([cifar_x, cifar_x_test])
    return (cifar_x, transform_to_one_hot(cifar_y, depth=10)), (cifar_x_test, transform_to_one_hot(cifar_y_test, depth=10))

def load_mnist_images():
    """ Loads 0-1-normalized MNIST images ranging from 0-1
    """
    (mnist_x, mnist_y), (mnist_x_test, mnist_y_test) = mnist.load_data()

    mnist_x, mnist_x_test = np.reshape(mnist_x, (-1, 28, 28, 1)), np.reshape(mnist_x_test, (-1, 28, 28, 1))

    # Scale everything to 0-1
    mnist_x, mnist_x_test, = normalize_0_1([mnist_x, mnist_x_test])

    return (mnist_x, transform_to_one_hot(mnist_y, depth=10)), (mnist_x_test, transform_to_one_hot(mnist_y_test, depth=10))


def load_emnist_images():
    """ Loads 0-1-normalized MNIST images ranging from 0-1
    """
    (mnist_x, mnist_y), (mnist_x_test, mnist_y_test) = mnist.load_data()
    emnist_x, emnist_y = extract_training_samples('digits')
    mnist_x, mnist_x_test, emnist_x = np.reshape(mnist_x, (-1, 28, 28, 1)), np.reshape(mnist_x_test, (-1, 28, 28, 1)), \
                                      np.reshape(emnist_x, (-1, 28, 28, 1))
    mnist_x = np.vstack((mnist_x, emnist_x))
    mnist_y = np.hstack((mnist_y, emnist_y))

    # Scale everything to 0-1
    mnist_x, mnist_x_test, = normalize_0_1([mnist_x, mnist_x_test])

    return (mnist_x, transform_to_one_hot(mnist_y, depth=10)), (mnist_x_test, transform_to_one_hot(mnist_y_test, depth=10))


def clean_files(paths):
    """ Deletes a file if it exists
    """
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def write_data(filename: str, data: tuple):
    """ Writes tuple output data to the data folder in the fs
    """
    outfile = "../data/" + filename
    np.savez(outfile, x_train=data[0], y_train=data[1])


def read_data(infile: str, one_hot=False):
    """ Reads a dataset from the data repository
    """
    with np.load("../data/" + infile + ".npz") as data:
        x_train = data['x_train']
        y_train = data['y_train']

        # Add empty dim for channel if it is not present
        if len(x_train.shape) <= 3:
            x_train = np.expand_dims(x_train, axis=-1)
        if one_hot:
            # Transform labels to one-hot encoding
            y_train = transform_to_one_hot(y_train, depth=10)

        return x_train, y_train


def transform_to_one_hot(labels: np.ndarray, depth: int) -> np.ndarray:
    """ Transforms an array of labels to its one-hot representation
    """
    b = np.zeros((len(labels), depth))
    for i, dp in enumerate(labels):
        b[i, dp] = 1
    return b

