""" Main entry point for generating the graphs
    Just redirects to the respective files for generation and serves as top-level API
"""
import tensorflow as tf
from keras import backend as K
import numpy as np

from src.adversarial_main import adversarial_blackbox, zerobit_embed
from src.generators.mnist_generator import GAN_MNIST
from src.discriminators import get_lenet_model_for_mnist
from src.preprocess_data import load_emnist_images
from src.util import plot_blackbox


def gan_loss_func(y_true, y_pred, y_true_img, y_des_img):
    """pred_user = original_model(y_pred)
    pred_stolen = stolen_model(y_pred)
    pred_random = rnd_model(y_pred)"""

    y_des_img = tf.convert_to_tensor(y_des_img, np.float32)
    y_true_img = tf.convert_to_tensor(y_true_img, np.float32)

    # Predictions for stolen model equal but unequal for random model
    """loss_user = losses.binary_crossentropy(pred_user, y_true_img)
    loss_stolen = losses.binary_crossentropy(pred_stolen, y_true_img)
    loss_random = losses.binary_crossentropy(pred_random, y_true_img)"""

    # Related to original image
    loss_img = K.mean(K.abs(y_true - y_pred))

    # Idea: Move to areas with high uncertainty, make stolen loss as low as possible and
    # random loss as high as possible
    total_loss = loss_img #+ loss_img #+ pred_var

    return total_loss


def run_fingerprinting():
    sess = tf.Session()
    K.set_session(sess)

    split1, split2, split3, split4 = 15000, 30000, 45000, 60000

    (x_train, y_train), (x_test, y_test) = load_emnist_images()
    original_model = get_lenet_model_for_mnist()

    # Train the original model
    original_model.fit(x_train[:split1],
                       y_train[:split1],
                       batch_size=64,
                       epochs=1,
                       validation_data=(x_test, y_test),
                       verbose=1)

    train_batch_size = 128
    # Train the GAN
    gan = GAN_MNIST(gan_loss_func=gan_loss_func,
                    retrain_models=None,
                    oracle_model=original_model,
                    batch_size=train_batch_size)
    gan.train(train_data=(x_train, y_train),
              epochs=1,
              sample_stride=100)


def run_adversarial_mnist_tests():
    surr_model, all_history = adversarial_blackbox(
        load_dataset_func=load_emnist_images,  # Which dataset to choose. Should return training and testing data
        dataset_label="MNIST",  # Label of the dataset (for caching)
        load_wm_model_func=get_lenet_model_for_mnist,  # Model specification for wm_embedding
        wm_embed_func=zerobit_embed
    )

    plot_blackbox(all_history)


if __name__ == "__main__":
    run_fingerprinting()
