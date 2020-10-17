import math

import numpy as np
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras import Model
from tensorflow.python.keras.optimizers import Adam

from src.generators.cifar_generator import GAN_CIFAR
from src.generators.mnist_generator import GAN_MNIST
from src.util import intersection, generate_uniform_keys, plot_adversarial_samples


def setup_cifar_gan(oracle_model: Model,
                    stolen_models: list,
                    reference_models: list,
                    retrain_models=None,
                    optimizer=None,
                    batch_size=None,
                    eps=None,
                    tb_logdir=None,
                    weight_path=None
                    ):
    """
    @:param oracle_model The model to extract the fingerprint from
    @:param stolen_models Stolen from oracle
    @:param reference_models Random discriminators for same task
    @:param retrain_models Models to retrain during training
    @:param optimizer
    @:param learning_rate
    @:param batch_size
    @:param eps Maximum perturbation norm without error
    @:param tb_logdir Folder to log for TensorBoard
    @:param weight_path Path to generator weights
    @:param verbose Whether to print steps
    """
    gan = GAN_CIFAR(host_model=oracle_model,
                    reference_models=reference_models,
                    stolen_models=stolen_models,
                    optimizer=optimizer,
                    batch_size=batch_size,
                    eps=eps,
                    tb_logdir=tb_logdir,
                    weight_path=weight_path,
                    retrain_models=retrain_models)
    return gan


def setup_mnist_gan(host_model: Model,
                    stolen_models: list,
                    reference_models: list,
                    retrain_models=None,
                    optimizer=Adam(),
                    batch_size=128,
                    eps=0.5,
                    use_ssim=False,
                    tb_logdir=None,
                    weight_path=None
                    ):
    """
    @:param oracle_model The model to extract the fingerprint from
    @:param stolen_models Stolen from oracle
    @:param reference_models Random discriminators for same task
    @:param retrain_models Models to retrain during training
    @:param optimizer
    @:param learning_rate
    @:param batch_size
    @:param eps Maximum perturbation norm without error
    @:param tb_logdir Folder to log for TensorBoard
    @:param weight_path Path to generator weights
    @:param verbose Whether to print steps
    """
    gan = GAN_MNIST(host_model=host_model,
                    reference_models=reference_models,
                    stolen_models=stolen_models,
                    optimizer=optimizer,
                    batch_size=batch_size,
                    eps=eps,
                    tb_logdir=tb_logdir,
                    use_ssim=use_ssim,
                    weight_path=weight_path,
                    retrain_models=retrain_models)
    return gan


def train_gan(gan,
              x_train,
              y_train,
              sess,
              show_images=True,
              validation=None,
              disc_train_rate=50,
              sample_stride=50,
              n_batches=300):
    """ Trains the GAN
    @:param gan The generator
    @:param x_train The training image inputs
    @:param y_train The training labels
    @:param sess The current tensorflow session
    @:param show_images Whether to plot images during training
    @:param disc_train_rate How often to retrain the discriminators
    @:param sample_stride How often to output losses
    @:param n_batches How many batches to train for
    """
    # Adjust training length to number of batches
    if len(x_train) > n_batches * gan.batch_size:
        nl = int(n_batches * gan.batch_size)
        x_train, y_train = x_train[:nl], y_train[:nl]
        epochs = 1
    else:
        batches_per_epoch = len(x_train) // gan.batch_size
        epochs = math.ceil(n_batches / batches_per_epoch)

    gan.train(train_data=(x_train, y_train),
              validation=validation,
              epochs=epochs,
              sess=sess,
              show_images=show_images,
              disc_train_rate=disc_train_rate,
              sample_stride=sample_stride)
    print("Done training!")


def generate_keys_fair(gan,
                  x_train,
                  y_train,
                  original_model: Model,
                  stolen_models: list,
                  random_models: list,
                  trigger_samples_per_class=10,
                  verbose=False):
    """ @:param gan The generator network
        @:param x_train Starting point data to be perturbed
        @:param y_train True labels of that data
        @:param original_model The model to generate keys for
        @:param stolen_models The stolen discriminators of the original model
        @:param random_models Reference discriminators

    Takes the GAN and generates the 'best' keys that are least perturbed and yet differently classified
    """
    # Perturb all given data to each class
    print("Perturbing all data ..")
    print(x_train.shape)
    final_trigger_x, final_trigger_y, final_true_y = np.empty((trigger_samples_per_class * 10, 28, 28, 1)), np.empty(
        (trigger_samples_per_class * 10, 10)), np.empty((trigger_samples_per_class * 10, 10))
    print("Trigger_X Min {}, Max {}".format(final_trigger_x.min(), final_trigger_x.max()))
    for i in range(10):
        rand_des_classes = np.random.randint(i, i + 1, size=len(x_train))
        true_classes = np.argmax(y_train, 1)
        (true_match_idx,) = np.where(rand_des_classes != true_classes)

        # Define input data, perturbed labels and trigger
        x_train_batch, rand_des_classes = x_train[true_match_idx], rand_des_classes[true_match_idx]
        y_train_batch = y_train[true_match_idx]
        y_des = to_categorical(rand_des_classes, num_classes=10).reshape(-1, 10)
        trigger_x = gan.generator.predict([x_train_batch, y_des, y_des], verbose=1)
        # Obtain predictions
        pred_user = original_model.predict(trigger_x)
        preds_stolen = [stolen_model.predict(trigger_x) for stolen_model in stolen_models]
        preds_random = [random_model.predict(trigger_x) for random_model in random_models]

        # Sort according to perturbation size
        # deltas = np.sum(np.linalg.norm(x_train - trigger_x, axis=(1, 2)), axis=-1).reshape(-1)
        # sorted_idx = np.argsort(deltas)

        # Apply sorting
        # pred_user = pred_user[sorted_idx]
        # preds_stolen = [pred_stolen[sorted_idx] for pred_stolen in preds_stolen]
        # preds_random = [pred_random[sorted_idx] for pred_random in preds_random]

        # x_train_s, y_train_s = x_train[sorted_idx], y_train[sorted_idx]
        # trigger_x_s = trigger_x[sorted_idx]
        # y_des_s = y_des[sorted_idx]

        # Compute the success rate for original model
        mc_user = np.where(np.argmax(pred_user, 1) == np.argmax(y_des, 1))[0]
        mc_stolen = intersection([np.where(np.argmax(ps, 1) == np.argmax(y_des, 1))[0] for ps in preds_stolen])
        ccr_idx = intersection([np.where(np.argmax(pr, 1) != np.argmax(y_des, 1))[0] for pr in preds_random])

        trigger_idx = intersection([mc_user, mc_stolen, ccr_idx])
        trigger_x_success = trigger_x[trigger_idx]
        trigger_y_success = pred_user[trigger_idx]
        trigger_y_true_success = y_train_batch[trigger_idx]

        plot_adversarial_samples(np.vstack((trigger_x[:5], trigger_x[-5:])),
                                 original_model, random_models[0],
                                 msg="Top: Lowest Perturbation, Bottom: Highest Perturbation",
                                 use_random=False)

        print("Statistics class {}: ".format(i))
        print("    + Original FP Ret: {}%".format(100 * len(mc_user) / len(pred_user)))
        print("    + Stolen FP Ret: {}%".format(100 * len(mc_stolen) / len(pred_user)))
        print("    + Reference Non FP Ret: {}%".format(100 * len(ccr_idx) / len(pred_user)))
        print("    + Overall success: {}%".format(100 * len(trigger_x_success) / len(pred_user)))

        print("Trigger_X_Success Min {}, Max {}".format(trigger_x_success.min(), trigger_x_success.max()))

        final_trigger_x[i * trigger_samples_per_class:(i + 1) * trigger_samples_per_class] = trigger_x_success[
                                                                                             :trigger_samples_per_class]
        final_trigger_y[i * trigger_samples_per_class:(i + 1) * trigger_samples_per_class] = trigger_y_success[
                                                                                             :trigger_samples_per_class]
        final_true_y[i * trigger_samples_per_class:(i + 1) * trigger_samples_per_class] = trigger_y_true_success[
                                                                                          :trigger_samples_per_class]
        print("Trigger_X Min {}, Max {}".format(final_trigger_x.min(), final_trigger_x.max()))

    print("Trigger_X Min {}, Max {}".format(final_trigger_x.min(), final_trigger_x.max()))
    return final_trigger_x, final_trigger_y, final_true_y

def generate_keys(gan,
                  x_train,
                  y_train,
                  original_model: Model,
                  stolen_models: list,
                  random_models: list,
                  trigger_samples_per_class=10,
                  verbose=False):
    """ @:param gan The generator network
        @:param x_train Starting point data to be perturbed
        @:param y_train True labels of that data
        @:param original_model The model to generate keys for
        @:param stolen_models The stolen discriminators of the original model
        @:param random_models Reference discriminators

    Takes the GAN and generates the 'best' keys that are least perturbed and yet differently classified
    """
    # Perturb all given data
    print("Perturbing all data ..")

    rand_des_classes = np.random.randint(0, 10, size=len(x_train))
    true_classes = np.argmax(y_train, 1)
    (true_match_idx,) = np.where(rand_des_classes == true_classes)
    for idx in true_match_idx:
        rand_des_classes[idx] = (true_classes[idx] + np.random.randint(1, 9)) % 10
    y_des = to_categorical(rand_des_classes, num_classes=10).reshape(-1, 10)

    trigger_x = gan.generator.predict(
        [x_train, generate_uniform_keys(len(x_train)), y_des],
        verbose=1)

    if verbose:
        plot_adversarial_samples(trigger_x, original_model, random_models[0])

    print("Evaluating generated data ..")
    pred_user = original_model.predict(trigger_x)
    preds_stolen = [stolen_model.predict(trigger_x) for stolen_model in stolen_models]
    preds_random = [random_model.predict(trigger_x) for random_model in random_models]

    # Sort according to perturbation size
    deltas = np.sum(np.linalg.norm(x_train - trigger_x, axis=(1, 2)), axis=-1).reshape(-1)
    sorted_idx = np.argsort(deltas)

    # Apply sorting
    pred_user = pred_user[sorted_idx]
    preds_stolen = [pred_stolen[sorted_idx] for pred_stolen in preds_stolen]
    preds_random = [pred_random[sorted_idx] for pred_random in preds_random]

    x_train, y_train = x_train[sorted_idx], y_train[sorted_idx]
    trigger_x = trigger_x[sorted_idx]
    y_des = y_des[sorted_idx]

    # Compute the success rate for original model
    mc_user = np.where(np.argmax(pred_user, 1) == np.argmax(y_des, 1))[0]
    mc_stolen = intersection([np.where(np.argmax(ps, 1) == np.argmax(y_des, 1))[0] for ps in preds_stolen])
    ccr_idx = intersection([np.where(np.argmax(pr, 1) != np.argmax(y_des, 1))[0] for pr in preds_random])

    trigger_idx = intersection([mc_user, mc_stolen, ccr_idx])
    trigger_x, trigger_y = trigger_x[trigger_idx], pred_user[trigger_idx]

    print("Delta Lowest: {}, Highest: {}".format(deltas[sorted_idx[0]], deltas[sorted_idx[-1]]))
    plot_adversarial_samples(np.vstack((trigger_x[:5], trigger_x[-5:])), original_model, random_models[0],
                             msg="Top: Lowest Perturbation, Bottom: Highest Perturbation", use_random=False)

    print("Statistics: ")
    print("    + Original misclassified: {}%".format(100 * len(mc_user) / len(pred_user)))
    print("    + Stolen misclassified: {}%".format(100 * len(mc_stolen) / len(pred_user)))
    print("    + Correct misclassified: {}%".format(100 * len(ccr_idx) / len(pred_user)))
    print("    + Overall success: {}%".format(100 * len(trigger_x) / len(pred_user)))

    # Select diverse subsample
    def select_diverse_subsample(trigger_x, trigger_y, trigger_y_true, samples_per_class=10):
        counts = [0] * 10
        trigger_x_final, trigger_y_final, trigger_y_true_final = [], [], []
        for x, y, yt in zip(trigger_x, trigger_y, trigger_y_true):
            idx = np.argmax(yt)
            if counts[idx] < samples_per_class:
                trigger_x_final.append(x)
                trigger_y_final.append(y)
                trigger_y_true_final.append(yt)
                counts[idx] += 1
        return np.array(trigger_x_final), np.array(trigger_y_final), np.array(trigger_y_true_final)

    trigger_x, trigger_y, trigger_y_true = select_diverse_subsample(trigger_x, trigger_y, y_train,
                                                                    trigger_samples_per_class)
    return trigger_x, trigger_y, trigger_y_true


def generate_fingerprint(original_model: Model,
                         random_models: list,
                         stolen_models: list,
                         x_train,
                         y_train,
                         batches_gan=900,
                         epochs_random=2,
                         epochs_stolen=2,
                         raw_trigger_size=1000,
                         trigger_size_per_class=10):
    """
    @:param original_model Model to extract the fingerprint from
    @:param x_train The training dataset
    @:param y_train The testing dataset
    @:param batches_gan The number of batches the gan should train for
    @:param epochs_random The number of epochs to train the random model for
    @:param epochs_stolen The number of epochs to train the random model for
    @:param raw_trigger_size How many trigger examples to generate raw before selection
    @:param trigger_size_per_class How many examples to include per class in the trigger
    @:return trigger_x, trigger_y, y_true
    Convenience function to fully extract a fingerprint from a model.
    """
    # Train a random other model on the training data

    [random_model.fit(x_train, y_train, batch_size=64, epochs=epochs_random, verbose=True) for random_model in random_models]

    # Steal a model from the original model
    [stolen_model.fit(x_train, original_model.predict(x_train), batch_size=64, epochs=epochs_stolen, verbose=True) for stolen_model in stolen_models]

    gan = setup_mnist_gan(original_model,
                          stolen_models=stolen_models,
                          reference_models=random_models)
    train_gan(gan,
                    x_train,
                    y_train,
                    disc_train_rate=65,
                    sample_stride=50,
                    n_batches=batches_gan)

    trigger_x, trigger_y, trigger_y_true = generate_keys(gan,
                                                         x_train[:raw_trigger_size],
                                                         y_train[:raw_trigger_size],
                                                         original_model,
                                                         stolen_models,
                                                         random_models,
                                                         trigger_samples_per_class=trigger_size_per_class)

    return gan, trigger_x, trigger_y, trigger_y_true
