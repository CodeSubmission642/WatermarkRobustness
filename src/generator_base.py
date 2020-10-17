import abc

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import multi_gpu_model

from src.util import generate_uniform_keys


class Generator_Base:
    __metaclass__ = abc.ABCMeta
    """ Setup for a GAN that generates conferred adversarial samples for a host network
        Note that the GAN only learns a minimal clipped delta that is added on top of the image
    """

    def __init__(self,
                 oracle_model=None,
                 stolen_models=None,
                 reference_models=None,
                 retrain_models=None,
                 optimizer=Adam(lr=0.0001, beta_1=0.5),
                 batch_size=64,
                 eps=0.3,
                 nb_class=10,
                 tb_logdir=None,
                 weight_path=None):
        """
        @:param oracle_model The model to extract the fingerprint from
        @:param stolen_models Stolen models from the oracle model
        @:param reference_models Random other models for the same tasks
        @:param retrain_models Models that are re-trained during training. Pass as dict with keys 'oracle_model',
        'stolen_models' and 'reference_models'
        @:param optimizer Custom optimizer
        @:param batch_size Batch size for the GAN
        @:param learning_rate
        @:param weight_path Save/Load path for the generator
        """
        self.nb_class = nb_class
        self.batch_size = batch_size
        self.eps = eps
        self.optimizer = optimizer
        self.weight_path = weight_path

        # Original models for prediction (left unchanged during training!)
        self.oracle_model = oracle_model
        self.reference_models = reference_models
        self.stolen_models = stolen_models

        # Models that are re-trained during training to increase generalizability of learner
        self.retrain_models = dict()
        if retrain_models is None:
            self.cl_oracle_model = oracle_model
            self.cl_reference_models = reference_models
            self.cl_stolen_models = stolen_models
        else:
            self.retrain_models = {
                "oracle_model": [retrain_models["oracle_model"]],
                "stolen_models": retrain_models['stolen_models'],
                "reference_models": retrain_models['reference_models']
            }
            self.cl_oracle_model = retrain_models['oracle_model']
            self.cl_stolen_models = retrain_models['stolen_models']
            self.cl_reference_models = retrain_models['reference_models']

        self.generator = self.build_generator()

        self.generator.compile(optimizer=self.optimizer)

        # Tensorboard logging
        self.train_summary_writer = None
        self.logdir = tb_logdir
        if tb_logdir:
            self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_logdir, histogram_freq=1)
            print("Initialized Tensorboard at {}".format(tb_logdir))
            self.merged, self.var_dict = self.initialize_tensorboard_graph()

        # Load cached weights
        if self.weight_path:
            try:
                self.generator.load_weights(self.weight_path)
                print("Loaded cached generator model!")
                return
            except Exception as e:
                print(e)

    def initialize_tensorboard_graph(self):
        """ Placeholders for variables that are logged through TensorBoard
        """
        names = ['val_rand', 'val_err', 'val_stolen', 'loss_img', 'loss_user', 'loss_random',
                 'loss_stolen', 'loss_total', 'val_gan']

        ph_dict = dict()
        for name in names:
            ph = tf.placeholder("float", shape=None, name=name)
            tf.summary.scalar(name, ph)
            ph_dict[name] = ph
        merged_op = tf.summary.merge_all()
        return merged_op, ph_dict

    @abc.abstractmethod
    def build_generator(self):
        """ Builds the generator
        """
        return

    @staticmethod
    def next_batch(x_train, y_train, batch_size, batch_idx):
        return x_train[batch_idx * batch_size:(batch_idx + 1) * batch_size], y_train[batch_idx * batch_size:(batch_idx + 1) * batch_size]

    def train(self,
              train_data,
              epochs,
              sess,
              show_images=True,
              validation=None,
              disc_train_rate=10,
              sample_stride=30):
        """
        @:param train_data The whole training data for the GAN (x_train, y_train), one_hot encoded labels
        @:param epochs How many epochs to train the generator for
        @:param sess The current TensorFlow session
        @:param validation (NN, x_test, y_test) @ToDo: Add list of NNs
        @:param disc_train_rate After how many iterations to retrain the discriminators in [self.retrain_models]
        @:param sample_stride After how many iterations to plot learned samples and log results to tensorboard
        Trains the generator on the given MNIST training data
        """
        self.train_summary_writer = tf.summary.FileWriter(self.logdir, sess.graph)

        x_train, y_train = train_data
        batch_size = min(len(x_train), self.batch_size)

        for e in range(epochs):
            for i in range(len(x_train) // batch_size):
                x_batch, y_batch = self.next_batch(x_train, y_train, batch_size, i)

                rand_des_classes = np.random.randint(0, self.nb_class, size=batch_size)
                true_classes = np.argmax(y_batch, 1)
                (true_match_idx,) = np.where(rand_des_classes == true_classes)
                for idx in true_match_idx:
                    rand_des_classes[idx] = (true_classes[idx] + np.random.randint(0, 9)) % self.nb_class
                y_des = to_categorical(rand_des_classes, num_classes=self.nb_class).reshape(-1, self.nb_class)

                batch_idx = (e * (len(x_train) // batch_size) + i)

                # Generate a sample of fake images
                x_fake = self.generator.predict([x_batch, y_batch, y_des])

                # Train the discriminators on a fake and real batch
                if batch_idx != 0 and (batch_idx % disc_train_rate) == 0:
                    for discriminators in self.retrain_models.values():
                        for discriminator in discriminators:
                            discriminator.train_on_batch(x=x_batch, y=y_batch)
                            discriminator.train_on_batch(x=x_fake, y=y_batch)

                # Train the generator to generate an adversarial sample
                loss_gan = self.generator.train_on_batch(x=[x_batch, y_batch, y_des], y=None)

                # Validate the test error
                if (batch_idx % sample_stride) == 0:
                    log_params = self.log_errors((x_batch, x_fake, y_batch, y_des), batch_idx, self.var_dict, validation)
                    summary = sess.run(self.merged, feed_dict=log_params)
                    self.train_summary_writer.add_summary(summary, batch_idx)
                    self.train_summary_writer.flush()

                print('epoch = %d/%d, batch = %d/%d, g_loss=%.3f' % (
                    e + 1, epochs, i, len(x_train) // batch_size, K.eval(tf.reduce_mean(loss_gan))),
                      100 * ' ',
                      end='\r')
                if show_images and ((e * (len(x_train) // batch_size) + i) % sample_stride) == 0:
                    self.plot_generator_samples(x_batch, y_batch, y_des)

        # Save the generator model
        if self.weight_path:
            self.generator.save_weights(self.weight_path)

    def gan_loss_func_maximum_confidence(self,
                                         x_true,
                                         x_pred,
                                         y_true,
                                         y_des):
        """ @:param y_true: The original image
            @:param y_pred: The generated fake image
            @:param y_true: The original label
            @:param y_des: The target label

            A loss function that tries to find maximum confidence adversarial examples
        """
        pred_oracle = self.cl_oracle_model(x_pred)
        preds_stolen = [stolen_model(x_pred) for stolen_model in self.cl_stolen_models]
        preds_random = [random_model(x_pred) for random_model in self.cl_reference_models]

        # Fool the oracle
        beta = 1
        loss_oracle = beta * tf.keras.backend.categorical_crossentropy(y_des, pred_oracle)

        # Fool the stolen models in the exact same way!
        gamma = 100
        loss_stolen = 0
        for pred_stolen in preds_stolen:
            loss_stolen += tf.keras.backend.categorical_crossentropy(y_des, pred_stolen)
        loss_stolen = gamma * loss_stolen / tf.cast(tf.shape(pred_stolen)[0], tf.float32)

        # Reference models should project to correct class
        omega = 100
        loss_random = 0
        for pred_random in preds_random:
            # Determine highest class except y_des
            """pred_random_no_des = pred_random - y_des * pred_random
            max_entry = tf.argmax(pred_random_no_des, 1)
            y_highest = tf.one_hot(max_entry, self.nb_class)"""
            loss_random += tf.keras.backend.categorical_crossentropy(y_true, pred_random)
        loss_random = omega * loss_random / tf.cast(tf.shape(pred_random)[0], tf.float32)

        # Make perturbation to image as small as possible
        alpha = 1  # Normalization factor
        eps = self.eps
        delta = tf.abs(x_true - tf.reshape(x_pred, tf.shape(x_true)))
        delta_errors = tf.nn.relu(delta - eps)
        loss_img = alpha * tf.norm(tf.reduce_sum(delta_errors, axis=-1), axis=(1, 2))

        total_loss = loss_oracle + loss_random + loss_stolen + loss_img

        return total_loss, tf.reduce_mean(loss_img), tf.reduce_mean(loss_oracle), \
               tf.reduce_mean(loss_stolen), tf.reduce_mean(loss_random)

    def log_errors(self,
                   batch_data,
                   batch_idx,
                   var_dict=None,
                   validation_data=None,
                   verbose=True):
        """
        @:param batch_data The data used for learning in the batch
        @:param validation_data Testing data and model
        Logs errors to show in TensorBoard
        """
        tb_log_dict = dict()

        val_nn, x_test, y_test = validation_data
        x_true, x_fake, y_true, y_des = batch_data

        # Predict with all models
        pred_val_nn = np.argmax(val_nn.predict(x_fake, verbose=0), 1)                   # Validation model
        pred_orig = np.argmax(self.oracle_model.predict(x_fake, verbose=0), 1)          # Oracle model
        pred_rand = np.argmax(self.reference_models[0].predict(x_fake, verbose=0), 1)   # Reference model
        pred_stolen = np.argmax(self.stolen_models[0].predict(x_fake, verbose=0), 1)    # Stolen model

        # Obtain loss and log to tensorboard
        total_loss, loss_img, loss_user, loss_stolen, loss_random = self.gan_loss_func_maximum_confidence(x_true, x_fake, y_true, y_des)
        total_loss, loss_img, loss_user, loss_stolen, loss_random = K.eval(tf.reduce_mean(total_loss)), K.eval(loss_img), K.eval(loss_user), K.eval(loss_stolen), K.eval(loss_random)
        tb_log_dict[var_dict["loss_total"]] = total_loss
        tb_log_dict[var_dict["loss_img"]] = loss_img
        tb_log_dict[var_dict["loss_user"]] = loss_user
        tb_log_dict[var_dict["loss_stolen"]] = loss_stolen
        tb_log_dict[var_dict["loss_random"]] = loss_random

        y_des_cl = np.argmax(y_des, 1)
        n = len(x_fake)

        # Validation Error is wm retention in random model
        (val_err_idx,) = np.where(pred_val_nn == y_des_cl)
        val_err = 100*len(val_err_idx) / n
        tb_log_dict[var_dict["val_err"]] = val_err

        # GAN success is how often the oracle model is fooled to y_des
        (gan_success,) = np.where(pred_orig == y_des_cl)
        val_gan = 100*len(gan_success) / n
        tb_log_dict[var_dict["val_gan"]] = val_gan

        # Random Error is how often the random model is fooled, i.e. wm retention in random training model
        (rand_err,) = np.where(pred_rand == y_des_cl)
        val_rand = 100*len(rand_err) / n
        tb_log_dict[var_dict["val_rand"]] = val_rand

        # Stolen Error is how often the stolen model is fooled, i.e. wm retention in stolen training model
        (stolen_err,) = np.where(pred_stolen == y_des_cl)
        val_stolen = 100*len(stolen_err) / n
        tb_log_dict[var_dict["val_stolen"]] = val_stolen

        if verbose:
            print()
            print("({}) Statistics: ".format(batch_idx))
            print("     + Random (Test)  WM Ret {:.2f}%".format(val_err))
            print("     + Random (Train) WM Ret {:.2f}%".format(val_rand))
            print("     + Oracle WM Ret         {:.2f}%".format(val_gan))
            print("     + Stolen (Train) WM Ret {:.2f}%".format(val_stolen))
            print("--------------------------------")
            print("Losses: ")
            print("Total: {:.3f}, Img: {:.3f}, User: {:.3f}, Stolen: {:.3f}, Random: {:.3f}".format(total_loss, loss_img, loss_user, loss_stolen, loss_random))

        return tb_log_dict

    def predict_random(self, data_x):
        """ Predicts for a target class
        """
        y_fake = generate_uniform_keys(len(data_x))
        x_fake = self.generator.predict([data_x, y_fake, y_fake])
        return x_fake, y_fake

    def gan_loss_func_impl(self,
                           x_true,
                           x_pred,
                           y_true,
                           y_des):
        """ @:param y_true: The original image
            @:param y_pred: The generated fake image
            @:param y_true: The original label
            @:param y_des: The target label

            An exemplary GAN loss function for fingerprinting
        """
        """ Extract predictions 
        """
        pred_user = self.oracle_model(x_pred)
        preds_stolen = [stolen_model(x_pred) for stolen_model in self.stolen_models]
        preds_random = [random_model(x_pred) for random_model in self.reference_models]

        # Limit perturbation norm to image
        thresh = self.eps  # Maximal threshold for L2 norm
        alpha = 1. / 20  # Normalization factor
        delta = tf.reshape(x_pred, tf.shape(x_true)) - x_true
        loss_img = alpha * tf.reduce_mean((tf.norm(delta, axis=(1, 2)) - thresh))

        # Penalize if image has values out of norm
        zeros = tf.zeros((tf.shape(x_pred)))
        loss_large_pixel = tf.reduce_mean(tf.maximum(zeros, tf.abs(x_pred) - 1) - tf.minimum(zeros, x_pred))

        # User prediction should be misclassification, no loss in that case
        real_user = tf.reduce_sum(y_des * pred_user, 1)
        other_user = tf.reduce_max((1 - y_des) * pred_user - y_des * 10000, 1)
        loss_user = tf.reduce_mean(tf.maximum(0.0, other_user - real_user))

        # Stolen prediction should be misclassification, no loss in that case
        loss_stolen = 0
        for pred_stolen in preds_stolen:
            real_stolen = tf.reduce_sum(y_des * pred_stolen, 1)
            other_stolen = tf.reduce_max((1 - y_des) * pred_stolen - y_des * 10000, 1)
            loss_stolen += tf.reduce_mean(tf.maximum(0.0, other_stolen - real_stolen))
        loss_stolen /= len(preds_stolen)  # Normalize

        # Random predicts true label, no loss in that case
        loss_random = 0
        for pred_random in preds_random:
            real_random = tf.reduce_sum(y_true * pred_random, 1)
            other_random = tf.reduce_max((1 - y_true) * pred_random - y_true * 10000, 1)
            loss_random += tf.reduce_mean(tf.maximum(0.0, other_random - real_random))
        loss_random /= len(preds_random)

        total_loss = tf.reduce_sum(loss_user + 3*loss_random + loss_stolen + loss_img + loss_large_pixel)

        return total_loss, loss_img, loss_user, loss_stolen, loss_random

    def gan_loss(self, x_true, x_pred, y_true, y_des):
        """
            @:param x_true The original image
            @:param x_pred The generated image
            @:param y_true The true image label
            @:param y_des The desired image label
        """
        return self.gan_loss_func_maximum_confidence(x_true, x_pred, y_true, y_des)[0]

    def plot_generator_samples(self, x_batch, y_batch, y_des):
        """ Plots ten samples with the prediction from the discriminator
        """

        plt_idx = np.random.randint(low=0, high=self.batch_size, size=10)
        x_fake = self.generator.predict([x_batch[plt_idx], y_batch[plt_idx], y_des[plt_idx]])

        y_true = np.argmax(y_batch[plt_idx], axis=1)
        y_pred_fake = np.argmax(self.oracle_model.predict(x_fake), axis=1)
        ref_pred = np.argmax(self.reference_models[0].predict(x_fake), axis=1)
        stolen_pred = np.argmax(self.stolen_models[0].predict(x_fake), axis=1)
        des_label = np.argmax(y_des[plt_idx], axis=1)

        plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
        for k in range(10):
            plt.subplot(2, 5, k + 1)
            if self.oracle_model:
                plt.title(
                    '({}, {}) - ({}, {}, {})'.format(y_true[k], ref_pred[k], y_pred_fake[k], stolen_pred[k], des_label[k]),
                    fontsize=12)
            plt.imshow(x_fake[k].squeeze(), cmap='gray')
            plt.xticks([])
            plt.yticks([])
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()


