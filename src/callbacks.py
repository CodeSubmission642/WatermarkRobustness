import time
import warnings
import tensorflow.keras
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback

from src.util import plot_4d_image, plot_confusion_matrix


class ShowErrorsCallback(Callback):
    def __init__(self, dataset, prefix="", plot_one_example=False):
        super(Callback, self).__init__()
        self.dataset = dataset
        self.prefix = prefix
        self.plot_one_example = plot_one_example

    def on_epoch_end(self, epoch, logs={}):
        predictions = np.argmax(self.model.predict(self.dataset[0]), axis=1)
        true_labels = np.argmax(self.dataset[1], axis=1)

        incorrect_indices = np.nonzero(predictions != true_labels)[0]
        if len(incorrect_indices) > 0:
            plot_confusion_matrix(true_labels, predictions, title=self.prefix)

        # Plot 1 example!
        if self.plot_one_example:
            if len(incorrect_indices) > 0:
                rnd = random.randint(0, len(incorrect_indices))
                actual_index = incorrect_indices[rnd]
                img = self.dataset[0][actual_index]
                img = img[np.newaxis, :]
                plt.title("Prediction: {}, True label: {} ".format(
                    np.argmax(predictions[actual_index]),
                    np.argmax(self.dataset[1][actual_index])))
                plot_4d_image(img, plt)
                plt.show()


class EarlyStoppingByWatermarkRet(Callback):
    def __init__(self,
                 monitor='watermark_val',
                 value=0.1,
                 patience=2,
                 stop_when_smaller=True,
                 verbose=True):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.stop_when_smaller = stop_when_smaller
        self.verbose = verbose
        self.patience = patience

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                "Early stopping requires %s available!" % self.monitor,
                RuntimeWarning)
        else:
            if (self.stop_when_smaller and self.value > current[-1]) or (
                    not self.stop_when_smaller and current[-1] > self.value):
                print("(Early stopping) Current: {} My value: {}".format(
                    current, self.value))
                if self.patience == 0:
                    print("Epoch %05d: early stopping THR" % epoch)
                    self.model.stop_training = True
                else:
                    self.patience -= 1
                    print("Early Stopping By WM: Patience decreased to {}".
                          format(self.patience))


class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        if not hasattr(self, 'times'):
            self.times = []
        if not hasattr(self, 'history'):
            self.history = {}
        self.training_time_start = time.time()

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        times = time.time() - self.epoch_time_start
        self.history.setdefault("time", []).append(times)
        self.history.setdefault(
            "time_total", []).append(time.time() - self.training_time_start)
        print("=> {}: {}".format("Time: ", times))


class AdditionalValidationSets(Callback):
    def __init__(self, validation_sets, verbose=0, batch_size=None):
        super(AdditionalValidationSets, self).__init__()
        self.validation_sets = validation_sets
        for validation_set in self.validation_sets:
            if len(validation_set) not in [2, 3]:
                raise ValueError()
        self.verbose = verbose
        self.batch_size = batch_size

    def on_train_batch_begin(self, *args, **kwargs):
        pass

    def on_train_batch_end(self, *args, **kwargs):
        pass

    def on_test_batch_begin(self, *args, **kwargs):
        pass

    def on_test_batch_end(self, *args, **kwargs):
        pass

    def on_test_begin(self, *args, **kwargs ):
        pass

    def on_test_end(self, *args, **kwargs):
        pass

    def on_train_begin(self, logs=None):
        if not hasattr(self, 'epoch'):
            self.epoch = []
        if not hasattr(self, 'history'):
            self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        is_cluster = False
        trigger = {}
        # evaluate on the additional validation sets
        for validation_set in self.validation_sets:
            if len(validation_set) == 2:
                is_cluster = True
                trigger, validation_set_name = validation_set
            elif len(validation_set) == 3:
                validation_data, validation_targets, validation_set_name = validation_set
                sample_weights = None
            elif len(validation_set) == 4:
                validation_data, validation_targets, sample_weights, validation_set_name = validation_set
            else:
                raise ValueError()

            if is_cluster:
                print(type(trigger))
                print(validation_set_name)
                preds = np.argmax(self.model.predict(trigger["keys"][0]),
                                  axis=1)
                decoded_bits = np.isin(preds,
                                       trigger["clusters"][0]).astype('int')
                ham_dist = np.sum(trigger["clusters"][2] != decoded_bits)
                results = [0, ham_dist / len(trigger["clusters"][2])]
            else:
                results = self.model.evaluate(x=validation_data,
                                              y=validation_targets,
                                              verbose=self.verbose,
                                              sample_weight=sample_weights,
                                              batch_size=self.batch_size)

            for i, result in enumerate(results):
                if i == 0:
                    valuename = validation_set_name + '_loss'
                else:
                    valuename = validation_set_name + '_val'
                    print()
                    print("=> {}: {:.2f}".format(valuename, result))
                self.history.setdefault(valuename, []).append(result)
                logs.setdefault(valuename, []).append(result)


class MNISTSequence(tensorflow.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.origin_x, self.origin_y = np.copy(x_set), np.copy(y_set)
        self.epoch = 0
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y

    def on_epoch_end(self):
        self.x = self.origin_x
        self.y = self.origin_y
        attacker_data_size = self.x.shape[0]
        random_selection = np.random.random_sample(attacker_data_size)
        random_selection = (random_selection < 0.05).astype('int64')
        random_target = np.random.randint(10, size=sum(random_selection))
        random_index = np.where(random_selection == 1)[0]
        self.y[random_index] = tensorflow.keras.utils.to_categorical(random_target, 10)
        print(sum(random_selection), " attacker data is twisted...")
        self.epoch += 1