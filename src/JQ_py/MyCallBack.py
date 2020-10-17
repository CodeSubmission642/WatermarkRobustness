import warnings

import keras
import numpy as np
import pylab
import matplotlib.pyplot as plt


class MyCallBack(keras.callbacks.Callback):
    wm_test_acc = []
    non_wm_test_acc = []
    acc = []
    val_acc = []
    wm_test = {}
    non_wm_test = {}

    def __init__(self, my_wm_test, my_non_wm_test, monitor='val_loss',save_best_only=True, mode='auto'):
        super().__init__()
        self.wm_test = my_wm_test
        self.non_wm_test = my_non_wm_test

    def on_train_begin(self, logs={}):
        self.wm_test_acc = []
        self.non_wm_test_acc = []
        self.acc = []

    def on_train_end(self, logs={}):
        plt.plot(self.acc, color='y', label='train accuracy')
        plt.plot(self.non_wm_test_acc, color='r', label='non watermark test acc')
        plt.plot(self.wm_test_acc, color='m', label='watermark test acc')
        plt.plot(self.val_acc, color='c', label='validation set accuracy')
        pylab.legend(loc='best')
        plt.show()
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        wm_X_test = self.wm_test['X']
        wm_y_test = self.wm_test['y']
        non_wm_X_test = self.non_wm_test['X']
        non_wm_y_test = self.non_wm_test['y']

        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))

        wm_y_pred = np.argmax(self.model.predict(wm_X_test), axis=1)
        self.wm_test_acc.append(np.sum(wm_y_pred == wm_y_test)/len(wm_y_test))

        non_wm_y_pred = np.argmax(self.model.predict(non_wm_X_test), axis=1)
        print(np.sum(wm_y_pred == wm_y_test))
        print(len(wm_y_pred))
        print(len(wm_y_test))
        self.non_wm_test_acc.append(np.sum(non_wm_y_pred == non_wm_y_test) / len(non_wm_y_test))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


class EarlyStoppingByLossVal(keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(keras.callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True