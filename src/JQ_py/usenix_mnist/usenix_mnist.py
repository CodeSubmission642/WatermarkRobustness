import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils
from scipy import ndimage, misc
import os
import numpy as np
import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense
from keras.datasets import mnist
import pylab
import matplotlib.pyplot as plt
from scipy import stats
import random


class MyCallBack(keras.callbacks.Callback):
    wm_test_acc = []
    non_wm_test_acc = []
    acc = []
    val_acc = []
    wm_test = {}
    non_wm_test = {}
    train = {}
    val = {}

    def __init__(self, my_wm_test, my_non_wm_test, my_train, my_val, monitor='val_loss',save_best_only=True, mode='auto'):
        super().__init__()
        self.wm_test = my_wm_test
        self.non_wm_test = my_non_wm_test
        self.train = my_train
        self.val = my_val

    def on_train_begin(self, logs={}):
        self.wm_test_acc = []
        self.non_wm_test_acc = []
        self.acc = []
        self.val_acc = []

    def on_train_end(self, logs={}):
        print('training accuracy:')
        print(self.acc)
        print('non-wm test accuracy:')
        print(self.non_wm_test_acc)
        print('wm test accuracy:')
        print(self.wm_test_acc)
        print('wm train accuracy:')
        print(self.val_acc)
        plt.plot(self.acc, color='y', label='train accuracy')
        plt.plot(self.non_wm_test_acc, color='r', label='non watermark test acc')
        plt.plot(self.wm_test_acc, color='m', label='watermark test acc')
        plt.plot(self.val_acc, color='c', label='validation set accuracy')
        pylab.legend(loc='best')
        plt.show()

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        X_train = self.train['X']
        y_train = self.train['y']
        wm_X_test = self.wm_test['X']
        wm_y_test = self.wm_test['y']
        non_wm_X_test = self.non_wm_test['X']
        non_wm_y_test = self.non_wm_test['y']
        val_X = self.val['X']
        val_y = self.val['y']

        self.acc.append(self.model.evaluate(X_train, y_train)[1])
        self.val_acc.append(self.model.evaluate(val_X, val_y)[1])
        self.non_wm_test_acc.append(self.model.evaluate(non_wm_X_test, non_wm_y_test)[1])
        self.wm_test_acc.append(self.model.evaluate(wm_X_test, wm_y_test)[1])
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


# generate a callback using raw data
def gen_callback(X_train, y_train,
                X_wm_test, y_wm_test,
                X_non_wm_test, y_non_wm_test,
                X_val, y_val):
    p1 = preprocess_training_set(X_train, y_train)
    X_train_ = p1[0]
    y_train_ = p1[1]
    p2 = preprocess_training_set(X_val, y_val)
    X_val_ = p2[0]
    y_val_ = p2[1]
    p3 = preprocess_training_set(X_wm_test, y_wm_test)
    X_wm_test_ = p3[0]
    y_wm_test_ = p3[1]
    p4 = preprocess_training_set(X_non_wm_test, y_non_wm_test)
    X_non_wm_test_ = p4[0]
    y_non_wm_test_ = p4[1]
    callback = MyCallBack(my_wm_test={'X': X_wm_test_, 'y': y_wm_test_},
                          my_non_wm_test={'X': X_non_wm_test_, 'y': y_non_wm_test_},
                          my_train={'X': X_train_, 'y': y_train_},
                          my_val={'X': X_val_, 'y': y_val_})
    return callback


def preprocess_training_set(x_train, y_train):
    trainX = np.reshape(x_train, [x_train.shape[0], x_train.shape[1]*x_train.shape[2]])
    trainX_normal = (trainX/255.)
    trainX_normal = trainX_normal.astype('float32')
    y_train = np_utils.to_categorical(y_train, 10)
    return {0:trainX_normal, 1:y_train}

def get_model():
    model = Sequential()
    model.add(Dense(512, activation='relu', use_bias=True, input_shape=(784,)))
    model.add(Dense(128, activation='relu', use_bias=True))
    model.add(Dense(10, activation='softmax', use_bias=True))
    model.compile(loss='mean_squared_error',
                  optimizer='Adam',
                  metrics=['accuracy'])
    return model


def train_model_with_cb(X_train, y_train,
                X_wm_test, y_wm_test,
                X_non_wm_test, y_non_wm_test,
                X_val, y_val,
                name, epochs, callback):
    p1 = preprocess_training_set(X_train, y_train)
    X_train_ = p1[0]
    y_train_ = p1[1]
    p2 = preprocess_training_set(X_val, y_val)
    X_val_ = p2[0]
    y_val_ = p2[1]
    p3 = preprocess_training_set(X_wm_test, y_wm_test)
    X_wm_test_ = p3[0]
    y_wm_test_ = p3[1]
    p4 = preprocess_training_set(X_non_wm_test, y_non_wm_test)
    X_non_wm_test_ = p4[0]
    y_non_wm_test_ = p4[1]
    model = get_model()
    model.fit(X_train_, y_train_, batch_size=32, epochs=epochs,
              verbose=1, validation_data=(X_val_, y_val_),
              callbacks=[callback])
    model.save_weights(name+'.h5')
    return model


def train_model(X_train, y_train,
                X_wm_test, y_wm_test,
                X_non_wm_test, y_non_wm_test,
                X_val, y_val,
                name, epochs):
    p1 = preprocess_training_set(X_train, y_train)
    X_train_ = p1[0]
    y_train_ = p1[1]
    p2 = preprocess_training_set(X_val, y_val)
    X_val_ = p2[0]
    y_val_ = p2[1]
    p3 = preprocess_training_set(X_wm_test, y_wm_test)
    X_wm_test_ = p3[0]
    y_wm_test_ = p3[1]
    p4 = preprocess_training_set(X_non_wm_test, y_non_wm_test)
    X_non_wm_test_ = p4[0]
    y_non_wm_test_ = p4[1]
    callback = MyCallBack(my_wm_test={'X': X_wm_test_, 'y': y_wm_test_},
                          my_non_wm_test={'X': X_non_wm_test_, 'y': y_non_wm_test_},
                          my_train={'X': X_train_, 'y': y_train_},
                          my_val={'X': X_val_, 'y': y_val_})
    model = get_model()
    model.fit(X_train_, y_train_, batch_size=32, epochs=epochs,
              verbose=1, validation_data=(X_val_, y_val_),
              callbacks=[callback])
    model.save_weights(name + '.h5')
    return model


def get_M(num_epochs, name):
    X_train = np.load('usenix_X_train.npy')
    y_train = np.load('usenix_y_train.npy')
    wm_X_test = np.load('usenix_X_test.npy')
    wm_y_test = np.load('usenix_y_test.npy')
    X_test = np.load('mnist_X_test.npy')
    y_test = np.load('mnist_y_test.npy')
    return train_model(X_train, y_train, wm_X_test, wm_y_test, X_test, y_test, wm_X_test, wm_y_test, 'name', num_epochs)


def linear_fit_y(x, y):
    m,b = np.polyfit(x, y, 1)
    return [m*a+b for a in x]

def plot_line(x, y, color):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    if color == 1:
        plt.plot(x, linear_fit_y(x, y), 'r', label='y={:.5f}x+{:.5f}'.format(slope, intercept))
    if color == 2:
        plt.plot(x, linear_fit_y(x, y), 'm', label='y={:.5f}x+{:.5f}'.format(slope, intercept))
    if color == 3:
        plt.plot(x, linear_fit_y(x, y), 'c', label='y={:.5f}x+{:.5f}'.format(slope, intercept))

def add_lst(lst1, lst2):
    return [sum(x) for x in zip(lst1, lst2)]


if __name__ == '__main__':
    num_it = 10
    x_range = 30

    X_train = np.load('usenix_X_30000.npy')
    y_train = np.load('usenix_labels_30000.npy')
    wm_X_test = np.load('usenix_X_test.npy')
    wm_y_test = np.load('usenix_y_test.npy')
    X_test = np.load('mnist_X_test.npy')
    y_test = np.load('mnist_y_test.npy')
    X_val = np.load('mnist_wm_X.npy')
    y_val = np.load('mnist_wm_y.npy')

    callback_lst = [gen_callback(X_train, np.argmax(y_train, axis=1),
                                 wm_X_test, wm_y_test,
                                 X_test, y_test,
                                 X_val, y_val)] * num_it

    for i in range(0, num_it):
        train_model_with_cb(X_train, np.argmax(y_train, axis=1),
                            wm_X_test, wm_y_test,
                            X_test, y_test,
                            wm_X_test, wm_y_test,
                            'try', x_range, callback_lst[i])

    wm_test_acc = [0] * x_range
    non_wm_test_acc = [0] * x_range
    acc = [0] * x_range
    val_acc = [0] * x_range

    for i in callback_lst:
        wm_test_acc = add_lst(wm_test_acc, i.wm_test_acc)
        non_wm_test_acc = add_lst(non_wm_test_acc, i.non_wm_test_acc)
        acc = add_lst(acc, i.acc)
        val_acc = add_lst(val_acc, i.val_acc)

    wm_test_acc = [x / num_it for x in wm_test_acc]
    non_wm_test_acc = [x / num_it for x in non_wm_test_acc]
    acc = [x / num_it for x in acc]
    val_acc = [x / num_it for x in val_acc]

    x_range = list(range(0, x_range))

    plt.plot(acc, color='y', label='train accuracy')
    plt.plot(non_wm_test_acc, color='r', label='non watermark test acc')
    plt.plot(wm_test_acc, color='m', label='watermark test acc')
    plt.plot(val_acc, color='c', label='watermark retention')

    plot_line(x_range, wm_test_acc, 2)
    plot_line(x_range, val_acc, 3)

    pylab.legend(loc='best')

    plt.plot(acc, 'yo')
    plt.plot(non_wm_test_acc, 'ro')
    plt.plot(wm_test_acc, 'mo')
    plt.plot(val_acc, 'co')

    plt.show()
