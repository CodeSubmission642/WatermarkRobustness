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
import keras
from keras.models import Sequential
from keras import models, layers
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

total_count_train = 60000
watermark_color = 255
image_width = 28
image_height = 28
wm_width = 10
wm_height = 10
gname = 'graphs/'
lenet = False

(X_train, y_train), (X_test, y_test) = mnist.load_data()


def gen_wm(width, height):
    watermark_coords = {}
    width_lst = list(range(0, width))
    for i in range(0, height):
        watermark_coords[i] = random.sample(width_lst, random.randint(1, width))
    return watermark_coords


def count_result(y):
    count = [0] * 10
    for i in range(0, 10):
        count[i] = sum(1 for j in y if j == i)
    return count


def plt_result(y, name):
    np.save(name, y)
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    plt.figure()
    plt.pie(y, labels=labels, autopct='%.2f')
    plt.savefig(name)
    plt.clf()


class MyCallBack(keras.callbacks.Callback):
    wm_test_acc = []
    non_wm_test_acc = []
    acc = []
    val_acc = []
    wm_test = {}
    non_wm_test = {}
    train = {}
    val = {}

    def __init__(self, my_wm_test, my_non_wm_test, my_train, my_val, monitor='val_acc', save_best_only=True,
                 mode='auto'):
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
        # plt.plot(self.acc, color='y', label='train accuracy')
        # plt.plot(self.non_wm_test_acc, color='r', label='non watermark test acc')
        # plt.plot(self.wm_test_acc, color='m', label='watermark test acc')
        # plt.plot(self.val_acc, color='c', label='validation set accuracy')
        # pylab.legend(loc='best')
        # plt.show()

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

        # wm_y_pred = np.argmax(self.model.predict(wm_X_test), axis=1)
        # self.wm_test_acc.append(np.sum(wm_y_pred == wm_y_test)/len(wm_y_test))
        #
        # non_wm_y_pred = np.argmax(self.model.predict(non_wm_X_test), axis=1)
        # print(np.sum(non_wm_y_pred == non_wm_y_test))

        self.non_wm_test_acc.append(self.model.evaluate(non_wm_X_test, non_wm_y_test)[1])

        # self.non_wm_test_acc.append(np.sum(non_wm_y_pred == non_wm_y_test) / len(non_wm_y_test))
        self.wm_test_acc.append(self.model.evaluate(wm_X_test, wm_y_test)[1])
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def count_each_class(y, name):
    count = [0] * 10
    for i in range(0, 10):
        count[i] = sum(1 for j in y if j == i)
    print("Numbers of elements in each class are:")
    print(count)
    print("Portions of each class are:")
    result = [x / sum(count) for x in count]
    np.save(name, result)
    print(result)
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    plt.figure()
    plt.pie(count, labels=labels, autopct='%.3f')
    plt.savefig(name)
    plt.clf()
    return result


def add_wm(img, add):
    h_shift = random.randint(0, image_width - wm_width)
    v_shift = random.randint(0, image_height - wm_height)
    new_img = np.empty([28, 28])
    for i in range(0, image_width):
        for j in range(0, 28):
            new_img[i][j] = img[i][j]
    if not add:
        return new_img
    for i in watermark_coords:
        for j in watermark_coords[i]:
            new_img[i + v_shift][j + h_shift] = watermark_color
    return new_img


def preprocess_training_set(x_train, y_train, lenet=lenet):
    if lenet:
        return {0: ((x_train / 255).reshape(x_train.shape[0], 28, 28, 1)).astype('float32'),
                1: np_utils.to_categorical(y_train, 10).astype('float32')}
    trainX = np.reshape(x_train, [x_train.shape[0], x_train.shape[1] * x_train.shape[2]])
    trainX_normal = (trainX / 255.)
    trainX_normal = trainX_normal.astype('float32')
    y_train = np_utils.to_categorical(y_train, 10)
    return {0: trainX_normal, 1: y_train}


def get_model(lenet=lenet):
    if lenet:
        #         model = Sequential()
        #         model.add(Conv2D(20,(5,5), activation='relu', border_mode='same',input_shape=(32, 32, 1)))
        #         model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        #         model.add(Conv2D(50, (5, 5), activation='relu', border_mode='same'))
        #         model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
        #         model.add(Flatten())
        #         model.add(Dense(500,activation='relu', name='fc1'))
        #         model.add(Dense(84, activation='relu', name='fc2'))
        #         model.add(Dense(10, activation='softmax', name='fc3'))
        #     #plot_model(model, to_file=experiment_name + '.png', show_shapes=True, show_layer_names=True)
        #         model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-5), metrics=['accuracy'])

        model = Sequential()

        # C1 Convolutional Layer
        model.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(28, 28, 1),
                                padding='same'))

        # S2 Pooling Layer
        model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))

        # C3 Convolutional Layer
        model.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))

        # S4 Pooling Layer
        model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        # C5 Fully Connected Convolutional Layer
        model.add(layers.Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        # Flatten the CNN output so that we can connect it with fully connected layers
        model.add(layers.Flatten())

        # FC6 Fully Connected Layer
        model.add(layers.Dense(84, activation='tanh'))

        # Output Layer with softmax activation
        model.add(layers.Dense(10, activation='softmax'))
        # model.summary()

        # Compile the model
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer='SGD', metrics=['accuracy'])

        return model

    model = Sequential()
    model.add(Dense(512, activation='relu', use_bias=True, input_shape=(784,)))
    model.add(Dense(128, activation='relu', use_bias=True))
    model.add(Dense(10, activation='softmax', use_bias=True))
    mysgd = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(loss='mean_squared_error',
                  optimizer='Adam',
                  metrics=['accuracy'])
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
    filepath = name + '.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    model = get_model()
    model.fit(X_train_, y_train_, batch_size=32, epochs=epochs,
              verbose=1, validation_data=(X_val_, y_val_),
              callbacks=[callback,checkpoint])
    model.load_weights(name + '.h5')
    return model


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
    model.save_weights(name + '.h5')
    return model


def linear_fit_y(x, y):
    m, b = np.polyfit(x, y, 1)
    return [m * a + b for a in x]


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


def pad(x):
    return x
    x = np.expand_dims(x, axis=np.ndim(x))
    x = np.pad(x, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    return x


def preprocess_wm(wm, dname, lenet=lenet):
    wm_ytrain = np.empty(y_train.shape)

    all_wm_Xtrain = np.empty((6000, 28, 28))
    all_wm_ytrain = np.empty(6000)

    non_wm_Xtrain_1 = np.empty((27000, 28, 28))
    non_wm_ytrain_1 = np.empty(27000)

    non_wm_Xtrain_2 = np.empty((27000, 28, 28))
    non_wm_ytrain_2 = np.empty(27000)

    wm_count = [600] * 10
    # all training set shuffle
    shuffle_idx = np.arange(60000)
    np.random.shuffle(shuffle_idx)
    # watermark training set shuffle
    wm_shuffle_idx = np.arange(6000)
    np.random.shuffle(wm_shuffle_idx)
    # non-watermark training set shuffle
    non_wm_shuffle_idx = np.arange(54000)
    np.random.shuffle(non_wm_shuffle_idx)

    split_non_wm_set = np.hstack((([0] * 27000), ([1] * 27000)))
    np.random.shuffle(split_non_wm_set)

    non_wm_count = 0
    split_count = 0
    for i in range(0, total_count_train):
        j = shuffle_idx[i]
        img = np.empty([image_width, image_height])
        img_old = X_train[i]
        if wm_count[y_train[i]] > 0:
            img = add_wm(img_old, True)
            wm_ytrain[j] = wm
            k = wm_shuffle_idx[sum(wm_count) - 1]
            all_wm_Xtrain[k] = np.copy(img)
            all_wm_ytrain[k] = wm
            wm_count[y_train[i]] -= 1
        else:
            img = add_wm(img_old, False)
            wm_ytrain[j] = y_train[i]

            k = non_wm_shuffle_idx[non_wm_count]
            non_wm_count += 1
            if split_count < 27000:
                non_wm_Xtrain_1[split_count] = np.copy(img)
                non_wm_ytrain_1[split_count] = y_train[i]
            else:
                non_wm_Xtrain_2[split_count - 27000] = np.copy(img)
                non_wm_ytrain_2[split_count - 27000] = y_train[i]
            split_count += 1

        # wm_Xtrain[j] = np.copy(img)

    all_wm_Xtrain = np.array(all_wm_Xtrain)
    all_wm_ytrain = np.array(all_wm_ytrain)
    all_wm_ytrain = all_wm_ytrain.astype(int)

    non_wm_Xtrain_1 = np.array(non_wm_Xtrain_1)
    non_wm_ytrain_1 = np.array(non_wm_ytrain_1)
    non_wm_ytrain_1 = non_wm_ytrain_1.astype(int)

    non_wm_Xtrain_2 = np.array(non_wm_Xtrain_2)
    non_wm_ytrain_2 = np.array(non_wm_ytrain_2)
    non_wm_ytrain_2 = non_wm_ytrain_2.astype(int)

    wm_Xtrain = np.vstack((all_wm_Xtrain, non_wm_Xtrain_1))
    wm_ytrain = np.hstack((all_wm_ytrain, non_wm_ytrain_1))
    wm_ytrain = wm_ytrain.astype(int)

    X_test_ = [add_wm(x, True) for x in X_test]
    y_test_ = [wm for x in y_test]

    if lenet:
        np.save(dname + 'wm_Xtrain_33000_' + str(wm), pad(wm_Xtrain))
        np.save(dname + 'wm_ytrain_33000_' + str(wm), wm_ytrain)
        np.save(dname + 'all_wm_Xtrain_6000_' + str(wm), pad(all_wm_Xtrain))
        np.save(dname + 'all_wm_ytrain_6000_' + str(wm), all_wm_ytrain)
        np.save(dname + 'non_wm_Xtrain_27000_' + str(wm), pad(non_wm_Xtrain_2))
        np.save(dname + 'non_wm_ytrain_27000_' + str(wm), non_wm_ytrain_2)
        np.save(dname + 'wm_Xtest_10000', pad(X_test_))
    else:
        np.save(dname + 'wm_Xtrain_33000_' + str(wm), wm_Xtrain)
        np.save(dname + 'wm_ytrain_33000_' + str(wm), wm_ytrain)
        np.save(dname + 'all_wm_Xtrain_6000_' + str(wm), all_wm_Xtrain)
        np.save(dname + 'all_wm_ytrain_6000_' + str(wm), all_wm_ytrain)
        np.save(dname + 'non_wm_Xtrain_27000_' + str(wm), non_wm_Xtrain_2)
        np.save(dname + 'non_wm_ytrain_27000_' + str(wm), non_wm_ytrain_2)
        np.save(dname + 'wm_Xtest_10000', X_test_)

    y_test_ = [wm for x in y_test]
    np.save(dname + 'wm_ytest_10000_' + str(wm), y_test_)
    plt.imshow(X_test_[0])
    plt.savefig(dname + 'test_img_0')
    plt.clf()


def get_labels(wm, epochs=30, dname=''):
    X_train = np.load(dname + 'wm_Xtrain_33000_' + str(wm) + '.npy')
    y_train = np.load(dname + 'wm_ytrain_33000_' + str(wm) + '.npy')
    wm_X_test = np.load(dname + 'wm_Xtest_10000.npy')
    wm_y_test = np.load(dname + 'wm_ytest_10000_' + str(wm) + '.npy')
    X_test = np.load(dname + 'non_wm_Xtrain_27000_' + str(wm) + '.npy')
    y_test = np.load(dname + 'non_wm_ytrain_27000_' + str(wm) + '.npy')
    # X_val = np.load(dname + 'all_wm_Xtrain_6000_' + str(wm) + '.npy')
    #     # y_val = np.load(dname + 'all_wm_ytrain_6000_' + str(wm) + '.npy')
    X_val = np.vstack((wm_X_test, np.load('non_wm_Xtest_10000.npy')))
    y_val = np.hstack((wm_y_test, np.load('non_wm_ytest_10000.npy')))
    m = train_model(X_train, y_train, wm_X_test, wm_y_test, X_test, y_test, X_val, y_val, dname + 'M', epochs)
    p = preprocess_training_set(X_test, y_test)
    labels = m.predict(p[0])
    np.save(dname + 'labels_27000_' + str(wm), labels)


def train_surrogate(wm, num_it=10, x_range=30, gname='graphs/', aname='accs/', dname=''):
    X_train = np.load(dname + 'non_wm_Xtrain_27000_' + str(wm) + '.npy')
    y_train = np.load(dname + 'labels_27000_' + str(wm) + '.npy')
    wm_X_test = np.load(dname + 'wm_Xtest_10000.npy')
    wm_y_test = np.load(dname + 'wm_ytest_10000_' + str(wm) + '.npy')
    X_test = np.load('non_wm_Xtest_10000.npy')
    y_test = np.load('non_wm_ytest_10000.npy')
    X_val = np.load(dname + 'all_wm_Xtrain_6000_' + str(wm) + '.npy')
    y_val = np.load(dname + 'all_wm_ytrain_6000_' + str(wm) + '.npy')
    plt.imshow(X_test[0])
    plt.savefig('test_img_1')
    plt.clf()

    callback_lst = [gen_callback(X_train, np.argmax(y_train, axis=1),
                                 wm_X_test, wm_y_test,
                                 X_test, y_test,
                                 X_val, y_val)] * num_it

    result_lst = np.zeros(len(wm_y_test))

    for i in range(0, num_it):
        m = train_model_with_cb(X_train, np.argmax(y_train, axis=1),
                                wm_X_test, wm_y_test,
                                X_test, y_test,
                                wm_X_test, wm_y_test,
                                dname + str(wm), x_range, callback_lst[i])
        p = preprocess_training_set(wm_X_test, wm_y_test)
        print(m.evaluate(p[0], p[1]))
        result_lst = add_lst(result_lst, count_result(np.argmax(m.predict(p[0]), axis=1)))
        # result_lst = add_lst(result_lst, np.argmax(m.predict(p[0]), axis=1))
    # retval = count_each_class(result_lst, gname + str(wm) + '_2')
    retval = plt_result(result_lst, gname + str(wm) + '_2')

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

    np.save(aname + 'train_acc', acc)
    np.save(aname + 'non_wm_test_acc', non_wm_test_acc)
    np.save(aname + 'wm_test_acc', wm_test_acc)
    np.save(aname + 'watermark_retention_acc', val_acc)
    plt.title('random signature watermark class ' + str(wm))

    plot_line(x_range, wm_test_acc, 2)
    plot_line(x_range, val_acc, 3)

    pylab.legend(loc='best')

    plt.plot(acc, 'yo')
    plt.plot(non_wm_test_acc, 'ro')
    plt.plot(wm_test_acc, 'mo')
    plt.plot(val_acc, 'co')

    plt.savefig(gname + str(wm) + '_1')
    plt.clf()
    return result_lst


if __name__ == '__main__':
    np.save('non_wm_Xtest_10000', X_test)
    np.save('non_wm_ytest_10000', y_test)

    wm_ = 10
    x_range = 30
    class_count = [0] * 10
    epochs = 30
    num_it = 10

    for wm in range(0, wm_):
        dname_ = str(wm) + '/'
        dname_1 = str(wm) + '/graphs'
        dname_2 = str(wm) + '/accs'
        try:
            os.mkdir(dname_)
            os.mkdir(dname_1)
            os.mkdir(dname_2)
        except OSError:
            print("Creation of the directory %s failed" % str(wm))
        else:
            print("Successfully created the directory %s " % str(wm))

        watermark_coords = gen_wm(wm_width, wm_height)
        preprocess_wm(wm, dname=dname_)
        get_labels(wm, epochs=epochs, dname=dname_)
        class_count = add_lst(class_count,
                              train_surrogate(wm, num_it=num_it, x_range=x_range, gname=str(wm) + '/graphs/',
                                              aname=str(wm) + '/accs/', dname=dname_))
    np.save('class_count', class_count)
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    plt.figure()
    plt.pie(class_count, labels=labels, autopct='%.3f')
    plt.savefig('class_count')
    plt.clf()

    train_acc = np.load('0/accs/train_acc.npy')
    non_wm_test_acc = np.load('0/accs/non_wm_test_acc.npy')
    wm_test_acc = np.load('0/accs/wm_test_acc.npy')
    watermark_retention_acc = np.load('0/accs/watermark_retention_acc.npy')
    for i in range(1, wm_):
        aname = '/accs/'
        train_acc = add_lst(train_acc, np.load(str(i) + aname + 'train_acc.npy'))
        non_wm_test_acc = add_lst(non_wm_test_acc, np.load(str(i) + aname + 'non_wm_test_acc.npy'))
        wm_test_acc = add_lst(wm_test_acc, np.load(str(i) + aname + 'wm_test_acc.npy'))
        watermark_retention_acc = add_lst(watermark_retention_acc,
                                          np.load(str(i) + aname + 'watermark_retention_acc.npy'))
    train_acc = [x / wm_ for x in train_acc]
    non_wm_test_acc = [x / wm_ for x in non_wm_test_acc]
    wm_test_acc = [x / wm_ for x in wm_test_acc]
    watermark_retention_acc = [x / wm_ for x in watermark_retention_acc]

    x_range = list(range(0, x_range))

    plt.plot(train_acc, color='y', label='train accuracy')
    plt.plot(non_wm_test_acc, color='r', label='non watermark test acc')
    plt.plot(wm_test_acc, color='m', label='watermark test acc')
    plt.plot(watermark_retention_acc, color='c', label='watermark retention')

    np.save('train_acc', train_acc)
    np.save('non_wm_test_acc', non_wm_test_acc)
    np.save('wm_test_acc', wm_test_acc)
    np.save('watermark_retention_acc', watermark_retention_acc)
    plt.title('random signature average accuracies')

    plot_line(x_range, wm_test_acc, 2)
    plot_line(x_range, watermark_retention_acc, 3)

    pylab.legend(loc='best')

    plt.plot(train_acc, 'yo')
    plt.plot(non_wm_test_acc, 'ro')
    plt.plot(wm_test_acc, 'mo')
    plt.plot(watermark_retention_acc, 'co')

    plt.savefig('final_plot')
    plt.clf()


