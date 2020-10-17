import os
import numpy as np
import keras
import MyCallBack
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense
from keras.datasets import mnist

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=""


def preprocess_training_set(x_train, y_train):
    trainX = np.reshape(x_train, [x_train.shape[0], x_train.shape[1]*x_train.shape[2]])
    trainX_normal = (trainX/255.)
    trainX_normal = trainX_normal.astype('float32')
    y_train = np_utils.to_categorical(y_train, 10)
    return {0:trainX_normal, 1:y_train}


def preprocess_test_set(x_test, my_y_test):
    testX = np.reshape(x_test, [x_test.shape[0], x_test.shape[1]*x_test.shape[2]])
    testX_normal = (testX/255.)
    testX_normal = testX_normal.astype('float32')
    return {0:testX_normal, 1:np_utils.to_categorical(my_y_test, 10)}


def get_model():
    model = Sequential()
    model.add(Dense(512, activation='relu', use_bias=True, input_shape=(784,)))
    model.add(Dense(128, activation='relu', use_bias=True))
    model.add(Dense(10, activation='softmax', use_bias=True))
    mysgd = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(loss='mean_squared_error',
                  optimizer='Adam',
                  metrics=['accuracy'])
    return model


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = np.load('wm_Xtrain.npy')
y_train = np.load('wm_ytrain.npy')

wm_X_train = np.load('all_wm_Xtrain.npy')
wm_y_train = np.load('all_wm_ytrain.npy')

non_wm_X_train = np.load('non_wm_Xtrain.npy')
non_wm_y_train = np.load('non_wm_ytrain.npy')

wm_X_test = np.load('wm_X_test.npy')
wm_y_test = np.load('wm_y_test.npy')

train_set = preprocess_training_set(X_train, y_train)
trainX_normal = train_set[0]
Y_train = train_set[1]

non_wm_X_train_normal = preprocess_test_set(non_wm_X_train, non_wm_y_train)[0]

wm_X_train_normal = preprocess_test_set(wm_X_train, wm_y_train)[0]

X_test_normal = preprocess_test_set(X_test, y_test)[0]
y_test_normal = preprocess_test_set(X_test, y_test)[1]

wm_X_test_normal = preprocess_test_set(wm_X_test, wm_y_test)[0]
wm_y_test_normal = preprocess_test_set(wm_X_test, wm_y_test)[1]


# prepare callback
callback = MyCallBack.MyCallBack(my_wm_test={'X':wm_X_test_normal, 'y': wm_y_test},
                                 my_non_wm_test={'X': X_test_normal, 'y': y_test})
labels = np.load('labels_cnn_9777.npy')
model = get_model()
# MyCallBack.EarlyStoppingByLossVal(monitor='val_loss', value=0.001, verbose=1),
model.fit(non_wm_X_train_normal, np_utils.to_categorical(labels,10), batch_size=32, epochs=150,
          verbose=1, validation_data=(wm_X_train_normal, np_utils.to_categorical(wm_y_train, 10)), callbacks=[callback])
