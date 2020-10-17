from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, AveragePooling2D
from tensorflow.python.keras.optimizers import Adam

from src.discriminators.base_discriminator import Base_Discriminator

import numpy as np

class MNIST_Model_B(Base_Discriminator):
    """ A basic model for MNIST
    """

    def __init__(self,
                 train_data=None,
                 test_data=None,
                 prefix_path=None,
                 reg=0,
                 epochs=None,
                 loss=None,
                 optimizer=None,
                 callbacks=None,
                 retrain=False,
                 bootstrap_sampling=None):
        self.reg = reg
        self.loss = loss
        self.epochs = epochs
        self.optimizer = optimizer
        self.callbacks = callbacks
        super().__init__(train_data, test_data, prefix_path, bootstrap_sampling, retrain)

    def build_model(self):
        """ Gets a certain model architecture for MNIST with a high capacity
        """
        model = Sequential()

        model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(AveragePooling2D())
        model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
        model.add(AveragePooling2D())
        model.add(Flatten())
        model.add(Dense(units=120, activation='relu'))
        model.add(Dense(units=84, activation='relu'))
        model.add(Dense(units=10, activation='softmax'))
        return model

    def compile_model(self):
        loss = self.loss if self.loss else "categorical_crossentropy"
        optimizer = self.optimizer if self.optimizer else Adam()

        self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    def train(self,
              train_data,
              test_data):
        epochs = self.epochs if self.epochs else 8

        self.compile_model()
        history_cb = self.model.fit(train_data[0], train_data[1], validation_data=test_data,
                                      epochs=epochs, callbacks=self.callbacks)
        self.history = history_cb.history
        return self.model, self.history