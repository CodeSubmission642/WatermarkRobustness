from tensorflow.python.keras import Sequential, regularizers
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten
from tensorflow.python.keras.optimizers import Adam

from src.discriminators.base_discriminator import Base_Discriminator


class MNIST_Model_A(Base_Discriminator):
    """ A basic LeNet for MNIST
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
        img_size = (28, 28, 1)  # Width, Height, Numbers of Channels

        model = Sequential()
        model.add(Conv2D(20, (5, 5),
                         activation='relu',
                         input_shape=(img_size[0], img_size[1], img_size[2]),
                         padding='same',
                         kernel_regularizer=regularizers.l2(self.reg)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(50, (5, 5),
                         activation='relu',
                         padding='same',
                         kernel_regularizer=regularizers.l2(self.reg)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(500,
                        activation='relu',
                        name='fc1',
                        kernel_regularizer=regularizers.l2(self.reg)))
        model.add(Dense(84,
                        activation='relu',
                        name='fc2',
                        kernel_regularizer=regularizers.l2(self.reg)))
        model.add(Dense(10, name='fc3', kernel_regularizer=regularizers.l2(self.reg)))
        model.add(Activation('softmax'))
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
        history_cb = self.model.fit(train_data[0],
                                    train_data[1],
                                    validation_data=test_data,
                                    epochs=epochs,
                                    callbacks=self.callbacks)
        self.history = history_cb.history
        return self.model, self.history
