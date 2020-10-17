from tensorflow.python.keras import Sequential, regularizers
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.python.keras.optimizers import Adam

from src.discriminators.base_discriminator import Base_Discriminator


class CIFAR10_Custom(Base_Discriminator):
    """ A basic model for CIFAR
    """

    def __init__(self,
                 train_data,
                 test_data,
                 prefix_path=None,
                 reg=0,
                 dropout_rate_conv=0,
                 dropout_rate_dense=0.5,
                 epochs=None,
                 loss=None,
                 optimizer=None,
                 callbacks=None,
                 retrain=False):
        self.dropout_rate_conv = dropout_rate_conv
        self.dropout_rate_dense = dropout_rate_dense
        self.reg = reg
        self.loss = loss
        self.epochs = epochs
        self.optimizer = optimizer
        self.callbacks = callbacks
        super().__init__(train_data, test_data, prefix_path, retrain)

    def build_model(self):
        """ Gets a certain model architecture for MNIST with a high capacity
        """

        reg = self.reg
        dropout_rate_conv = self.dropout_rate_conv
        dropout_rate_dense = self.dropout_rate_dense

        model = Sequential()
        model.add(
            Conv2D(32, (3, 3),
                   activation='relu',
                   kernel_regularizer=regularizers.l2(reg),
                   input_shape=(32, 32, 3),
                   padding='same'))
        model.add(BatchNormalization(axis=-1))
        model.add(
            Conv2D(32, (3, 3),
                   activation='relu',
                   kernel_regularizer=regularizers.l2(reg),
                   padding='same'))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout_rate_conv))

        model.add(
            Conv2D(2 * 32, (3, 3),
                   activation='relu',
                   kernel_regularizer=regularizers.l2(reg),
                   padding='same'))
        model.add(BatchNormalization(axis=-1))
        model.add(
            Conv2D(2 * 32, (3, 3),
                   activation='relu',
                   kernel_regularizer=regularizers.l2(reg),
                   padding='same'))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout_rate_conv))

        model.add(
            Conv2D(4 * 32, (3, 3),
                   activation='relu',
                   kernel_regularizer=regularizers.l2(reg),
                   padding='same'))
        model.add(BatchNormalization(axis=-1))
        model.add(
            Conv2D(4 * 32, (3, 3),
                   activation='relu',
                   kernel_regularizer=regularizers.l2(reg),
                   padding='same'))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout_rate_conv))

        model.add(Flatten())
        model.add(
            Dense(512, activation='relu', kernel_regularizer=regularizers.l2(reg)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate_dense))
        model.add(Dense(10, activation='softmax'))
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
