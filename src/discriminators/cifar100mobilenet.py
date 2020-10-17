import numpy as np
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from src.discriminators.base_discriminator import Base_Discriminator


class CIFAR100_MobileNet(Base_Discriminator):
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

    def lr_schedule(self, epoch):
        """Learning Rate Schedule

        Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.

        # Arguments
            epoch (int): The number of epochs

        # Returns
            lr (float32): learning rate
        """
        lr = 1e-1
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr

    def build_model(self):
        """ Gets a certain model architecture for MNIST with a high capacity
        """
        model = MobileNet(include_top=True,
                          weights=None,
                          input_shape=(32, 32, 3),
                          classes=100)
        return model

    def compile_model(self):
        loss = self.loss if self.loss else "categorical_crossentropy"
        optimizer = self.optimizer if self.optimizer else Adam()

        self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    def train(self,
              train_data,
              test_data):
        # Define parameters
        epochs = self.epochs if self.epochs else 70
        batch_size = 128

        (x_train, y_train), (x_test, y_test) = train_data, test_data
        self.compile_model()

        # Actually train the model
        lr_scheduler = LearningRateScheduler(self.lr_schedule)
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-2)
        callbacks = [lr_reducer, lr_scheduler]
        if self.callbacks:
            callbacks += self.callbacks
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format="channels_last",
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        history_cb = self.model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                              validation_data=(x_test, y_test),
                                              epochs=epochs, verbose=1, workers=1,
                                              callbacks=callbacks)
        self.history = history_cb.history
        return self.model, self.history
