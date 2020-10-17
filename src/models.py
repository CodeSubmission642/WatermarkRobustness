from tensorflow.keras import regularizers
from tensorflow.keras.backend import categorical_crossentropy
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop


def get_model_c_for_mnist(optimizer='Adam',
                          loss=categorical_crossentropy,
                          reg=0):
    """ Gets a certain model architecture for MNIST with a low capacity
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model



def get_other_model_for_mnist(optimizer='Adam'):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def get_lenet_model_for_mnist(optimizer='Adam',
                              loss='categorical_crossentropy',
                              reg=0):
    """ Gets the LeNet Tensorflow model
    """
    img_size = (28, 28, 1)  # Width, Height, Numbers of Channels

    # Define model shape
    model = Sequential()
    model.add(
        Conv2D(20, (5, 5),
               activation='relu',
               input_shape=(img_size[0], img_size[1], img_size[2]),
               padding='same',
               kernel_regularizer=regularizers.l2(reg)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(
        Conv2D(50, (5, 5),
               activation='relu',
               padding='same',
               kernel_regularizer=regularizers.l2(reg)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(
        Dense(500,
              activation='relu',
              name='fc1',
              kernel_regularizer=regularizers.l2(reg)))

    model.add(
        Dense(84,
              activation='relu',
              name='fc2',
              kernel_regularizer=regularizers.l2(reg)))

    model.add(Dense(10, name='fc3', kernel_regularizer=regularizers.l2(reg)))

    model.add(Activation('softmax'))
    # Build tf graph for the model
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def get_simple_model(input_shape=(52,),
                     loss='binary_crossentropy',
                     optimizer='Adam'):
    """ Gets a simple model for the property inference attack
    """
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=input_shape))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def get_deep_cnn_for_cifar(
        optimizer=RMSprop(lr=0.001),
        dropout_rate_conv=0,
        dropout_rate_dense=0.5,
        freeze_first_layers=0,
        reg=0,  # Regularization
        loss='categorical_crossentropy',
        cache=None):  # Filename to load the CNN weights
    """ Gets a model for CIFAR-10
    """
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

    # Freeze layer if required
    for layer in model.layers[:freeze_first_layers]:
        print("Froze layer {}".format(layer))
        layer.trainable = False

    if not (cache is None):
        model.load_weights(cache)

    model.compile(loss=loss, metrics=['accuracy'], optimizer=optimizer)
    return model
