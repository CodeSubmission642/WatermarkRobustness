from tensorflow.keras.layers import Dense, Reshape, BatchNormalization, Conv2DTranspose, \
    Concatenate, Add, LeakyReLU, \
    Embedding, Flatten, Lambda
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.initializers import RandomNormal

import tensorflow as tf

from src.generators.base_generator import Generator_Base


class GAN_MNIST(Generator_Base):
    """ GAN for MNIST dataset that generates targeted adversarial samples for the discriminator
        Note that the GAN only learns a minimal clipped delta that is added on top of the image
    """
    def build_generator(self):
        """ Builds the generator which learns a delta vector to place on top of the image
        """

        def normalize(filters):
            img, delta = filters
            return tf.keras.backend.clip(img, 0, 1)

        init = RandomNormal(stddev=0.02)

        img_data = Input(shape=(28, 28, 1))
        img_flat = Flatten()(img_data)
        y_true = Input(shape=(self.nb_class,))

        y_des = Input(shape=(self.nb_class,))
        des_emb = Embedding(10, 64, input_length=self.nb_class)(y_des)
        dense1 = Dense(50, activation='sigmoid')(des_emb)
        flatten = Flatten()(dense1)

        cc0 = Concatenate()([img_flat, flatten])

        fc1 = Dense(7 * 7 * 256, input_shape=(28 * 28 + 50,), use_bias=False, kernel_initializer=init)(cc0)
        bn1 = BatchNormalization()(fc1)
        lk1 = LeakyReLU()(bn1)
        rs1 = Reshape((7, 7, 256))(lk1)
        conv1 = Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(rs1)
        bn2 = BatchNormalization()(conv1)
        lk2 = LeakyReLU()(bn2)
        conv2 = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(lk2)
        bn3 = BatchNormalization()(conv2)
        lk3 = LeakyReLU()(bn3)
        delta = Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(lk3)
        img_raw = Add()([delta, img_data])
        output = Lambda(normalize)([img_raw, delta])
        generator = Model(inputs=[img_data, y_true, y_des], outputs=output)
        generator.add_loss(self.gan_loss(img_data, output, y_true, y_des))
        return generator

