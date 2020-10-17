
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense, Reshape, BatchNormalization, Conv2DTranspose, \
    Concatenate, Add, LeakyReLU, \
    Embedding, Flatten, Lambda
from tensorflow.python.keras.initializers import RandomNormal
import tensorflow as tf

from src.generators.base_generator import Generator_Base


class GAN_CIFAR(Generator_Base):
    """ GAN for CIFAR dataset that generates targeted adversarial samples for the discriminator
        Note that the GAN only learns a minimal clipped delta that is added on top of the image
    """

    def build_generator(self):
        """ Builds the generator which learns a delta vector to place on top of the image
        """
        def normalize(filters):
            img, delta = filters

            return tf.keras.backend.clip(
                img,
                0,
                1
            )
            # return img / (tf.ones_like(delta)+delta)

        init = RandomNormal(stddev=0.02)

        img_data = Input(shape=(32, 32, 3))
        img_flat = Flatten()(img_data)
        y_true = Input(shape=(self.nb_class,))

        y_des = Input(shape=(self.nb_class,))
        des_emb = Embedding(10, 64, input_length=self.nb_class)(y_des)
        dense1 = Dense(50, activation='sigmoid')(des_emb)
        flatten = Flatten()(dense1)

        cc0 = Concatenate()([img_flat, flatten])

        fc1 = Dense(4 * 4 * 256, input_shape=(28 * 28 * 3 + 50,), use_bias=False, kernel_initializer=init)(cc0)
        bn1 = BatchNormalization()(fc1)
        lk1 = LeakyReLU(alpha=0.2)(bn1)
        rs1 = Reshape((4, 4, 256))(lk1)
        conv1 = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False)(rs1)
        bn2 = BatchNormalization()(conv1)
        lk2 = LeakyReLU(alpha=0.2)(bn2)
        conv2 = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False)(lk2)
        bn3 = BatchNormalization()(conv2)
        lk3 = LeakyReLU(alpha=0.2)(bn3)
        conv3 = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False)(lk3)
        lk4 = LeakyReLU(alpha=0.2)(conv3)
        delta = Conv2D(3, (3, 3), activation='tanh', padding='same')(lk4)
        img_raw = Add()([delta, img_data])
        output = Lambda(normalize)([img_raw, delta])

        generator = Model(inputs=[img_data, y_true, y_des], outputs=output)
        generator.add_loss(self.gan_loss(img_data, output, y_true, y_des))
        return generator

