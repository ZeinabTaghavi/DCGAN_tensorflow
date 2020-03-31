import tensorflow as tf
import glob
from tensorflow.keras import layers


def Generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    # model oputput shape is (batch_size , 7 ,7 , 256)

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    # h1=7 , padding=same , stride=1 -> 7*1 = 7
    # model oputput shape is (batch_size , 7 ,7 , 128)

    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    # h1=7 , padding=same , stride=2 -> 7*2 = 14
    # model oputput shape is (batch_size , 14 ,14 , 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    # h1=14 , padding=same , stride=2 -> 14*2 = 28
    # model oputput shape is (batch_size , 28 ,28 , 1)

    return model

def G_loss(fake_output,cross_entropy):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def Discriminator():
    model = tf.keras.Sequential()
    # Generator output shape was (batch_size , 28, 28, 1)
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

def D_loss(real_output, fake_output,cross_entropy):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss







