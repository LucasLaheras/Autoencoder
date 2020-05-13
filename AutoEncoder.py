import os
import keras
import numpy as np
from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard
import tensorflow as tf
from keras import backend as K


class ConvAutoEncoder:

    def __init__(self, input_shape, output_dim, filters=None,
                 kernel=(3, 3), optimizer='adadelta', lossfn='binary_crossentropy'):

        if filters is None:
            filters = [16, 8, 8]

        self.mse = None

        self.input_shape = input_shape
        self.output_dim = output_dim

        input_layer = keras.layers.Input(input_shape)

        x = Conv2D(filters[0], kernel, activation='relu', padding='same')(input_layer)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(filters[1], kernel, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(filters[2], kernel, activation='relu', padding='same')(x)
        self.encoder = MaxPooling2D((2, 2), padding='same')(x)

        # at this point the representation is (4, 4, 8) i.e. 128-dimensional

        x = Conv2D(filters[2], kernel, activation='relu', padding='same')(self.encoder)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(filters[1], kernel, activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(filters[0], kernel, activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        self.decoder = Conv2D(1, kernel, activation='sigmoid', padding='same')(x)

        reconstructed = self.decoder

        # compile model
        self.autoencoder = Model(inputs=input_layer, outputs=reconstructed)
        self.autoencoder.compile(optimizer=optimizer, loss=lossfn)

        self.autoencoder.summary()

    def fit(self, train, test, epochs=50, batch_size=None, shuffle=True, validation_data=None, callbacks=None):

        if callbacks is None:
            callbacks = [TensorBoard(log_dir='/tmp/autoencoder')]

        self.autoencoder.fit(x=train, y=train, epochs=epochs, batch_size=batch_size, shuffle=shuffle,
                             validation_data=validation_data,
                             callbacks=callbacks)

        self.mse = self.autoencoder.evaluate(test, test)
        print('CAE MSE on validation data: ', self.mse)

    def encode(self, input):
        return self.encoder.predict(input)

    def decode(self, codes):
        return self.decoder.predict(codes)

    def save_weights(self, path=None, prefix=""):
        if path is None: path = os.getcwd()
        self.encoder.save_weights(os.path.join(path, prefix + "encoder_weights.h5"))
        self.decoder.save_weights(os.path.join(path, prefix + "decoder_weights.h5"))

    def load_weights(self, path=None, prefix=""):
        if path is None: path = os.getcwd()
        self.encoder.load_weights(os.path.join(path, prefix + "encoder_weights.h5"))
        self.decoder.load_weights(os.path.join(path, prefix + "decoder_weights.h5"))


def configureDataset():
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

    return x_train, x_test

if __name__ == '__main__':

    x_train, x_test = configureDataset()

    auto = ConvAutoEncoder(x_train[0].shape, x_train[0].shape)

    print(x_train.shape, x_test.shape)

    auto.fit(x_train, x_test, epochs=1, batch_size=128, shuffle=True, validation_data=(x_test, x_test))

    auto.save_weights("/content/drive/","mnist")
