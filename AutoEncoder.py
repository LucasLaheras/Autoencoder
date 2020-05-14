import os
import numpy as np
from keras.datasets import mnist
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard
from PIL import Image

class ConvAutoEncoder:

    def __init__(self, input_shape, output_dim, filters=None,
                 kernel=(3, 3), optimizer='adadelta', lossfn='binary_crossentropy'):

        if filters is None:
            filters = [16, 8, 8]

        self.mse = None

        self.input_shape = input_shape
        self.output_dim = output_dim

        input_layer = Input(input_shape)

        x = Conv2D(filters[0], kernel, activation='relu', padding='same')(input_layer)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(filters[1], kernel, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(filters[2], kernel, activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(filters[2], kernel, activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(filters[1], kernel, activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(filters[0], kernel, activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, kernel, activation='sigmoid', padding='same')(x)

        # create autoencoder and decoder model
        self.autoencoder = Model(input_layer, decoded)
        self.encoder = Model(input_layer, encoded)

        # create a placeholder for an encoded input
        enc_shape = encoded[0].shape

        encoded_input = Input(shape=(int(enc_shape[0]), int(enc_shape[1]), int(enc_shape[2])))

        # retrieve the decoder layers and apply to each prev layer
        num_decoder_layers = 7
        decoder_layer = encoded_input
        for i in range(-num_decoder_layers, 0):
            decoder_layer = self.autoencoder.layers[i](decoder_layer)

        # create the decoder model
        self.decoder = Model(encoded_input, decoder_layer)

        # compile model
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

    def autoencode(self, input):
        return self.autoencoder.predict(input)

    def save_weights(self, path=None, prefix=""):
        if path is None: path = os.getcwd()
        self.autoencoder.save_weights(os.path.join(path, prefix + "autoencoder_weights.h5"))
        self.encoder.save_weights(os.path.join(path, prefix + "encoder_weights.h5"))
        self.decoder.save_weights(os.path.join(path, prefix + "decoder_weights.h5"))

    def load_weights(self, path=None, prefix=""):
        if path is None: path = os.getcwd()
        self.autoencoder.load_weights(os.path.join(path, prefix + "autoencoder_weights.h5"))
        self.encoder.load_weights(os.path.join(path, prefix + "encoder_weights.h5"))
        self.decoder.load_weights(os.path.join(path, prefix + "decoder_weights.h5"))


def configureDataset():
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

    return x_train, x_test


def save_img(imgs, stri):
    num = imgs.shape[0]

    for i in range(num):
        A = imgs[i].copy()*255
        A = np.reshape(np.ravel(A), (28, 28))
        new_p = Image.fromarray(A)
        if new_p.mode != 'RGB':
            new_p = new_p.convert('RGB')
        new_p.save(os.path.join(stri, str(i) + ".jpg"))


if __name__ == '__main__':

    x_train, x_test = configureDataset()

    #save_img(x_test, 'IN/')

    auto = ConvAutoEncoder(x_train[0].shape, x_train[0].shape)

    print(x_train.shape, x_test.shape)

    auto.fit(x_train, x_test, epochs=1, batch_size=128, shuffle=True, validation_data=(x_test, x_test))

    auto.save_weights(prefix="mnist_")

    auto.load_weights(prefix="mnist_")

    a = auto.encode(x_test)
    b = auto.decode(a)

    #save_img(b, 'OUT/')
