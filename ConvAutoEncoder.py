import os
import numpy as np
from keras.datasets import mnist
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard
from PIL import Image
import pandas as pd
from Handler import Handler, mostra
from keras.utils.vis_utils import plot_model
import pickle
import matplotlib.pyplot as plt


class ConvAutoEncoder:

    def __init__(self, input_shape, output_dim, filters=None,
                 kernel=(3, 3), optimizer='adadelta', lossfn='mean_squared_error'):# mean_squared_error sparse_categorical_crossentropy categorical_crossentropy

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

        fit_return = self.autoencoder.fit(x=train, y=train, epochs=epochs, batch_size=batch_size, shuffle=shuffle,
                                          validation_data=validation_data, callbacks=callbacks)

        self.mse = self.autoencoder.evaluate(test, test)
        print('CAE MSE on validation data: ', self.mse)

        return fit_return

    def encode(self, input):
        return self.encoder.predict(input)

    def decode(self, codes):
        return self.decoder.predict(codes)

    def autoencode(self, input):
        return self.autoencoder.predict(input)

    def history(self):
        return self.autoencoder.history

    def save_weights(self, path=None, prefix=""):
        if path is None: path = os.getcwd()
        self.autoencoder.save_weights(os.path.join(path, prefix + "autoencoder_weights.h5"))
        self.autoencoder.save(os.path.join(path, prefix + "autoencoder_model.h5"))
        self.encoder.save_weights(os.path.join(path, prefix + "encoder_weights.h5"))
        self.encoder.save(os.path.join(path, prefix + "encoder_model.h5"))
        self.decoder.save_weights(os.path.join(path, prefix + "decoder_weights.h5"))
        self.decoder.save(os.path.join(path, prefix + "decoder_model.h5"))

    def load_weights(self, path=None, prefix=""):
        if path is None: path = os.getcwd()
        self.autoencoder.load_weights(os.path.join(path, prefix + "autoencoder_weights.h5"))
        self.encoder.load_weights(os.path.join(path, prefix + "encoder_weights.h5"))
        self.decoder.load_weights(os.path.join(path, prefix + "decoder_weights.h5"))


def configureDataset():
    (x_train, _), (x_test, _) = mnist.load_data()
    #bath normalization
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 52, 52, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 52, 52, 1))  # adapt this if using `channels_first` image data format

    return x_train, x_test


def save_img(imgs, stri):
    num = imgs.shape[0]

    for i in range(num):
        A = imgs[i].copy() * 255
        A = np.reshape(np.ravel(A), (Handler().img_size, Handler().img_size))
        new_p = Image.fromarray(A)
        if new_p.mode != 'RGB':
            new_p = new_p.convert('RGB')
        new_p.save(os.path.join(stri, str(i) + ".png"))

def save_dataframe(data, stri):
    # visualizing losses and accuracy
    #train_loss = data.history['loss']
    #val_loss = data.history['val_loss']
    #train_acc = data.history['acc']
    #val_acc = data.history['val_acc']
    #xc = range(num_epochs)

    # convert the history.history dict to a pandas DataFrame:
    hist_df = pd.DataFrame(data.history)

    # save to json:
    hist_json_file = 'history.json'
    with open(stri + hist_json_file, mode='w') as f:
        hist_df.to_json(f)

    # or save to csv:
    hist_csv_file = 'history.csv'
    with open(stri + hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)


if __name__ == '__main__':
    #x_train1, x_test1 = configureDataset()

    dir_train = "C:\\Users\\lucas\\PycharmProjects\\Autoencoder\\data\\training"
    dir_test = "C:\\Users\\lucas\\PycharmProjects\\Autoencoder\\data\\test"

    Handler(dir_train).write_data(0)
    Handler(dir_test).write_data(1)

    x_train = Handler().read_datafile("X_train")

    x_test = Handler().read_datafile("X_test")

    if not os.path.isdir('IN/'):
        os.makedirs('IN/')

    save_img(x_test, 'IN/')

    filters = []
    x = 8
    while x <= 32:
        y = 8
        while y <= 32:
            z = 8
            while z <= 32:
                if x >= y and y >= z:
                    filters.append([x, y, z])
                z *= 2
            y *= 2
        x *= 2
    print(filters)
    #filter_iterator = [8, 8, 8]

    for filter_iterator in filters:
        path = "OUT_%02d-%02d-%02d/" % (filter_iterator[0], filter_iterator[1], filter_iterator[2])

        if not os.path.isdir(path):
            os.makedirs(path)

        auto = ConvAutoEncoder(x_train[0].shape, x_train[0].shape, filters=filter_iterator)

        print(x_train.shape, x_test.shape)

        history = auto.fit(x_train, x_train, epochs=500, batch_size=128, shuffle=True, validation_data=(x_test, x_test))

        auto.save_weights(prefix=("db_%02d-%02d-%02d_" % (filter_iterator[0], filter_iterator[1], filter_iterator[2])))

        auto.load_weights(prefix=("db_%02d-%02d-%02d_" % (filter_iterator[0], filter_iterator[1], filter_iterator[2])))

        a = auto.encode(x_test)
        b = auto.decode(a)

        save_img(b, path)

        #results = auto.evaluate(x_test, x_test, verbose=2)

        #plot_model(auto, to_file=path + 'model.png')

        pickle_out = open(path + "history.pickle", "wb")
        pickle.dump(history.history, pickle_out)
        pickle_out.close()


        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
