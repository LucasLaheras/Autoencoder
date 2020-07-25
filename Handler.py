import pickle
import pandas
import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random


def mostra(img, name='Name'):
    img1 = img.copy()

    img1 = img1.astype(np.uint8)

    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name, img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class Handler():
    def __init__(self, dir=''):
        self.datadir = dir
        self.categories = ["benign", "malignant"]
        self.data = []
        self.img_size = 52

    def write_data(self, num):
        if num == 0:
            name = "X_train"
        else:
            name = "X_test"

        self.read_data()

        X = []

        for features in self.data:
            X.append(features)

        X = np.array(X).reshape(-1, self.img_size, self.img_size, 1)

        #mostra(X[1])

        pickle_out = open(name+".pickle", "wb")
        pickle.dump(X, pickle_out)
        pickle_out.close()

    def read_data(self):
        for category in self.categories:
            path = os.path.join(self.datadir, category)

            class_num = self.categories.index(category)
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array, (self.img_size, self.img_size))/255.
                    self.data.append(new_array)
                    new_array = cv2.rotate(new_array, cv2.ROTATE_90_CLOCKWISE)
                    #self.data.append(new_array)
                    new_array = cv2.rotate(new_array, cv2.ROTATE_90_CLOCKWISE)
                    #self.data.append(new_array)
                    new_array = cv2.rotate(new_array, cv2.ROTATE_90_CLOCKWISE)
                    #self.data.append(new_array)
                except Exception as e:
                    pass

    def read_datafile(self, name):
        pickle_in = open(name+".pickle", "rb")
        X = pickle.load(pickle_in)

        #mostra(X[1])

        return X