from mlxtend.data import loadlocal_mnist
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Mnist:
    def __init__(self):
        self.IMG_SIZE = 28
        self.train_img, self.train_label = loadlocal_mnist(images_path='mnist/train-images-idx3-ubyte', labels_path='mnist/train-labels-idx1-ubyte')
        self.test_img, self.test_label = loadlocal_mnist(images_path='mnist/t10k-images-idx3-ubyte', labels_path='mnist/t10k-labels-idx1-ubyte')
    
    def get_trainset(self, norm=False):
        if norm:
            scaler = MinMaxScaler()
            self.train_img = scaler.fit_transform(self.train_img)
        return self.train_img, self.train_label

    def get_testset(self, norm=False):
        if norm:
            scaler = MinMaxScaler()
            self.test_img = scaler.fit_transform(self.test_img)       
        return self.test_img, self.test_label

    def vet2mat(self, x):
        return x.reshape((self.IMG_SIZE, self.IMG_SIZE))

    def reshape_dataset(self, X):
        return np.array([self.vet2mat(x) for x in X])
            