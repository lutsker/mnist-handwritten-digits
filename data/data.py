import numpy as np 
from tensorflow.keras import datasets 
from tensorflow.keras.utils import to_categorical


def load_and_preprocess_mnist():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)    
    return (x_train, y_train), (x_test, y_test)

def dataset():
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist()
    def result(set):
        return (x_train, y_train) if set == 'train' else (x_test, y_test)
    return result 

def dataset_inv():
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist()
    x_train = np.concatenate((x_train, 1-x_train / np.max(x_train)), axis=0)
    y_train = np.concatenate((y_train, y_train), axis=0)
    x_train = np.concatenate((x_train, (x_train > 0.9) * x_train), axis=0)
    y_train = np.concatenate((y_train, y_train), axis=0)

    index = np.arange(2 * 120000)
    np.random.shuffle(index)
    x_train = x_train[index]
    y_train = y_train[index]
    def result(set):
        return (x_train, y_train) if set == 'train' else (x_test, y_test)
    return result 
