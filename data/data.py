import numpy as np 
from tensorflow.keras import datasets 
from tensorflow.keras.utils import to_categorical

def dataset():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)    
    def result(set):
        return (x_train, y_train) if set == 'train' else (x_test, y_test)
    return result 
