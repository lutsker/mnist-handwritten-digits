from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Input


def Conv(filters, kernel_size, block_name):
    def result(input):
        conv = Conv2D(filters, kernel_size, padding="same", name=block_name)(input)
        relu = Activation('relu')(conv)
        pool = MaxPooling2D(pool_size=(2, 2))(relu)
        return pool
    return result

def FC(num_classes):
    def result(input):
        flatten = Flatten(name='flat')(input)
        dropout = Dropout(0.5, name='drop')(flatten)
        dense = Dense(units=num_classes)(dropout)
        softmax = Activation('softmax', name='softmax_x')(dense)
        return softmax
    return result

def Inputs():
    return Input(shape=(28, 28, 1))
