from keras.models import Model

from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Input


def Conv(filters, kernel_size, block_name):
    def result(input):
        conv = Conv2D(filters, kernel_size, padding="valid", name='conv2d_' + block_name)(input)
        relu = Activation('relu', name='relu_' + block_name)(conv)
        pool = MaxPooling2D(pool_size=(2, 2), name='maxpool_' + block_name)(relu)
        return pool
    return result


def FlatSoftmax():
    def result(input):
        flatten = Flatten()(input)
        softmax = Activation('softmax')(flatten)
        return softmax
    return result


def model_factory():
    
    inp = Input(shape=(None, None, 1))

    conv_1 = Conv(filters=18, kernel_size=(5, 5), block_name='layer_1')(inp)
    conv_2 = Conv(filters=48, kernel_size=(5, 5), block_name='layer_2')(conv_1)
    conv_3 = Conv(filters=360, kernel_size=(3, 3), block_name='layer_3')(conv_2)
    conv_4 = Conv2D(filters=10, kernel_size=(1,1), padding="valid", name='conv2d_final')(conv_3)
    output = FlatSoftmax()(conv_4)

    return Model(inp, output)
