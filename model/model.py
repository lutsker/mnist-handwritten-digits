from keras.backend import shape
from keras.models import Model
from .layers import Conv
from .layers import FC
from .layers import Inputs


def model_factory():

    inp = Inputs()
    conv_1 = Conv(filters=32, kernel_size=(3, 3), block_name='conv_1')(inp)
    conv_2 = Conv(filters=64, kernel_size=(3, 3), block_name='conv_2')(conv_1)
    output = FC(num_classes=10)(conv_2)

    return Model(inp, output)
