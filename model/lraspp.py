import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU, Activation

class LRASPP(Layer):
    def __init__(self, out_channels: int):
        super().__init__()
        self.aspp1 = Sequential([
            Conv2D(out_channels, 1, use_bias=False),
            BatchNormalization(momentum=0.1, epsilon=1e-5),
            ReLU()
        ])
        self.aspp2 = Conv2D(out_channels, 1, use_bias=False, activation='sigmoid')
        
    def call(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(tf.reduce_mean(x, axis=[1, 2], keepdims=True))
        return x1 * x2
