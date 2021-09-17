import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU, Activation, AveragePooling2D, UpSampling2D

class RecurrentDecoder(Layer):
    def __init__(self, channels):
        super().__init__()
        self.avgpool = AveragePooling2D(padding='SAME')
        self.decode4 = BottleneckBlock(channels[0])
        self.decode3 = UpsamplingBlock(channels[1])
        self.decode2 = UpsamplingBlock(channels[2])
        self.decode1 = UpsamplingBlock(channels[3])
        self.decode0 = OutputBlock(channels[4])
        
    def call(self, inputs):
        s0, f1, f2, f3, f4, r1, r2, r3, r4 = inputs
        s1 = self.avgpool(s0)
        s2 = self.avgpool(s1)
        s3 = self.avgpool(s2)
        x4, r4 = self.decode4([f4, r4])
        x3, r3 = self.decode3([x4, f3, s3, r3])
        x2, r2 = self.decode2([x3, f2, s2, r2])
        x1, r1 = self.decode1([x2, f1, s1, r1])
        x0 = self.decode0([x1, s0])
        return x0, r1, r2, r3, r4


class BottleneckBlock(Layer):
    def __init__(self, channels):
        super().__init__()
        self.gru = ConvGRU(channels // 2)
        
    def call(self, inputs):
        x, r = inputs
        a, b = tf.split(x, 2, -1)
        b, r = self.gru([b, r])
        x = tf.concat([a, b], -1)
        return x, r


class UpsamplingBlock(Layer):
    def __init__(self, channels):
        super().__init__()
        self.upsample = UpSampling2D(interpolation='bilinear')
        self.conv = Sequential([
            Conv2D(channels, 3, padding='SAME', use_bias=False),
            BatchNormalization(momentum=0.1, epsilon=1e-5),
            ReLU()
        ])
        self.gru = ConvGRU(channels // 2)
        
    def call(self, inputs):
        x, f, s, r = inputs
        x = self.upsample(x)
        x = tf.image.crop_to_bounding_box(x, 0, 0, tf.shape(s)[1], tf.shape(s)[2])
        x = tf.concat([x, f, s], -1)
        x = self.conv(x)
        a, b = tf.split(x, 2, -1)
        b, r = self.gru([b, r])
        x = tf.concat([a, b], -1)
        return x, r


class OutputBlock(Layer):
    def __init__(self, channels):
        super().__init__()
        self.upsample = UpSampling2D(interpolation='bilinear')
        self.conv = Sequential([
            Conv2D(channels, 3, padding='SAME', use_bias=False),
            BatchNormalization(momentum=0.1, epsilon=1e-5),
            ReLU(),
            Conv2D(channels, 3, padding='SAME', use_bias=False),
            BatchNormalization(momentum=0.1, epsilon=1e-5),
            ReLU(),
        ])
    
    def call(self, inputs):
        x, s = inputs
        x = self.upsample(x)
        x = tf.image.crop_to_bounding_box(x, 0, 0, tf.shape(s)[1], tf.shape(s)[2])
        x = tf.concat([x, s], -1)
        x = self.conv(x)
        return x


class ConvGRU(Layer):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.ih = Conv2D(channels * 2, kernel_size, padding='SAME', activation='sigmoid')
        self.hh = Conv2D(channels, kernel_size, padding='SAME', activation='tanh')
        
    def call(self, inputs):
        x, h = inputs
        h = tf.broadcast_to(h, tf.shape(x))
        r, z = tf.split(self.ih(tf.concat([x, h], -1)), 2, -1)
        c = self.hh(tf.concat([x, r * h], -1))
        h = (1 - z) * h + z * c
        return h, h