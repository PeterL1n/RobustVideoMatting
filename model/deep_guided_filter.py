import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, ZeroPadding2D

class DeepGuidedFilterRefiner(Layer):
    def __init__(self, hid_channels=16, radius=1):
        super().__init__()
        self.box_filter = Sequential([
            ZeroPadding2D(1),
            DepthwiseConv2D(3, use_bias=False)
        ])
        self.box_filer = Conv2D(4, 3, padding='SAME', dilation_rate=radius, use_bias=False, groups=4)
        self.conv = Sequential([
            Conv2D(hid_channels, 1, use_bias=False),
            BatchNormalization(momentum=0.1, epsilon=1e-5),
            ReLU(),
            Conv2D(hid_channels, 1, use_bias=False),
            BatchNormalization(momentum=0.1, epsilon=1e-5),
            ReLU(),
            Conv2D(4, 1)
        ])
        
    def call(self, inputs):
        fine_src, base_src, base_fgr, base_pha, base_hid = inputs
        fine_x = tf.concat([fine_src, tf.reduce_mean(fine_src, -1, keepdims=True)], -1)
        base_x = tf.concat([base_src, tf.reduce_mean(base_src, -1, keepdims=True)], -1)
        base_y = tf.concat([base_fgr, base_pha], -1)
        
        mean_x = self.box_filter(base_x)
        mean_y = self.box_filter(base_y)
        cov_xy = self.box_filter(base_x * base_y) - mean_x * mean_y
        var_x  = self.box_filter(base_x * base_x) - mean_x * mean_x
        
        A = self.conv(tf.concat([cov_xy, var_x, base_hid], -1))
        b = mean_y - A * mean_x
        
        H, W = tf.shape(fine_src)[1], tf.shape(fine_src)[2]
        mean_A = tf.image.resize(A, (H, W))
        mean_b = tf.image.resize(b, (H, W))
        out = mean_A * fine_x + mean_b
        fgr, pha = tf.split(out, [3, 1], -1)
        return fgr, pha