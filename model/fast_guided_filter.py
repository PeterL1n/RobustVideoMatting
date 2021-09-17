import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, ZeroPadding2D

"""
Adopted from <https://github.com/wuhuikai/DeepGuidedFilter/>
"""

class BoxFilter(Layer):
    def __init__(self, r):
        super(BoxFilter, self).__init__()
        self.kernel_size = 2 * r + 1
        self.filter_x = DepthwiseConv2D((1, self.kernel_size), use_bias=False)
        self.filter_y = DepthwiseConv2D((self.kernel_size, 1), use_bias=False)

    def build(self, input_shape):
        self.filter_x.build(input_shape)
        self.filter_y.build(input_shape)
        weight_x = self.filter_x.get_weights()[0]
        weight_y = self.filter_y.get_weights()[0]
        weight_x.fill(1 / self.kernel_size)
        weight_y.fill(1 / self.kernel_size)
        self.filter_x.set_weights([weight_x])
        self.filter_y.set_weights([weight_y])

    def call(self, x):
        return self.filter_x(self.filter_y(x))


class FastGuidedFilter(Layer):
    def __init__(self, r: int, eps: float = 1e-5):
        super().__init__()
        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)

    def call(self, inputs):
        lr_x, lr_y, hr_x = inputs
        mean_x = self.boxfilter(lr_x)
        mean_y = self.boxfilter(lr_y)
        cov_xy = self.boxfilter(lr_x * lr_y) - mean_x * mean_y
        var_x = self.boxfilter(lr_x * lr_x) - mean_x * mean_x

        A = cov_xy / (var_x + self.eps)
        b = mean_y - A * mean_x

        ## mean_A; mean_b
        H, W = tf.shape(hr_x)[1], tf.shape(hr_x)[2]
        mean_A = tf.image.resize(A, (H, W))
        mean_b = tf.image.resize(b, (H, W))
        return mean_A * hr_x + mean_b


class FastGuidedFilterRefiner(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.guilded_filter = FastGuidedFilter(1)
    
    def call(self, inputs):
        fine_src, base_src, base_fgr, base_pha = inputs[:4]
        
        fine_src_gray = tf.reduce_mean(fine_src, -1, keepdims=True)
        base_src_gray = tf.reduce_mean(base_src, -1, keepdims=True)
        
        out = self.guilded_filter([
            tf.concat([base_src, base_src_gray], -1),
            tf.concat([base_fgr, base_pha], -1),
            tf.concat([fine_src, fine_src_gray], -1)
        ])
        
        fgr, pha = tf.split(out, [3, 1], -1)
        
        return fgr, pha
    