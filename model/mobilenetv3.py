import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU, ZeroPadding2D, DepthwiseConv2D
from typing import Optional

from .utils import normalize


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _hard_sigmoid(x):
    return tf.nn.relu6(x + 3) / 6


class SqueezeExcitation(Layer):
    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = Conv2D(squeeze_channels, 1)
        self.relu = ReLU()
        self.fc2 = Conv2D(input_channels, 1)
        
    def call(self, x):
        scale = tf.reduce_mean(x, axis=[1,2], keepdims=True)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = _hard_sigmoid(scale)
        return scale * x
    
    
class Hardswish(Layer):
    def call(self, x):
        return x * _hard_sigmoid(x)


class ConvBNActivation(Layer):
    def __init__(self, filters, kernel_size, stride=1, groups=1, dilation=1, activation_layer=None):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        if padding != 0:
            self.pad = ZeroPadding2D(padding)
        if groups == 1:
            self.conv = Conv2D(filters, kernel_size, stride, dilation_rate=dilation, groups=groups, use_bias=False)
        else:
            self.conv = DepthwiseConv2D(kernel_size, stride, dilation_rate=dilation, use_bias=False)
        self.bn = BatchNormalization(momentum=0.01, epsilon=1e-3)
        if activation_layer:
            self.act = activation_layer()

    def call(self, x):
        if hasattr(self, 'pad'):
            x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x)
        if hasattr(self, 'act'):
            x = self.act(x)
        return x


class InvertedResidual(Layer):
    def __init__(self,
                 input_channels: int,
                 kernel: int,
                 expanded_channels: int,
                 out_channels: int,
                 use_se: bool,
                 activation: str,
                 stride: int,
                 dilation: int):
        super().__init__()
        if not (1 <= stride <= 2):
            raise ValueError('illegal stride value')

        self.use_res_connect = stride == 1 and input_channels == out_channels

        layers = []
        activation_layer = Hardswish if activation == 'HS' else ReLU

        # expand
        if expanded_channels != input_channels:
            layers.append(ConvBNActivation(
                expanded_channels,
                kernel_size=1,
                activation_layer=activation_layer))
        
        # depthwise
        stride = 1 if dilation > 1 else stride
        layers.append(ConvBNActivation(
            expanded_channels,
            kernel_size=kernel,
            stride=stride,
            dilation=dilation,
            groups=expanded_channels,
            activation_layer=activation_layer))
        if use_se:
            layers.append(SqueezeExcitation(expanded_channels))

        # project
        layers.append(ConvBNActivation(
            out_channels,
            kernel_size=1,
            activation_layer=None))

        self.block = Sequential(layers)

    def call(self, input):
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class MobileNetV3Encoder(Layer):
    def __init__(self):
        super().__init__()
        self.features = [
            ConvBNActivation(16, kernel_size=3, stride=2, activation_layer=Hardswish),
            InvertedResidual(16, 3, 16, 16, False, 'RE', 1, 1),
            InvertedResidual(16, 3, 64, 24, False, 'RE', 2, 1), # C1
            InvertedResidual(24, 3, 72, 24, False, 'RE', 1, 1),
            InvertedResidual(24, 5, 72, 40, True, 'RE', 2, 1), # C2
            InvertedResidual(40, 5, 120, 40, True, 'RE', 1, 1),
            InvertedResidual(40, 5, 120, 40, True, 'RE', 1, 1),
            InvertedResidual(40, 3, 240, 80, False, 'HS', 2, 1), # C3
            InvertedResidual(80, 3, 200, 80, False, 'HS', 1, 1),
            InvertedResidual(80, 3, 184, 80, False, 'HS', 1, 1),
            InvertedResidual(80, 3, 184, 80, False, 'HS', 1, 1),
            InvertedResidual(80, 3, 480, 112, True, 'HS', 1, 1),
            InvertedResidual(112, 3, 672, 112, True, 'HS', 1, 1),
            InvertedResidual(112, 5, 672, 160, True, 'HS', 2, 2), # C4
            InvertedResidual(160, 5, 960, 160, True, 'HS', 1, 2),
            InvertedResidual(160, 5, 960, 160, True, 'HS', 1, 2),
            ConvBNActivation(960, kernel_size=1, activation_layer=Hardswish)
        ]
        
    def call(self, x):
        x = normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        x = self.features[0](x)
        x = self.features[1](x)
        f1 = x
        x = self.features[2](x)
        x = self.features[3](x)
        f2 = x
        x = self.features[4](x)
        x = self.features[5](x)
        x = self.features[6](x)
        f3 = x
        x = self.features[7](x)
        x = self.features[8](x)
        x = self.features[9](x)
        x = self.features[10](x)
        x = self.features[11](x)
        x = self.features[12](x)
        x = self.features[13](x)
        x = self.features[14](x)
        x = self.features[15](x)
        x = self.features[16](x)
        f4 = x
        return f1, f2, f3, f4
