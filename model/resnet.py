from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU, MaxPool2D, ZeroPadding2D


class ResNet50Encoder(Layer):
    def __init__(self):
        super().__init__()

        blocks = [3, 4, 6, 3]
        filters = [64, 256, 512, 1024, 2048]
        
        self.pad1 = ZeroPadding2D(3)
        self.conv1 = Conv2D(filters[0], 7, 2, use_bias=False)
        self.bn1 = BatchNormalization(momentum=0.1, epsilon=1e-5)
        self.relu = ReLU()
        self.pad2 = ZeroPadding2D(1)
        self.maxpool = MaxPool2D(3, 2)

        self.layer1 = self._make_layer(filters[1], blocks[0], strides=1, dilation_rate=1)
        self.layer2 = self._make_layer(filters[2], blocks[1], strides=2, dilation_rate=1)
        self.layer3 = self._make_layer(filters[3], blocks[2], strides=2, dilation_rate=1)
        self.layer4 = self._make_layer(filters[4], blocks[3], strides=1, dilation_rate=2)

    def _make_layer(self, filters, blocks, strides, dilation_rate):
        layers = [ResNetBlock(filters, 3, strides, 1, True)]
        for _ in range(1, blocks):
            layers.append(ResNetBlock(filters, 3, 1, dilation_rate, False))
        return Sequential(layers)
        
    def call(self, x, training=None):
        x = self.pad1(x)
        x = self.conv1(x, training=training)
        x = self.bn1(x, training=training)
        x = self.relu(x, training=training)
        f1 = x  # 1/2
        x = self.pad2(x)
        x = self.maxpool(x, training=training)
        x = self.layer1(x, training=training)
        f2 = x  # 1/4
        x = self.layer2(x, training=training)
        f3 = x  # 1/8
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        f4 = x  # 1/16
        return f1, f2, f3, f4


class ResNetBlock(Layer):
    def __init__(self, filters, kernel_size=3, strides=1, dilation_rate=1, conv_shortcut=True):
        super().__init__()
        self.conv1 = Conv2D(filters // 4, 1, use_bias=False)
        self.bn1 = BatchNormalization(momentum=0.1, epsilon=1e-5)
        self.pad2 = ZeroPadding2D(dilation_rate)
        self.conv2 = Conv2D(filters // 4, kernel_size, strides, dilation_rate=dilation_rate, use_bias=False)
        self.bn2 = BatchNormalization(momentum=0.1, epsilon=1e-5)
        self.conv3 = Conv2D(filters, 1, use_bias=False)
        self.bn3 = BatchNormalization(momentum=0.1, epsilon=1e-5)
        self.relu = ReLU()
        if conv_shortcut:
            self.convd = Conv2D(filters, 1, strides, use_bias=False)
            self.bnd = BatchNormalization(momentum=0.1, epsilon=1e-5)

    def call(self, x, training=None):
        if hasattr(self, 'convd'):
            shortcut = self.convd(x, training=training)
            shortcut = self.bnd(shortcut, training=training)
        else:
            shortcut = x

        x = self.conv1(x, training=training)
        x = self.bn1(x, training=training)
        x = self.relu(x, training=training)
        x = self.pad2(x, training=training)
        x = self.conv2(x, training=training)
        x = self.bn2(x, training=training)
        x = self.relu(x, training=training)
        x = self.conv3(x, training=training)
        x = self.bn3(x, training=training)
        x += shortcut
        x = self.relu(x, training=training)
        return x