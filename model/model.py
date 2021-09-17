import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D

from .mobilenetv3 import MobileNetV3Encoder
from .resnet import ResNet50Encoder
from .lraspp import LRASPP
from .decoder import RecurrentDecoder
from .fast_guided_filter import FastGuidedFilterRefiner
from .deep_guided_filter import DeepGuidedFilterRefiner

class MattingNetwork(Model):
    def __init__(self,
                 variant: str = 'mobilenetv3',
                 refiner: str = 'deep_guided_filter'):
        super().__init__()
        assert variant in ['mobilenetv3', 'resnet50']
        assert refiner in ['deep_guided_filter', 'fast_guided_filter']
        
        if variant == 'mobilenetv3':
            self.backbone = MobileNetV3Encoder()
            self.aspp = LRASPP(128)
            self.decoder = RecurrentDecoder([128, 80, 40, 32, 16])
        else:
            self.backbone = ResNet50Encoder()
            self.aspp = LRASPP(256)
            self.decoder = RecurrentDecoder([256, 128, 64, 32, 16])
            
        self.project_mat = Conv2D(4, 1)
            
        if refiner == 'deep_guided_filter':
            self.refiner = DeepGuidedFilterRefiner()
        else:
            self.refiner = FastGuidedFilterRefiner()
    
    def call(self, inputs):
        src, *rec, downsample_ratio = inputs

        src_sm = tf.cond(downsample_ratio == 1,
            lambda: src,
            lambda: self._downsample(src, downsample_ratio))
        
        f1, f2, f3, f4 = self.backbone(src_sm)
        f4 = self.aspp(f4)
        hid, *rec = self.decoder([src_sm, f1, f2, f3, f4, *rec])
        out = self.project_mat(hid)
        fgr_residual, pha = tf.split(out, [3, 1], -1)
        
        fgr_residual, pha = tf.cond(downsample_ratio == 1,
            lambda: (fgr_residual, pha), 
            lambda: self.refiner([src, src_sm, fgr_residual, pha, hid]))
        
        fgr = fgr_residual + src
        fgr = tf.clip_by_value(fgr, 0, 1)
        pha = tf.clip_by_value(pha, 0, 1)
        return fgr, pha, *rec
    
    def _downsample(self, x, downsample_ratio):
        size = tf.shape(x)[1:3]
        size = tf.cast(size, tf.float32) * tf.cast(downsample_ratio, tf.float32)
        size = tf.cast(size, tf.int32)
        return tf.image.resize(x, size)