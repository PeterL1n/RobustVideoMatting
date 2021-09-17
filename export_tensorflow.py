"""
python export_tensorflow.py \
    --model-variant mobilenetv3 \
    --model-variant deep_guided_filter \
    --pytorch-checkpoint rvm_mobilenetv3.pth \
    --tensorflow-output rvm_mobilenetv3_tf
"""

import argparse
import torch
import tensorflow as tf

from model import MattingNetwork, load_torch_weights


# Add input output names and shapes
class MattingNetworkWrapper(MattingNetwork):
    @tf.function(input_signature=[[
        tf.TensorSpec(tf.TensorShape([None, None, None, 3]), tf.float32, 'src'),
        tf.TensorSpec(tf.TensorShape(None), tf.float32, 'r1i'),
        tf.TensorSpec(tf.TensorShape(None), tf.float32, 'r2i'),
        tf.TensorSpec(tf.TensorShape(None), tf.float32, 'r3i'),
        tf.TensorSpec(tf.TensorShape(None), tf.float32, 'r4i'),
        tf.TensorSpec(tf.TensorShape(None), tf.float32, 'downsample_ratio')
    ]])
    def call(self, inputs):
        fgr, pha, r1o, r2o, r3o, r4o = super().call(inputs)
        return {'fgr': fgr, 'pha': pha, 'r1o': r1o, 'r2o': r2o, 'r3o': r3o, 'r4o': r4o}

        
class Exporter:
    def __init__(self):
        self.parse_args()
        self.init_model()
        self.export()
        
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model-variant', type=str, required=True, choices=['mobilenetv3', 'resnet50'])
        parser.add_argument('--model-refiner', type=str, default='deep_guided_filter', choices=['deep_guided_filter', 'fast_guided_filter'])
        parser.add_argument('--pytorch-checkpoint', type=str, required=True)
        parser.add_argument('--tensorflow-output', type=str, required=True)
        self.args = parser.parse_args()
        
    def init_model(self):
        # Construct model
        self.model = MattingNetworkWrapper(self.args.model_variant, self.args.model_refiner)
        # Build model
        src = tf.random.normal([1, 1080, 1920, 3])
        rec = [ tf.constant(0.) ] * 4
        downsample_ratio = tf.constant(0.25)
        self.model([src, *rec, downsample_ratio])
        # Load PyTorch checkpoint
        load_torch_weights(self.model, torch.load(self.args.pytorch_checkpoint, map_location='cpu'))

    def export(self):
        self.model.save(self.args.tensorflow_output)


if __name__ == '__main__':
    Exporter()