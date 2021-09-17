"""
python export_onnx.py \
    --model-variant mobilenetv3 \
    --checkpoint rvm_mobilenetv3.pth \
    --precision float16 \
    --opset 12 \
    --device cuda \
    --output model.onnx
    
Note:
    The device is only used for exporting. It has nothing to do with the final model.
    Float16 must be exported through cuda. Float32 can be exported through cpu.
"""

import argparse
import torch

from model import MattingNetwork


class Exporter:
    def __init__(self):
        self.parse_args()
        self.init_model()
        self.export()
        
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model-variant', type=str, required=True, choices=['mobilenetv3', 'resnet50'])
        parser.add_argument('--model-refiner', type=str, default='deep_guided_filter', choices=['deep_guided_filter', 'fast_guided_filter'])
        parser.add_argument('--precision', type=str, required=True, choices=['float16', 'float32'])
        parser.add_argument('--opset', type=int, required=True)
        parser.add_argument('--device', type=str, required=True)
        parser.add_argument('--checkpoint', type=str, required=False)
        parser.add_argument('--output', type=str, required=True)
        self.args = parser.parse_args()
        
    def init_model(self):
        self.precision = torch.float32 if self.args.precision == 'float32' else torch.float16
        self.model = MattingNetwork(self.args.model_variant, self.args.model_refiner).eval().to(self.args.device, self.precision)
        if self.args.checkpoint is not None:
            self.model.load_state_dict(torch.load(self.args.checkpoint, map_location=self.args.device), strict=False)
        
    def export(self):
        rec = (torch.zeros([1, 1, 1, 1]).to(self.args.device, self.precision),) * 4
        src = torch.randn(1, 3, 1080, 1920).to(self.args.device, self.precision)
        downsample_ratio = torch.tensor([0.25]).to(self.args.device)
        
        dynamic_spatial = {0: 'batch_size', 2: 'height', 3: 'width'}
        dynamic_everything = {0: 'batch_size', 1: 'channels', 2: 'height', 3: 'width'}
        
        torch.onnx.export(
            self.model,
            (src, *rec, downsample_ratio),
            self.args.output,
            export_params=True,
            opset_version=self.args.opset,
            do_constant_folding=True,
            input_names=['src', 'r1i', 'r2i', 'r3i', 'r4i', 'downsample_ratio'],
            output_names=['fgr', 'pha', 'r1o', 'r2o', 'r3o', 'r4o'],
            dynamic_axes={
                'src': dynamic_spatial,
                'fgr': dynamic_spatial,
                'pha': dynamic_spatial,
                'r1i': dynamic_everything,
                'r2i': dynamic_everything,
                'r3i': dynamic_everything,
                'r4i': dynamic_everything,
                'r1o': dynamic_spatial,
                'r2o': dynamic_spatial,
                'r3o': dynamic_spatial,
                'r4o': dynamic_spatial,
            })

if __name__ == '__main__':
    Exporter()