"""
python export_coreml.py \
    --model-variant mobilenetv3 \
    --checkpoint rvm_mobilenetv3.pth \
    --resolution 1920 1080 \
    --downsample-ratio 0.25 \
    --quantize-nbits 16 \
    --output model.mlmodel
"""


import argparse
import coremltools as ct
import torch

from coremltools.models.neural_network.quantization_utils import quantize_weights
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.proto import FeatureTypes_pb2 as ft        

from model import MattingNetwork

class Exporter:
    def __init__(self):
        self.parse_args()
        self.init_model()
        self.register_custom_ops()
        self.export()
    
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model-variant', type=str, required=True, choices=['mobilenetv3', 'resnet50'])
        parser.add_argument('--model-refiner', type=str, default='deep_guided_filter', choices=['deep_guided_filter', 'fast_guided_filter'])
        parser.add_argument('--checkpoint', type=str, required=False)
        parser.add_argument('--resolution', type=int, required=True, nargs=2)
        parser.add_argument('--downsample-ratio', type=float, required=True)
        parser.add_argument('--quantize-nbits', type=int, required=True, choices=[8, 16, 32])
        parser.add_argument('--output', type=str, required=True)
        self.args = parser.parse_args()
        
    def init_model(self):
        downsample_ratio = self.args.downsample_ratio
        class Wrapper(MattingNetwork):
            def forward(self, src, r1=None, r2=None, r3=None, r4=None):
                # Hardcode downsample_ratio into the network instead of taking it as input. This is needed for torchscript tracing.
                # Also, we are multiply result by 255 to convert them to CoreML image format
                fgr, pha, r1, r2, r3, r4 = super().forward(src, r1, r2, r3, r4, downsample_ratio)
                return fgr.mul(255), pha.mul(255), r1, r2, r3, r4
        
        self.model = Wrapper(self.args.model_variant, self.args.model_refiner).eval()
        if self.args.checkpoint is not None:
            self.model.load_state_dict(torch.load(self.args.checkpoint, map_location='cpu'), strict=False)
    
    def register_custom_ops(self):
        @register_torch_op(override=True)
        def hardswish_(context, node):
            inputs = _get_inputs(context, node, expected=1)
            x = inputs[0]
            y = mb.sigmoid_hard(x=inputs[0], alpha=1.0/6, beta=0.5)
            z = mb.mul(x=x, y=y, name=node.name)
            context.add(z)

        @register_torch_op(override=True)
        def hardsigmoid_(context, node):
            inputs = _get_inputs(context, node, expected=1)
            res = mb.sigmoid_hard(x=inputs[0], alpha=1.0/6, beta=0.5, name=node.name)
            context.add(res)

        @register_torch_op(override=True)
        def type_as(context, node):
            inputs = _get_inputs(context, node)
            context.add(mb.cast(x=inputs[0], dtype='fp32'), node.name)

        @register_torch_op(override=True)
        def upsample_bilinear2d(context, node):
            # Change to use `resize_bilinear` instead to support iOS 13.
            inputs = _get_inputs(context, node)
            x = inputs[0]
            output_size = inputs[1]
            align_corners = bool(inputs[2].val)
            scale_factors = inputs[3]

            if scale_factors is not None and scale_factors.val is not None \
                    and scale_factors.rank == 1 and scale_factors.shape[0] == 2:
                scale_factors = scale_factors.val
                resize = mb.resize_bilinear(
                    x=x,
                    target_size_height=int(x.shape[-2] * scale_factors[0]),
                    target_size_width=int(x.shape[-1] * scale_factors[1]),
                    sampling_mode='ALIGN_CORNERS',
                    name=node.name,
                )
                context.add(resize)
            else:
                resize = mb.resize_bilinear(
                    x=x,
                    target_size_height=output_size.val[0],
                    target_size_width=output_size.val[1],
                    sampling_mode='ALIGN_CORNERS',
                    name=node.name,
                )
                context.add(resize)
            
    def export(self):
        src = torch.zeros([1, 3, *self.args.resolution[::-1]])
        _, _, r1, r2, r3, r4 = self.model(src)
        
        model_traced = torch.jit.trace(self.model, (src, r1, r2, r3, r4))
        model_coreml = ct.convert(
            model_traced,
            inputs=[
                ct.ImageType(name='src', shape=(ct.RangeDim(), *src.shape[1:]), channel_first=True, scale=1/255),
                ct.TensorType(name='r1i', shape=(ct.RangeDim(), *r1.shape[1:])),
                ct.TensorType(name='r2i', shape=(ct.RangeDim(), *r2.shape[1:])),
                ct.TensorType(name='r3i', shape=(ct.RangeDim(), *r3.shape[1:])),
                ct.TensorType(name='r4i', shape=(ct.RangeDim(), *r4.shape[1:])),
            ],
        )
    
        if self.args.quantize_nbits in [8, 16]:
            out = quantize_weights(model_coreml, nbits=self.args.quantize_nbits)
            if isinstance(out, ct.models.model.MLModel):
                # When the export is done on OSX, return is an mlmodel.
                spec = out.get_spec()
            else:
                # When the export is done on Linux, the return is a spec. 
                spec = out
        else:
            spec = model_coreml.get_spec()
        
        # Some internal outputs are also named 'fgr' and 'pha'. 
        # We change them to avoid conflicts.
        for layer in spec.neuralNetwork.layers:
            for i in range(len(layer.input)):
                if layer.input[i] == 'fgr':
                    layer.input[i] = 'fgr_internal'
                if layer.input[i] == 'pha':
                    layer.input[i] = 'pha_internal'
            for i in range(len(layer.output)):
                if layer.output[i] == 'fgr':
                    layer.output[i] = 'fgr_internal'
                if layer.output[i] == 'pha':
                    layer.output[i] = 'pha_internal'
        
        # Update output names
        ct.utils.rename_feature(spec, spec.description.output[0].name, 'fgr')
        ct.utils.rename_feature(spec, spec.description.output[1].name, 'pha')
        ct.utils.rename_feature(spec, spec.description.output[2].name, 'r1o')
        ct.utils.rename_feature(spec, spec.description.output[3].name, 'r2o')
        ct.utils.rename_feature(spec, spec.description.output[4].name, 'r3o')
        ct.utils.rename_feature(spec, spec.description.output[5].name, 'r4o')
        
        # Update model description
        spec.description.metadata.author = 'Shanchuan Lin'
        spec.description.metadata.shortDescription = 'A robust human video matting model with recurrent architecture. The model has recurrent states that must be passed to subsequent frames. Please refer to paper "Robust High-Resolution Video Matting with Temporal Guidance" for more details.'
        spec.description.metadata.license = 'Apache License 2.0'
        spec.description.metadata.versionString = '1.0.0'
        spec.description.input[0].shortDescription = 'Source frame'
        spec.description.input[1].shortDescription = 'Recurrent state 1. Initial state is an all zero tensor. Subsequent state is received from r1o.'
        spec.description.input[2].shortDescription = 'Recurrent state 2. Initial state is an all zero tensor. Subsequent state is received from r2o.'
        spec.description.input[3].shortDescription = 'Recurrent state 3. Initial state is an all zero tensor. Subsequent state is received from r3o.'
        spec.description.input[4].shortDescription = 'Recurrent state 4. Initial state is an all zero tensor. Subsequent state is received from r4o.'
        spec.description.output[0].shortDescription = 'Foreground prediction'
        spec.description.output[1].shortDescription = 'Alpha prediction'
        spec.description.output[2].shortDescription = 'Recurrent state 1. Needs to be passed as r1i input in the next time step.'
        spec.description.output[3].shortDescription = 'Recurrent state 2. Needs to be passed as r2i input in the next time step.'
        spec.description.output[4].shortDescription = 'Recurrent state 3. Needs to be passed as r3i input in the next time step.'
        spec.description.output[5].shortDescription = 'Recurrent state 4. Needs to be passed as r4i input in the next time step.'

        # Update output types
        spec.description.output[0].type.imageType.colorSpace = ft.ImageFeatureType.RGB
        spec.description.output[0].type.imageType.width = src.size(3)
        spec.description.output[0].type.imageType.height = src.size(2)
        spec.description.output[1].type.imageType.colorSpace = ft.ImageFeatureType.GRAYSCALE
        spec.description.output[1].type.imageType.width = src.size(3)
        spec.description.output[1].type.imageType.height = src.size(2)

        # Set recurrent states as optional inputs
        spec.description.input[1].type.isOptional = True
        spec.description.input[2].type.isOptional = True
        spec.description.input[3].type.isOptional = True
        spec.description.input[4].type.isOptional = True
        
        # Save output
        ct.utils.save_spec(spec, self.args.output)
        
        
        
        
if __name__ == '__main__':
    Exporter()