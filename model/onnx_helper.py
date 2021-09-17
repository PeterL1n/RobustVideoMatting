import torch
from torch import Tensor
from torch.nn import functional as F
from torch.autograd import Function

"""
We implement custom ONNX export logics because PyTorch doesn't trace the use of "Shape" op very well.
The custom export logics add support for runtime downsample_ratio input, and clean up the ONNX graph.
"""

class CustomOnnxResizeByFactorOp(Function):
    """
    This implements resize by scale_factor. Unlike PyTorch which can only export the scale_factor is a hardcoded int,
    we implement it such that the scale_factor can be a tensor provided at runtime.
    """
    
    @staticmethod
    def forward(ctx, x, scale_factor):
        assert x.ndim == 4
        return F.interpolate(x, scale_factor=scale_factor.item(), 
                             mode='bilinear', recompute_scale_factor=False, align_corners=False)
    
    @staticmethod
    def symbolic(g, x, scale_factor):
        empty_roi = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))
        scale_factor = g.op("Concat", 
                            g.op("Constant", value_t=torch.tensor([1, 1], dtype=torch.float32)),
                            scale_factor, scale_factor,  axis_i=0)
        return g.op('Resize',
            x,
            empty_roi,
            scale_factor,
            coordinate_transformation_mode_s='pytorch_half_pixel',
            cubic_coeff_a_f=-0.75,
            mode_s='linear',
            nearest_mode_s="floor")
    
class CustomOnnxResizeToMatchSizeOp(Function):
    """
    This implements bilinearly resize a tensor to match the size of another.
    This implementation has cleaner ONNX graph than PyTorch's default export.
    """
    @staticmethod
    def forward(ctx, x, y):
        assert x.ndim == 4 and y.ndim == 4
        return F.interpolate(x, y.shape[2:], mode='bilinear', align_corners=False)
    
    @staticmethod
    def symbolic(g, x, y):
        empty_roi = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))
        empty_scales = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))
        BC = g.op('Slice',
            g.op('Shape', x), # input
            g.op('Constant', value_t=torch.tensor([0])), # starts
            g.op('Constant', value_t=torch.tensor([2])), # ends
            g.op('Constant', value_t=torch.tensor([0])), # axes
        )
        HW = g.op('Slice',
            g.op('Shape', y), # input
            g.op('Constant', value_t=torch.tensor([2])), # starts
            g.op('Constant', value_t=torch.tensor([4])), # ends
            g.op('Constant', value_t=torch.tensor([0])), # axes
        )
        output_shape = g.op('Concat', BC, HW, axis_i=0)
        return g.op('Resize',
            x,
            empty_roi,
            empty_scales,
            output_shape,
            coordinate_transformation_mode_s='pytorch_half_pixel',
            cubic_coeff_a_f=-0.75,
            mode_s='linear',
            nearest_mode_s="floor")

class CustomOnnxCropToMatchSizeOp(Function):
    """
    This implements cropping a tensor to match the size of another.
    This implementation has cleaner ONNX graph than PyTorch's default export.
    """
    @staticmethod
    def forward(ctx, x, y):
        assert x.ndim == 4 and y.ndim == 4
        return x[:, :, :y.size(2), :y.size(3)]
    
    @staticmethod
    def symbolic(g, x, y):
        size = g.op('Slice',
            g.op('Shape', y), # input
            g.op('Constant', value_t=torch.tensor([2])), # starts
            g.op('Constant', value_t=torch.tensor([4])), # ends
            g.op('Constant', value_t=torch.tensor([0])), # axes
        )
        return g.op('Slice', 
            x, # input
            g.op('Constant', value_t=torch.tensor([0, 0])), # starts
            size, # ends
            g.op('Constant', value_t=torch.tensor([2, 3])), # axes
        )
