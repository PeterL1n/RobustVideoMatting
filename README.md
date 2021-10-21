# Export ONNX

## Overview

This branch contains our modified model code specifically for ONNX export. We made a few modifications because directly exporting with PyTorch 1.9 has several limitations.

* PyTorch only allows `F.interpolate(x, scale_factor)` to accept `scale_factor` as `float` but not `tensor`. This makes the value hardcoded into the ONNX graph. We modify downsampling to take scale factor as user provided tensor, such that the `downsample_ratio` hyperparameter can configured at runtime.

* PyTorch does not trace `Tensor.Shape` very well. It creates a messy graph. We customize it so that the graph is the cleanest.

Our custom export logis are implemented in `model/onnx_helper.py`

## Export Yourself

The following procedures were used to generate our ONNX models.

1. Install dependencies
```sh
pip install -r requirements.txt
```

2. (Only for PyTorch <= 1.9) A few modifications to the PyTorch source. This is needed before pull request [#60080](https://github.com/pytorch/pytorch/pull/60080) is merged into later version of PyTorch. If you are exporting MobileNetV3 variant, go to your local PyTorch install and override the following method to file `site-packages/torch/onnx/symbolic_opset9.py`. This allows export of `hardswish` as native ops.

Also note, if your inference backend does not support `hardswish` or `hardsigmoid`. You can also use this hack to replace them with primitive ops.

```python
@parse_args("v")
def hardswish(g, self):
    hardsigmoid = g.op('HardSigmoid', self, alpha_f=1 / 6)
    return g.op("Mul", self, hardsigmoid)
```

3. Use the export script. The `device` argument is only for export tracing. Float16 must be exported using a cuda device. Our export script only support opset 11 and up. If you need older opset support. You must adapt the code yourself.
```sh
python export_onnx.py \
    --model-variant mobilenetv3 \
    --checkpoint rvm_mobilenetv3.pth \
    --precision float16 \
    --opset 12 \
    --device cuda \
    --output model.onnx
```

## Additional

Our model is tested to work on ONNX Runtime's CPU and CUDA backends. If your inference backend has compatibility issue to certain ops, you can file an issue on GitHub, but we don't guarantee solutions. Feel free to write your own export code that fits your need.
