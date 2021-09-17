# Export CoreML

## Overview

This branch contains our code to export to CoreML models. The `/model` folder is the same as the `master` branch. The main exporting logics are in `export_coreml.py`.

At the time of this writing, CoreML's `ResizeBilinear` and `Upsample` ops don't not support dynamic scale parameters, so the `downsample_ratio` hyperparameter must be hardcoded.

Our export script is written to have input size fixed. The output coreml models require iOS14+, MacOS11+. If you have other requirements, feel free to modify the export script. Contributions are welcomed.

## Export Yourself

The following procedures were used to generate our CoreML models.

1. Install dependencies
```sh
pip install -r requirements.txt
```

2. Use the export script. You can change the `resolution` and `downsample-ratio` to fit your need. You can change quantization to one of `[8, 16, 32]`, denoting `int8`, `fp16`, and `fp32`.
```sh
python export_coreml.py \
    --model-variant mobilenetv3 \
    --checkpoint rvm_mobilenetv3.pth \
    --resolution 1920 1080 \
    --downsample-ratio 0.25 \
    --quantize-nbits 16 \
    --output model.mlmodel
```