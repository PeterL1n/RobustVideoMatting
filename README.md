# TensorFlow Implementation

## Overview

This branch contains our model implemented in TensorFlow 2. We have transferred the weights from the official PyTorch version and provide the model in TensorFlow SavedModel format. If you only need inference, you do not need any source code from this branch. If you need the weights in native TensorFlow model for advanced experiments, we show you how to load weights from PyTorch.

## Transfer PyTorch Weights to TensorFlow

```python
import tensorflow as tf
import torch

from model import MattingNetwork, load_torch_weights

# Create a TF model
model = MattingNetwork('mobilenetv3')

# Create dummpy inputs.
src = tf.random.normal([1, 1080, 1920, 3])
rec = [ tf.constant(0.) ] * 4
downsample_ratio = tf.constant(0.25)

# Do a forward pass to initialize the model.
out = model([src, *rec, downsample_ratio])

# Transfer PyTorch weights to TF model.
state_dict = torch.load('rvm_mobilenetv3.pth')
load_torch_weights(model, state_dict)
```

## Export TensorFlow SavedModel

We use the following script to generate the official TensorFlow SavedModel from the PyTorch checkpoint.

```sh
python export_tensorflow.py \
    --model-variant mobilenetv3 \
    --model-refiner deep_guided_filter \
    --pytorch-checkpoint rvm_mobilenetv3.pth \
    --tensorflow-output rvm_mobilenetv3_tf
```

## Export TensorFlow.js

We already provide an exported TensorFlow.js model. If you need other configurations, use the export procedure below.

Currently TensorFlow.js only supports Fast Guided Filter. To export to tfjs, first use the `export_tensorflow.py` script above with `--model-refiner fast_guided_filter` to generate a TensorFlow SavedModel. Then convert the SavedModel to tfjs:

```sh
pip install tensorflowjs

tensorflowjs_converter \
    --quantize_uint8 \
    --input_format=tf_saved_model \
    rvm_mobilenetv3_tf \
    rvm_mobilenetv3_tfjs_int8
```