# Inference

<p align="center">English | <a href="inference_zh_Hans.md">中文</a></p>

## Content

* [Concepts](#concepts)
    * [Downsample Ratio](#downsample-ratio)
    * [Recurrent States](#recurrent-states)
* [PyTorch](#pytorch)
* [TorchHub](#torchhub)
* [TorchScript](#torchscript)
* [ONNX](#onnx)
* [TensorFlow](#tensorflow)
* [TensorFlow.js](#tensorflowjs)
* [CoreML](#coreml)

<br>


## Concepts

### Downsample Ratio

The table provides a general guideline. Please adjust based on your video content.

| Resolution    | Portrait      | Full-Body      |
| ------------- | ------------- | -------------- |
| <= 512x512    | 1             | 1              |
| 1280x720      | 0.375         | 0.6            |
| 1920x1080     | 0.25          | 0.4            |
| 3840x2160     | 0.125         | 0.2            |

Internally, the model resizes down the input for stage 1. Then, it refines at high-resolution for stage 2.

Set `downsample_ratio` so that the downsampled resolution is between 256 and 512. For example, for `1920x1080` input with `downsample_ratio=0.25`, the resized resolution `480x270` is between 256 and 512.

Adjust `downsample_ratio` base on the video content. If the shot is portrait, a lower `downsample_ratio` is sufficient. If the shot contains the full human body, use high `downsample_ratio`. Note that higher `downsample_ratio` is not always better.


<br>

### Recurrent States
The model is a recurrent neural network. You must process frames sequentially and recycle its recurrent states. 

**Correct Way**

The recurrent outputs are recycled back as input when processing the next frame. The states are essentially the model's memory.

```python
rec = [None] * 4  # Initial recurrent states are None

for frame in YOUR_VIDEO:
    fgr, pha, *rec = model(frame, *rec, downsample_ratio)
```

**Wrong Way**

The model does not utilize the recurrent states. Only use it to process independent images.

```python
for frame in YOUR_VIDEO:
    fgr, pha = model(frame, downsample_ratio)[:2]
```

More technical details are in the [paper](https://peterl1n.github.io/RobustVideoMatting/).

<br><br><br>


## PyTorch

Model loading:

```python
import torch
from model import MattingNetwork

model = MattingNetwork(variant='mobilenetv3').eval().cuda() # Or variant="resnet50"
model.load_state_dict(torch.load('rvm_mobilenetv3.pth'))
```

Example inference loop:
```python
rec = [None] * 4 # Set initial recurrent states to None

for src in YOUR_VIDEO:  # src can be [B, C, H, W] or [B, T, C, H, W]
    fgr, pha, *rec = model(src, *rec, downsample_ratio=0.25)
```

* `src`: Input frame. 
    * Can be of shape `[B, C, H, W]` or `[B, T, C, H, W]`. 
    * If `[B, T, C, H, W]`, a chunk of `T` frames can be given at once for better parallelism.
    * RGB input is normalized to `0~1` range.

* `fgr, pha`: Foreground and alpha predictions. 
    * Can be of shape `[B, C, H, W]` or `[B, T, C, H, W]` depends on `src`. 
    * `fgr` has `C=3` for RGB, `pha` has `C=1`.
    * Outputs normalized to `0~1` range.
* `rec`: Recurrent states. 
    * Type of `List[Tensor, Tensor, Tensor, Tensor]`. 
    * Initial `rec` can be `List[None, None, None, None]`.
    * It has 4 recurrent states because the model has 4 ConvGRU layers.
    * All tensors are rank 4 regardless of `src` rank.
    * If a chunk of `T` frames is given, only the last frame's recurrent states will be returned.

To inference on video, here is a complete example:

```python
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from inference_utils import VideoReader, VideoWriter

reader = VideoReader('input.mp4', transform=ToTensor())
writer = VideoWriter('output.mp4', frame_rate=30)

bgr = torch.tensor([.47, 1, .6]).view(3, 1, 1).cuda()  # Green background.
rec = [None] * 4                                       # Initial recurrent states.

with torch.no_grad():
    for src in DataLoader(reader):
        fgr, pha, *rec = model(src.cuda(), *rec, downsample_ratio=0.25)  # Cycle the recurrent states.
        writer.write(fgr * pha + bgr * (1 - pha))
```

Or you can use the provided video converter:

```python
from inference import convert_video

convert_video(
    model,                           # The loaded model, can be on any device (cpu or cuda).
    input_source='input.mp4',        # A video file or an image sequence directory.
    input_resize=(1920, 1080),       # [Optional] Resize the input (also the output).
    downsample_ratio=0.25,           # [Optional] If None, make downsampled max size be 512px.
    output_type='video',             # Choose "video" or "png_sequence"
    output_composition='com.mp4',    # File path if video; directory path if png sequence.
    output_alpha="pha.mp4",          # [Optional] Output the raw alpha prediction.
    output_foreground="fgr.mp4",     # [Optional] Output the raw foreground prediction.
    output_video_mbps=4,             # Output video mbps. Not needed for png sequence.
    seq_chunk=12,                    # Process n frames at once for better parallelism.
    num_workers=1,                   # Only for image sequence input. Reader threads.
    progress=True                    # Print conversion progress.
)
```

The converter can also be invoked in command line:

```sh
python inference.py \
    --variant mobilenetv3 \
    --checkpoint "CHECKPOINT" \
    --device cuda \
    --input-source "input.mp4" \
    --downsample-ratio 0.25 \
    --output-type video \
    --output-composition "composition.mp4" \
    --output-alpha "alpha.mp4" \
    --output-foreground "foreground.mp4" \
    --output-video-mbps 4 \
    --seq-chunk 12
```

<br><br><br>

## TorchHub

Model loading:

```python
model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3") # or "resnet50"
```

Use the conversion function. Refer to the documentation for `convert_video` function above.

```python
convert_video = torch.hub.load("PeterL1n/RobustVideoMatting", "converter")

convert_video(model, ...args...)
```

<br><br><br>

## TorchScript

Model loading:

```python
import torch
model = torch.jit.load('rvm_mobilenetv3.torchscript')
```

Optionally, freeze the model. This will trigger graph optimization, such as BatchNorm fusion etc. Frozen models are faster.

```python
model = torch.jit.freeze(model)
```

Then, you can use the `model` exactly the same as a PyTorch model, with the exception that you must manually provide `device` and `dtype` to the converter API for frozen model. For example:

```python
convert_video(frozen_model, ...args..., device='cuda', dtype=torch.float32)
```

<br><br><br>

## ONNX

Model spec:
* Inputs: [`src`, `r1i`, `r2i`, `r3i`, `r4i`, `downsample_ratio`]. 
    * `src` is the RGB input frame of shape `[B, C, H, W]` normalized to `0~1` range. 
    * `rXi` are the recurrent state inputs. Initial recurrent states are zero value tensors of shape `[1, 1, 1, 1]`.
    * `downsample_ratio` is a tensor of shape `[1]`.
    * Only `downsample_ratio` must have `dtype=FP32`. Other inputs must have `dtype` matching the loaded model's precision.
* Outputs: [`fgr`, `pha`, `r1o`, `r2o`, `r3o`, `r4o`]
    * `fgr, pha` are the foreground and alpha prediction. Normalized to `0~1` range.
    * `rXo` are the recurrent state outputs.

We only show examples of using onnxruntime CUDA backend in Python.

Model loading

```python
import onnxruntime as ort

sess = ort.InferenceSession('rvm_mobilenetv3_fp16.onnx')
```

Naive inference loop

```python
import numpy as np

rec = [ np.zeros([1, 1, 1, 1], dtype=np.float16) ] * 4  # Must match dtype of the model.
downsample_ratio = np.array([0.25], dtype=np.float32)  # dtype always FP32

for src in YOUR_VIDEO:  # src is of [B, C, H, W] with dtype of the model.
    fgr, pha, *rec = sess.run([], {
        'src': src, 
        'r1i': rec[0], 
        'r2i': rec[1], 
        'r3i': rec[2], 
        'r4i': rec[3], 
        'downsample_ratio': downsample_ratio
    })
```

If you use GPU version of ONNX Runtime, the above naive implementation has recurrent states transferred between CPU and GPU on every frame. They could have just stayed on the GPU for better performance. Below is an example using `iobinding` to eliminate useless transfers.

```python
import onnxruntime as ort
import numpy as np

# Load model.
sess = ort.InferenceSession('rvm_mobilenetv3_fp16.onnx')

# Create an io binding.
io = sess.io_binding()

# Create tensors on CUDA.
rec = [ ort.OrtValue.ortvalue_from_numpy(np.zeros([1, 1, 1, 1], dtype=np.float16), 'cuda') ] * 4
downsample_ratio = ort.OrtValue.ortvalue_from_numpy(np.asarray([0.25], dtype=np.float32), 'cuda')

# Set output binding.
for name in ['fgr', 'pha', 'r1o', 'r2o', 'r3o', 'r4o']:
    io.bind_output(name, 'cuda')

# Inference loop
for src in YOUR_VIDEO:
    io.bind_cpu_input('src', src)
    io.bind_ortvalue_input('r1i', rec[0])
    io.bind_ortvalue_input('r2i', rec[1])
    io.bind_ortvalue_input('r3i', rec[2])
    io.bind_ortvalue_input('r4i', rec[3])
    io.bind_ortvalue_input('downsample_ratio', downsample_ratio)

    sess.run_with_iobinding(io)

    fgr, pha, *rec = io.get_outputs()

    # Only transfer `fgr` and `pha` to CPU.
    fgr = fgr.numpy()
    pha = pha.numpy()
```

Note: depending on the inference tool you choose, it may not support all the operations in our official ONNX model. You are responsible for modifying the model code and exporting your own ONNX model. You can refer to our exporter code in the [onnx branch](https://github.com/PeterL1n/RobustVideoMatting/tree/onnx).

<br><br><br>

### TensorFlow

An example usage:

```python
import tensorflow as tf

model = tf.keras.models.load_model('rvm_mobilenetv3_tf')
model = tf.function(model)

rec = [ tf.constant(0.) ] * 4         # Initial recurrent states.
downsample_ratio = tf.constant(0.25)  # Adjust based on your video.

for src in YOUR_VIDEO:  # src is of shape [B, H, W, C], not [B, C, H, W]!
    out = model([src, *rec, downsample_ratio])
    fgr, pha, *rec = out['fgr'], out['pha'], out['r1o'], out['r2o'], out['r3o'], out['r4o']
```

Note the the tensors are all channel last. Otherwise, the inputs and outputs are exactly the same as PyTorch.

We also provide the raw TensorFlow model code in the [tensorflow branch](https://github.com/PeterL1n/RobustVideoMatting/tree/tensorflow). You can transfer PyTorch checkpoint weights to TensorFlow models.

<br><br><br>

### TensorFlow.js

We provide a starter code in the [tfjs branch](https://github.com/PeterL1n/RobustVideoMatting/tree/tfjs). The example is very self-explanatory. It shows how to properly use the model.

<br><br><br>

### CoreML

We only show example usage of the CoreML models in Python API using `coremltools`. In production, the same logic can be applied in Swift. When processing the first frame, do not provide recurrent states. CoreML will internally construct zero tensors of the correct shapes as the initial recurrent states.

```python
import coremltools as ct

model = ct.models.model.MLModel('rvm_mobilenetv3_1920x1080_s0.25_int8.mlmodel')

r1, r2, r3, r4 = None, None, None, None

for src in YOUR_VIDEO:  # src is PIL.Image.
    
    if r1 is None:
        # Initial frame, do not provide recurrent states.
        inputs = {'src': src}
    else:
        # Subsequent frames, provide recurrent states.
        inputs = {'src': src, 'r1i': r1, 'r2i': r2, 'r3i': r3, 'r4i': r4}

    outputs = model.predict(inputs)

    fgr = outputs['fgr']  # PIL.Image.
    pha = outputs['pha']  # PIL.Image.
    
    r1 = outputs['r1o']  # Numpy array.
    r2 = outputs['r2o']  # Numpy array.
    r3 = outputs['r3o']  # Numpy array.
    r4 = outputs['r4o']  # Numpy array.

```

Our CoreML models only support fixed resolutions. If you need other resolutions, you can export them yourself. See [coreml branch](https://github.com/PeterL1n/RobustVideoMatting/tree/coreml) for model export. 