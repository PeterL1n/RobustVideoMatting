# 推断文档

<p align="center"><a href="inference.md">English</a> | 中文</p>

## 目录

* [概念](#概念)
    * [下采样比](#下采样比)
    * [循环记忆](#循环记忆)
* [PyTorch](#pytorch)
* [TorchHub](#torchhub)
* [TorchScript](#torchscript)
* [ONNX](#onnx)
* [TensorFlow](#tensorflow)
* [TensorFlow.js](#tensorflowjs)
* [CoreML](#coreml)

<br>


## 概念

### 下采样比

该表仅供参考。可根据视频内容进行调节。

| 分辨率         | 人像           | 全身            |
| ------------- | ------------- | -------------- |
| <= 512x512    | 1             | 1              |
| 1280x720      | 0.375         | 0.6            |
| 1920x1080     | 0.25          | 0.4            |
| 3840x2160     | 0.125         | 0.2            |

模型在内部将高分辨率输入缩小做初步的处理，然后再放大做细分处理。

建议设置 `downsample_ratio` 使缩小后的分辨率维持在 256 到 512 像素之间. 例如，`1920x1080` 的输入用 `downsample_ratio=0.25`，缩小后的分辨率 `480x270` 在 256 到 512 像素之间。

根据视频内容调整 `downsample_ratio`。若视频是上身人像，低 `downsample_ratio` 足矣。若视频是全身像，建议尝试更高的 `downsample_ratio`。但注意，过高的 `downsample_ratio` 反而会降低效果。


<br>

### 循环记忆
此模型是循环神经网络（Recurrent Neural Network）。必须按顺序处理视频每帧，并提供网络循环记忆。

**正确用法**

循环记忆输出被传递到下一帧做输入。

```python
rec = [None] * 4  # 初始值设置为 None

for frame in YOUR_VIDEO:
    fgr, pha, *rec = model(frame, *rec, downsample_ratio)
```

**错误用法**

没有使用循环记忆。此方法仅可用于处理单独的图片。

```python
for frame in YOUR_VIDEO:
    fgr, pha = model(frame, downsample_ratio)[:2]
```

更多技术细节见[论文](https://peterl1n.github.io/RobustVideoMatting/)。

<br><br><br>


## PyTorch

载入模型：

```python
import torch
from model import MattingNetwork

model = MattingNetwork(variant='mobilenetv3').eval().cuda() # 或 variant="resnet50"
model.load_state_dict(torch.load('rvm_mobilenetv3.pth'))
```

推断循环：
```python
rec = [None] * 4 # 初始值设置为 None

for src in YOUR_VIDEO:  # src 可以是 [B, C, H, W] 或 [B, T, C, H, W]
    fgr, pha, *rec = model(src, *rec, downsample_ratio=0.25)
```

* `src`: 输入帧（Source）。 
    * 可以是 `[B, C, H, W]` 或 `[B, T, C, H, W]` 的张量。 
    * 若是 `[B, T, C, H, W]`，可给模型一次 `T` 帧，做一小段一小段地处理，用于更好的并行计算。
    * RGB 通道输入，范围为 `0~1`。

* `fgr, pha`: 前景（Foreground）和透明度通道（Alpha）的预测。 
    * 根据`src`，可为 `[B, C, H, W]` 或 `[B, T, C, H, W]` 的输出。
    * `fgr` 是 RGB 三通道，`pha` 为一通道。
    * 输出范围为 `0~1`。
* `rec`: 循环记忆（Recurrent States）。 
    * `List[Tensor, Tensor, Tensor, Tensor]` 类型。 
    * 初始 `rec` 为 `List[None, None, None, None]`。
    * 有四个记忆，因为网络使用四个 `ConvGRU` 层。
    * 无论 `src` 的 Rank，所有记忆张量的 Rank 为 4。
    * 若一次给予 `T` 帧，只返回处理完最后一帧后的记忆。

完整的推断例子：

```python
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from inference_utils import VideoReader, VideoWriter

reader = VideoReader('input.mp4', transform=ToTensor())
writer = VideoWriter('output.mp4', frame_rate=30)

bgr = torch.tensor([.47, 1, .6]).view(3, 1, 1).cuda()  # 绿背景
rec = [None] * 4                                       # 初始记忆

with torch.no_grad():
    for src in DataLoader(reader):
        fgr, pha, *rec = model(src.cuda(), *rec, downsample_ratio=0.25)  # 将上一帧的记忆给下一帧
        writer.write(fgr * pha + bgr * (1 - pha))
```

或者使用提供的视频转换 API：

```python
from inference import convert_video

convert_video(
    model,                           # 模型，可以加载到任何设备（cpu 或 cuda）
    input_source='input.mp4',        # 视频文件，或图片序列文件夹
    input_resize=(1920, 1080),       # [可选项] 缩放视频大小
    downsample_ratio=0.25,           # [可选项] 下采样比，若 None，自动下采样至 512px
    output_type='video',             # 可选 "video"（视频）或 "png_sequence"（PNG 序列）
    output_composition='com.mp4',    # 若导出视频，提供文件路径。若导出 PNG 序列，提供文件夹路径
    output_alpha="pha.mp4",          # [可选项] 输出透明度预测
    output_foreground="fgr.mp4",     # [可选项] 输出前景预测
    output_video_mbps=4,             # 若导出视频，提供视频码率
    seq_chunk=12,                    # 设置多帧并行计算
    num_workers=1,                   # 只适用于图片序列输入，读取线程
    progress=True                    # 显示进度条
)
```

也可通过命令行调用转换 API：

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

载入模型：

```python
model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3") # or "resnet50"
```

使用转换 API，具体请参考之前对 `convert_video` 的文档。

```python
convert_video = torch.hub.load("PeterL1n/RobustVideoMatting", "converter")

convert_video(model, ...args...)
```

<br><br><br>

## TorchScript

载入模型：

```python
import torch
model = torch.jit.load('rvm_mobilenetv3.torchscript')
```

也可以可选的将模型固化（Freeze）。这会对模型进行优化，例如 BatchNorm Fusion 等。固化的模型更快。

```python
model = torch.jit.freeze(model)
```

然后，可以将 `model` 作为普通的 PyTorch 模型使用。但注意，若用固化模型调用转换 API，必须手动提供 `device` 和 `dtype`:

```python
convert_video(frozen_model, ...args..., device='cuda', dtype=torch.float32)
```

<br><br><br>

## ONNX

模型规格:
* 输入: [`src`, `r1i`, `r2i`, `r3i`, `r4i`, `downsample_ratio`]. 
    * `src`：输入帧，RGB 通道，形状为 `[B, C, H, W]`，范围为`0~1`。
    * `rXi`：记忆输入，初始值是是形状为 `[1, 1, 1, 1]` 的零张量。
    * `downsample_ratio` 下采样比，张量形状为 `[1]`。
    * 只有 `downsample_ratio` 必须是 `FP32`，其他输入必须和加载的模型使用一样的 `dtype`。
* 输出: [`fgr`, `pha`, `r1o`, `r2o`, `r3o`, `r4o`]
    * `fgr, pha`：前景和透明度通道输出，范围为 `0~1`。
    * `rXo`：记忆输出。

我们只展示用 ONNX Runtime CUDA Backend 在 Python 上的使用范例。

载入模型：

```python
import onnxruntime as ort

sess = ort.InferenceSession('rvm_mobilenetv3_fp16.onnx')
```

简单推断循环，但此方法不是最优化的：

```python
import numpy as np

rec = [ np.zeros([1, 1, 1, 1], dtype=np.float16) ] * 4  # 必须用模型一样的 dtype
downsample_ratio = np.array([0.25], dtype=np.float32)  # 必须是 FP32

for src in YOUR_VIDEO:  # src 张量是 [B, C, H, W] 形状，必须用模型一样的 dtype
    fgr, pha, *rec = sess.run([], {
        'src': src, 
        'r1i': rec[0], 
        'r2i': rec[1], 
        'r3i': rec[2], 
        'r4i': rec[3], 
        'downsample_ratio': downsample_ratio
    })
```

若使用 GPU，上例会将记忆输出传回到 CPU，再在下一帧时传回到 GPU。这种传输是无意义的，因为记忆值可以留在 GPU 上。下例使用 `iobinding` 来杜绝无用的传输。

```python
import onnxruntime as ort
import numpy as np

# 载入模型
sess = ort.InferenceSession('rvm_mobilenetv3_fp16.onnx')

# 创建 io binding.
io = sess.io_binding()

# 在 CUDA 上创建张量
rec = [ ort.OrtValue.ortvalue_from_numpy(np.zeros([1, 1, 1, 1], dtype=np.float16), 'cuda') ] * 4
downsample_ratio = ort.OrtValue.ortvalue_from_numpy(np.asarray([0.25], dtype=np.float32), 'cuda')

# 设置输出项
for name in ['fgr', 'pha', 'r1o', 'r2o', 'r3o', 'r4o']:
    io.bind_output(name, 'cuda')

# 推断
for src in YOUR_VIDEO:
    io.bind_cpu_input('src', src)
    io.bind_ortvalue_input('r1i', rec[0])
    io.bind_ortvalue_input('r2i', rec[1])
    io.bind_ortvalue_input('r3i', rec[2])
    io.bind_ortvalue_input('r4i', rec[3])
    io.bind_ortvalue_input('downsample_ratio', downsample_ratio)

    sess.run_with_iobinding(io)

    fgr, pha, *rec = io.get_outputs()

    # 只将 `fgr` 和 `pha` 回传到 CPU
    fgr = fgr.numpy()
    pha = pha.numpy()
```

注：若你使用其他推断框架，可能有些 ONNX ops 不被支持，需被替换。可以参考 [onnx](https://github.com/PeterL1n/RobustVideoMatting/tree/onnx) 分支的代码做自行导出。

<br><br><br>

### TensorFlow

范例:

```python
import tensorflow as tf

model = tf.keras.models.load_model('rvm_mobilenetv3_tf')
model = tf.function(model)

rec = [ tf.constant(0.) ] * 4         # 初始记忆
downsample_ratio = tf.constant(0.25)  # 下采样率，根据视频调整

for src in YOUR_VIDEO:  # src 张量是 [B, H, W, C] 的形状，而不是 [B, C, H, W]!
    out = model([src, *rec, downsample_ratio])
    fgr, pha, *rec = out['fgr'], out['pha'], out['r1o'], out['r2o'], out['r3o'], out['r4o']
```

注意，在 TensorFlow 上，所有张量都是 Channal Last 的格式。

我们提供 TensorFlow 的原始模型代码，请参考 [tensorflow](https://github.com/PeterL1n/RobustVideoMatting/tree/tensorflow) 分支。您可自行将 PyTorch 的权值转到 TensorFlow 模型上。


<br><br><br>

### TensorFlow.js

我们在 [tfjs](https://github.com/PeterL1n/RobustVideoMatting/tree/tfjs) 分支提供范例代码。代码简单易懂，解释如何正确使用模型。

<br><br><br>

### CoreML

我们只展示在 Python 下通过 `coremltools` 使用 CoreML 模型。在部署时，同样逻辑可用于 Swift。模型的循环记忆输入不需要在处理第一帧时提供。CoreML 内部会自动创建零张量作为初始记忆。

```python
import coremltools as ct

model = ct.models.model.MLModel('rvm_mobilenetv3_1920x1080_s0.25_int8.mlmodel')

r1, r2, r3, r4 = None, None, None, None

for src in YOUR_VIDEO:  # src 是 PIL.Image.
    
    if r1 is None:
        # 初始帧, 不用提供循环记忆
        inputs = {'src': src}
    else:
        # 剩余帧，提供循环记忆
        inputs = {'src': src, 'r1i': r1, 'r2i': r2, 'r3i': r3, 'r4i': r4}

    outputs = model.predict(inputs)

    fgr = outputs['fgr']  # PIL.Image
    pha = outputs['pha']  # PIL.Image
    
    r1 = outputs['r1o']  # Numpy array
    r2 = outputs['r2o']  # Numpy array
    r3 = outputs['r3o']  # Numpy array
    r4 = outputs['r4o']  # Numpy array

```

我们的 CoreML 模型只支持固定分辨率。如果你需要其他分辨率，可自行导出。导出代码见 [coreml](https://github.com/PeterL1n/RobustVideoMatting/tree/coreml) 分支。