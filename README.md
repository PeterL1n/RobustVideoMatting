# Robust Video Matting (RVM)

![Teaser](/documentation/image/teaser.gif)

<p align="center">English | <a href="README_zh_Hans.md">中文</a></p>

Official repository for the paper [Robust High-Resolution Video Matting with Temporal Guidance](https://peterl1n.github.io/RobustVideoMatting/). RVM is specifically designed for robust human video matting. Unlike existing neural models that process frames as independent images, RVM uses a recurrent neural network to process videos with temporal memory. RVM can perform matting in real-time on any videos without additional inputs. It achieves **4K 76FPS** and **HD 104FPS** on an Nvidia GTX 1080 Ti GPU. The project was developed at [ByteDance Inc.](https://www.bytedance.com/)

<br>

## News

* [Nov 03 2021] Fixed a bug in [train.py](https://github.com/PeterL1n/RobustVideoMatting/commit/48effc91576a9e0e7a8519f3da687c0d3522045f).
* [Sep 16 2021] Code is re-released under GPL-3.0 license.
* [Aug 25 2021] Source code and pretrained models are published.
* [Jul 27 2021] Paper is accepted by WACV 2022.

<br>

## Showreel
Watch the showreel video ([YouTube](https://youtu.be/Jvzltozpbpk), [Bilibili](https://www.bilibili.com/video/BV1Z3411B7g7/)) to see the model's performance. 

<p align="center">
    <a href="https://youtu.be/Jvzltozpbpk">
        <img src="documentation/image/showreel.gif">
    </a>
</p>

All footage in the video are available in [Google Drive](https://drive.google.com/drive/folders/1VFnWwuu-YXDKG-N6vcjK_nL7YZMFapMU?usp=sharing).

<br>


## Demo
* [Webcam Demo](https://peterl1n.github.io/RobustVideoMatting/#/demo): Run the model live in your browser. Visualize recurrent states.
* [Colab Demo](https://colab.research.google.com/drive/10z-pNKRnVNsp0Lq9tH1J_XPZ7CBC_uHm?usp=sharing): Test our model on your own videos with free GPU. 

<br>

## Download

We recommend MobileNetv3 models for most use cases. ResNet50 models are the larger variant with small performance improvements. Our model is available on various inference frameworks. See [inference documentation](documentation/inference.md) for more instructions.

<table>
    <thead>
        <tr>
            <td>Framework</td>
            <td>Download</td>
            <td>Notes</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>PyTorch</td>
            <td>
                <a  href="https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth">rvm_mobilenetv3.pth</a><br>
                <a  href="https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth">rvm_resnet50.pth</a>
            </td>
            <td>
                Official weights for PyTorch. <a href="documentation/inference.md#pytorch">Doc</a>
            </td>
        </tr>
        <tr>
            <td>TorchHub</td>
            <td>
                Nothing to Download.
            </td>
            <td>
                Easiest way to use our model in your PyTorch project. <a href="documentation/inference.md#torchhub">Doc</a>
            </td>
        </tr>
        <tr>
            <td>TorchScript</td>
            <td>
                <a  href="https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.torchscript">rvm_mobilenetv3_fp32.torchscript</a><br>
                <a  href="https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp16.torchscript">rvm_mobilenetv3_fp16.torchscript</a><br>
                <a  href="https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50_fp32.torchscript">rvm_resnet50_fp32.torchscript</a><br>
                <a  href="https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50_fp16.torchscript">rvm_resnet50_fp16.torchscript</a>
            </td>
            <td>
                If inference on mobile, consider export int8 quantized models yourself. <a href="documentation/inference.md#torchscript">Doc</a>
            </td>
        </tr>
        <tr>
            <td>ONNX</td>
            <td>
                <a  href="https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx">rvm_mobilenetv3_fp32.onnx</a><br>
                <a  href="https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp16.onnx">rvm_mobilenetv3_fp16.onnx</a><br>
                <a  href="https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50_fp32.onnx">rvm_resnet50_fp32.onnx</a><br>
                <a  href="https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50_fp16.onnx">rvm_resnet50_fp16.onnx</a>
            </td>
            <td>
                Tested on ONNX Runtime with CPU and CUDA backends. Provided models use opset 12. <a href="documentation/inference.md#onnx">Doc</a>, <a href="https://github.com/PeterL1n/RobustVideoMatting/tree/onnx">Exporter</a>.
            </td>
        </tr>
        <tr>
            <td>TensorFlow</td>
            <td>
                <a  href="https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_tf.zip">rvm_mobilenetv3_tf.zip</a><br>
                <a  href="https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50_tf.zip">rvm_resnet50_tf.zip</a>
            </td>
            <td>
                TensorFlow 2 SavedModel. <a href="documentation/inference.md#tensorflow">Doc</a>
            </td>
        </tr>
        <tr>
            <td>TensorFlow.js</td>
            <td>
                <a  href="https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_tfjs_int8.zip">rvm_mobilenetv3_tfjs_int8.zip</a><br>
            </td>
            <td>
                Run the model on the web. <a href="https://peterl1n.github.io/RobustVideoMatting/#/demo">Demo</a>, <a href="https://github.com/PeterL1n/RobustVideoMatting/tree/tfjs">Starter Code</a>
            </td>
        </tr>
        <tr>
            <td>CoreML</td>
            <td>
                <a  href="https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_1280x720_s0.375_fp16.mlmodel">rvm_mobilenetv3_1280x720_s0.375_fp16.mlmodel</a><br>
                <a  href="https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_1280x720_s0.375_int8.mlmodel">rvm_mobilenetv3_1280x720_s0.375_int8.mlmodel</a><br>
                <a  href="https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_1920x1080_s0.25_fp16.mlmodel">rvm_mobilenetv3_1920x1080_s0.25_fp16.mlmodel</a><br>
                <a  href="https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_1920x1080_s0.25_int8.mlmodel">rvm_mobilenetv3_1920x1080_s0.25_int8.mlmodel</a><br>
            </td>
            <td>
                CoreML does not support dynamic resolution. Other resolutions can be exported yourself. Models require iOS 13+. <code>s</code> denotes <code>downsample_ratio</code>. <a href="documentation/inference.md#coreml">Doc</a>, <a href="https://github.com/PeterL1n/RobustVideoMatting/tree/coreml">Exporter</a>
            </td>
        </tr>
    </tbody>
</table>

All models are available in [Google Drive](https://drive.google.com/drive/folders/1pBsG-SCTatv-95SnEuxmnvvlRx208VKj?usp=sharing) and [Baidu Pan](https://pan.baidu.com/s/1puPSxQqgBFOVpW4W7AolkA) (code: gym7).

<br>

## PyTorch Example

1. Install dependencies:
```sh
pip install -r requirements_inference.txt
```

2. Load the model:

```python
import torch
from model import MattingNetwork

model = MattingNetwork('mobilenetv3').eval().cuda()  # or "resnet50"
model.load_state_dict(torch.load('rvm_mobilenetv3.pth'))
```

3. To convert videos, we provide a simple conversion API:

```python
from inference import convert_video

convert_video(
    model,                           # The model, can be on any device (cpu or cuda).
    input_source='input.mp4',        # A video file or an image sequence directory.
    output_type='video',             # Choose "video" or "png_sequence"
    output_composition='com.mp4',    # File path if video; directory path if png sequence.
    output_alpha="pha.mp4",          # [Optional] Output the raw alpha prediction.
    output_foreground="fgr.mp4",     # [Optional] Output the raw foreground prediction.
    output_video_mbps=4,             # Output video mbps. Not needed for png sequence.
    downsample_ratio=None,           # A hyperparameter to adjust or use None for auto.
    seq_chunk=12,                    # Process n frames at once for better parallelism.
)
```

4. Or write your own inference code:
```python
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from inference_utils import VideoReader, VideoWriter

reader = VideoReader('input.mp4', transform=ToTensor())
writer = VideoWriter('output.mp4', frame_rate=30)

bgr = torch.tensor([.47, 1, .6]).view(3, 1, 1).cuda()  # Green background.
rec = [None] * 4                                       # Initial recurrent states.
downsample_ratio = 0.25                                # Adjust based on your video.

with torch.no_grad():
    for src in DataLoader(reader):                     # RGB tensor normalized to 0 ~ 1.
        fgr, pha, *rec = model(src.cuda(), *rec, downsample_ratio)  # Cycle the recurrent states.
        com = fgr * pha + bgr * (1 - pha)              # Composite to green background. 
        writer.write(com)                              # Write frame.
```

5. The models and converter API are also available through TorchHub.

```python
# Load the model.
model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3") # or "resnet50"

# Converter API.
convert_video = torch.hub.load("PeterL1n/RobustVideoMatting", "converter")
```

Please see [inference documentation](documentation/inference.md) for details on `downsample_ratio` hyperparameter, more converter arguments, and more advanced usage.

<br>

## Training and Evaluation

Please refer to the [training documentation](documentation/training.md) to train and evaluate your own model.

<br>

## Speed

Speed is measured with `inference_speed_test.py` for reference.

| GPU            | dType | HD (1920x1080) | 4K (3840x2160) |
| -------------- | ----- | -------------- |----------------|
| RTX 3090       | FP16  | 172 FPS        | 154 FPS        |
| RTX 2060 Super | FP16  | 134 FPS        | 108 FPS        |
| GTX 1080 Ti    | FP32  | 104 FPS        | 74 FPS         |

* Note 1: HD uses `downsample_ratio=0.25`, 4K uses `downsample_ratio=0.125`. All tests use batch size 1 and frame chunk 1.
* Note 2: GPUs before Turing architecture does not support FP16 inference, so GTX 1080 Ti uses FP32.
* Note 3: We only measure tensor throughput. The provided video conversion script in this repo is expected to be much slower, because it does not utilize hardware video encoding/decoding and does not have the tensor transfer done on parallel threads. If you are interested in implementing hardware video encoding/decoding in Python, please refer to [PyNvCodec](https://github.com/NVIDIA/VideoProcessingFramework).

<br>  

## Project Members
* [Shanchuan Lin](https://www.linkedin.com/in/shanchuanlin/)
* [Linjie Yang](https://sites.google.com/site/linjieyang89/)
* [Imran Saleemi](https://www.linkedin.com/in/imran-saleemi/)
* [Soumyadip Sengupta](https://homes.cs.washington.edu/~soumya91/)

<br>

## Third-Party Projects

* [NCNN C++ Android](https://github.com/FeiGeChuanShu/ncnn_Android_RobustVideoMatting) ([@FeiGeChuanShu](https://github.com/FeiGeChuanShu))
* [lite.ai.toolkit](https://github.com/DefTruth/RobustVideoMatting.lite.ai.toolkit) ([@DefTruth](https://github.com/DefTruth))
* [Gradio Web Demo](https://huggingface.co/spaces/akhaliq/Robust-Video-Matting) ([@AK391](https://github.com/AK391))
* [Unity Engine demo with NatML](https://hub.natml.ai/@natsuite/robust-video-matting) ([@natsuite](https://github.com/natsuite))  
* [MNN C++ Demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/mnn/cv/mnn_rvm.cpp) ([@DefTruth](https://github.com/DefTruth))
* [TNN C++ Demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/tnn/cv/tnn_rvm.cpp) ([@DefTruth](https://github.com/DefTruth))

