"""
Loading model
    model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3")
    model = torch.hub.load("PeterL1n/RobustVideoMatting", "resnet50")

Converter API
    convert_video = torch.hub.load("PeterL1n/RobustVideoMatting", "converter")
"""


dependencies = ['torch', 'torchvision']

import torch
from model import MattingNetwork


def mobilenetv3(pretrained: bool = True, progress: bool = True):
    model = MattingNetwork('mobilenetv3')
    if pretrained:
        url = 'https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth'
        model.load_state_dict(torch.hub.load_state_dict_from_url(url, map_location='cpu', progress=progress))
    return model


def resnet50(pretrained: bool = True, progress: bool = True):
    model = MattingNetwork('resnet50')
    if pretrained:
        url = 'https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth'
        model.load_state_dict(torch.hub.load_state_dict_from_url(url, map_location='cpu', progress=progress))
    return model


def converter():
    try:
        from inference import convert_video
        return convert_video
    except ModuleNotFoundError as error:
        print(error)
        print('Please run "pip install av tqdm pims"')
