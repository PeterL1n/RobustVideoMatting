from tensorflow.keras.layers import DepthwiseConv2D
from .mobilenetv3 import *
from .resnet import ResNet50Encoder
from .deep_guided_filter import DeepGuidedFilterRefiner

# --------------------------- Load torch weights  ---------------------------
def load_torch_weights(model, state_dict):
    if isinstance(model.backbone, MobileNetV3Encoder):
        load_MobileNetV3_weights(model.backbone, state_dict, 'backbone')
    if isinstance(model.backbone, ResNet50Encoder):
        load_ResNetEncoder_weights(model.backbone, state_dict, 'backbone')
    load_LRASPP_weights(model.aspp, state_dict, 'aspp')
    load_RecurrentDecoder_weights(model.decoder, state_dict, 'decoder')
    load_conv_weights(model.project_mat, state_dict, 'project_mat.conv')
    if isinstance(model.refiner, DeepGuidedFilterRefiner):
        load_DeepGuidedFilter_weights(model.refiner, state_dict, 'refiner')
    
# --------------------------- General  ---------------------------
def load_conv_weights(conv, state_dict, name):
    weight = state_dict[name + '.weight']
    if isinstance(conv, DepthwiseConv2D):
        weight = weight.permute(2, 3, 0, 1).numpy()
    else:
        weight = weight.permute(2, 3, 1, 0).numpy()
    if name + '.bias' in state_dict:
        bias = state_dict[name + '.bias'].numpy()
        conv.set_weights([weight, bias])
    else:
        conv.set_weights([weight])

def load_bn_weights(bn, state_dict, name):
    weight = state_dict[name + '.weight']
    bias = state_dict[name + '.bias']
    running_mean = state_dict[name + '.running_mean']
    running_var = state_dict[name + '.running_var']
    bn.set_weights([weight, bias, running_mean, running_var])
        
# --------------------------- MobileNetV3 ---------------------------
def load_ConvBNActivation_weights(module, state_dict, name):
    load_conv_weights(module.conv, state_dict, name + '.0')
    load_bn_weights(module.bn, state_dict, name + '.1')

def load_InvertedResidual_weights(module, state_dict, name):
    for i, layer in enumerate(module.block.layers):
        if isinstance(layer, ConvBNActivation):
            load_ConvBNActivation_weights(layer, state_dict, f'{name}.block.{i}')
        if isinstance(layer, SqueezeExcitation):
            load_conv_weights(layer.fc1, state_dict, f'{name}.block.{i}.fc1')
            load_conv_weights(layer.fc2, state_dict, f'{name}.block.{i}.fc2')

def load_MobileNetV3_weights(backbone, state_dict, name):
    for i, module in enumerate(backbone.features):
        if isinstance(module, ConvBNActivation):
            load_ConvBNActivation_weights(module, state_dict, f'{name}.features.{i}')
        if isinstance(module, InvertedResidual):
            load_InvertedResidual_weights(module, state_dict, f'{name}.features.{i}')

# --------------------------- ResNet ---------------------------
def load_ResNetEncoder_weights(module, state_dict, name):
    load_conv_weights(module.conv1, state_dict, f'{name}.conv1')
    load_bn_weights(module.bn1, state_dict, f'{name}.bn1')
    for l in range(1, 5):
        for b, resblock in enumerate(getattr(module, f'layer{l}').layers):
            if hasattr(resblock, 'convd'):
                load_conv_weights(resblock.convd, state_dict, f'{name}.layer{l}.{b}.downsample.0')
                load_bn_weights(resblock.bnd, state_dict, f'{name}.layer{l}.{b}.downsample.1')
            load_conv_weights(resblock.conv1, state_dict, f'{name}.layer{l}.{b}.conv1')
            load_conv_weights(resblock.conv2, state_dict, f'{name}.layer{l}.{b}.conv2')
            load_conv_weights(resblock.conv3, state_dict, f'{name}.layer{l}.{b}.conv3')
            load_bn_weights(resblock.bn1, state_dict, f'{name}.layer{l}.{b}.bn1')
            load_bn_weights(resblock.bn2, state_dict, f'{name}.layer{l}.{b}.bn2')
            load_bn_weights(resblock.bn3, state_dict, f'{name}.layer{l}.{b}.bn3')

# --------------------------- LRASPP ---------------------------
def load_LRASPP_weights(module, state_dict, name):
    load_conv_weights(module.aspp1.layers[0], state_dict, f'{name}.aspp1.0')
    load_bn_weights(module.aspp1.layers[1], state_dict, f'{name}.aspp1.1')
    load_conv_weights(module.aspp2, state_dict, f'{name}.aspp2.1')
        
# --------------------------- RecurrentDecoder ---------------------------
def load_ConvGRU_weights(module, state_dict, name):
    load_conv_weights(module.ih, state_dict, f'{name}.ih.0')
    load_conv_weights(module.hh, state_dict, f'{name}.hh.0')

def load_BottleneckBlock_weights(module, state_dict, name):
    load_ConvGRU_weights(module.gru, state_dict, f'{name}.gru')

def load_UpsamplingBlock_weights(module, state_dict, name):
    load_conv_weights(module.conv.layers[0], state_dict, f'{name}.conv.0')
    load_bn_weights(module.conv.layers[1], state_dict, f'{name}.conv.1')
    load_ConvGRU_weights(module.gru, state_dict, f'{name}.gru')

def load_OutputBlock_weights(module, state_dict, name):
    load_conv_weights(module.conv.layers[0], state_dict, f'{name}.conv.0')
    load_bn_weights(module.conv.layers[1], state_dict, f'{name}.conv.1')
    load_conv_weights(module.conv.layers[3], state_dict, f'{name}.conv.3')
    load_bn_weights(module.conv.layers[4], state_dict, f'{name}.conv.4')

def load_RecurrentDecoder_weights(module, state_dict, name):
    load_BottleneckBlock_weights(module.decode4, state_dict, f'{name}.decode4')
    load_UpsamplingBlock_weights(module.decode3, state_dict, f'{name}.decode3')
    load_UpsamplingBlock_weights(module.decode2, state_dict, f'{name}.decode2')
    load_UpsamplingBlock_weights(module.decode1, state_dict, f'{name}.decode1')
    load_OutputBlock_weights(module.decode0, state_dict, f'{name}.decode0')
    
# --------------------------- DeepGuidedFilter ---------------------------
def load_DeepGuidedFilter_weights(module, state_dict, name):
    load_conv_weights(module.box_filter.layers[1], state_dict, f'{name}.box_filter')
    load_conv_weights(module.conv.layers[0], state_dict, f'{name}.conv.0')
    load_bn_weights(module.conv.layers[1], state_dict, f'{name}.conv.1')
    load_conv_weights(module.conv.layers[3], state_dict, f'{name}.conv.3')
    load_bn_weights(module.conv.layers[4], state_dict, f'{name}.conv.4')
    load_conv_weights(module.conv.layers[6], state_dict, f'{name}.conv.6')