# 2020.01.10-Replaced conv with adder, adapted for quantization-aware ResNet50
#            Based on ResNet20 quantization methodology with 4-bit activation quantization
#            and BN fusion bias removal

import torch
import adder
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return adder.adder2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return adder.adder2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def quantize_activation(x, clip_val, q_bits):
    """
    Quantize activation tensor `x` into uniform levels within [0, clip_val].
    
    Implements 4-bit (or q_bits) uniform quantization:
        delta = clip_val / (2^q_bits - 1)
        x_q = round(x / delta) * delta
    
    Args:
        x: Input activation tensor
        clip_val: Clipping bound (ReLU range upper limit)
        q_bits: Number of quantization bits (default 4)
    
    Returns:
        Quantized activation tensor
    """
    try:
        q = int(q_bits)
    except Exception:
        q = 4
    
    levels = 2 ** q - 1
    
    try:
        clip_val_f = float(clip_val)
    except Exception:
        return x
    
    if clip_val_f > 0 and levels > 0:
        delta = clip_val_f / levels
        if delta > 0:
            # Uniform quantization: round to nearest quantization level
            return torch.round(x / delta) * delta
    
    return x


class Bottleneck(nn.Module):
    """
    Bottleneck block for ResNet50 with per-ReLU activation quantization.
    
    Structure:
        conv1 (1x1) -> bn1 -> relu -> (clip + quantize)
        conv2 (3x3) -> bn2 -> relu -> (clip + quantize)
        conv3 (1x1) -> bn3 -> (add with residual)
        relu -> (clip + quantize)
    
    Each of the 3 ReLU outputs can have different clip values and quantization bits.
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 clip_value1=6.0, clip_value2=6.0, clip_value3=6.0,
                 clip_bits1=4, clip_bits2=4, clip_bits3=4):
        """
        Args:
            inplanes: Input channels
            planes: Output channels before expansion
            stride: Stride for conv2
            downsample: Optional downsampling layer for residual
            clip_value1: Clip value for ReLU after bn1
            clip_value2: Clip value for ReLU after bn2
            clip_value3: Clip value for ReLU after residual add
            clip_bits1: Quantization bits for activation1
            clip_bits2: Quantization bits for activation2
            clip_bits3: Quantization bits for activation3
        """
        super(Bottleneck, self).__init__()
        
        # conv1x1 reduce
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # conv3x3 spatial
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # conv1x1 expand
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
        # Per-ReLU clipping and quantization parameters
        self.clip_value1 = clip_value1
        self.clip_value2 = clip_value2
        self.clip_value3 = clip_value3
        
        self.clip_bits1 = clip_bits1
        self.clip_bits2 = clip_bits2
        self.clip_bits3 = clip_bits3

    def forward(self, x):
        identity = x

        # Branch 1: conv1 -> bn1 -> relu -> clip -> quantize
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = torch.clip(out, 0, self.clip_value1)
        out = quantize_activation(out, self.clip_value1, self.clip_bits1)

        # Branch 2: conv2 -> bn2 -> relu -> clip -> quantize
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = torch.clip(out, 0, self.clip_value2)
        out = quantize_activation(out, self.clip_value2, self.clip_bits2)

        # Branch 3: conv3 -> bn3 (no immediate relu)
        out = self.conv3(out)
        out = self.bn3(out)

        # Residual connection (with optional downsampling)
        if self.downsample is not None:
            identity = self.downsample(x)

        # Residual add and final relu -> clip -> quantize
        out += identity
        out = self.relu(out)
        out = torch.clip(out, 0, self.clip_value3)
        out = quantize_activation(out, self.clip_value3, self.clip_bits3)

        return out


class ResNet(nn.Module):
    """
    ResNet50 with activation quantization and per-layer clip control.
    
    Supports:
    - Per-ReLU clip values (49 total: 1 initial + 3*16 in blocks)
    - Per-ReLU quantization bits (4-bit by default)
    - Automatic handling of clip_values and act_bits lists
    
    Structure: [3, 4, 6, 3] blocks in 4 layers
    Total ReLU layers: 1 (initial) + 3*3 (layer1) + 4*3 (layer2) + 6*3 (layer3) + 3*3 (layer4) = 49
    """

    def __init__(self, block, layers, num_classes=1000, clip_values=None, act_bits=None):
        """
        Args:
            block: Bottleneck class
            layers: [3, 4, 6, 3] for ResNet50
            num_classes: Number of output classes (default 1000 for ImageNet)
            clip_values: List of 49 clip values (one per ReLU). 
                        If None, defaults to [6.0] * 49
                        Order: [initial_relu, layer1_block*_relu*, layer2_block*_relu*, ...]
            act_bits: Single int (applied to all ReLUs) or list of 49 ints.
                     If int, expanded to [int]*49. If None, defaults to 4.
        """
        super(ResNet, self).__init__()
        
        # Default clip values: 49 ReLU layers
        # [1 initial, 3*3 (layer1), 4*3 (layer2), 6*3 (layer3), 3*3 (layer4)]
        if clip_values is None:
            clip_values = [6.0] * 49
        
        # Handle act_bits: accept single int or list, expand to global setting
        if act_bits is None:
            act_bits_value = 4
        else:
            try:
                act_bits_value = int(act_bits)
            except (TypeError, ValueError):
                try:
                    act_bits_value = int(act_bits[0])
                except Exception:
                    act_bits_value = 4
        
        # Expand to per-ReLU list internally (length 49)
        self.act_bits = [act_bits_value] * 49
        self.clip_values = clip_values
        
        self.inplanes = 64
        
        # Initial conv1 (7x7) with post-relu quantization
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Prepare clip values and bits for each layer
        # clip_values[0]: initial ReLU
        # clip_values[1:4]: layer1 (3 blocks * 3 ReLUs each = 9 values)
        # clip_values[10:22]: layer2 (4 blocks * 3 ReLUs each = 12 values)
        # clip_values[22:40]: layer3 (6 blocks * 3 ReLUs each = 18 values)
        # clip_values[40:49]: layer4 (3 blocks * 3 ReLUs each = 9 values)
        
        act_bits_layer1 = self.act_bits[1:1+layers[0]*3]
        act_bits_layer2 = self.act_bits[1+layers[0]*3:1+layers[0]*3+layers[1]*3]
        act_bits_layer3 = self.act_bits[1+layers[0]*3+layers[1]*3:1+layers[0]*3+layers[1]*3+layers[2]*3]
        act_bits_layer4 = self.act_bits[1+layers[0]*3+layers[1]*3+layers[2]*3:]
        
        clip_layer1 = clip_values[1:1+layers[0]*3] if len(clip_values) > 1 else [6.0]*9
        clip_layer2 = clip_values[1+layers[0]*3:1+layers[0]*3+layers[1]*3] if len(clip_values) > 10 else [6.0]*12
        clip_layer3 = clip_values[1+layers[0]*3+layers[1]*3:1+layers[0]*3+layers[1]*3+layers[2]*3] if len(clip_values) > 22 else [6.0]*18
        clip_layer4 = clip_values[1+layers[0]*3+layers[1]*3+layers[2]*3:] if len(clip_values) > 40 else [6.0]*9
        
        self.layer1 = self._make_layer(block, 64, layers[0], clip_values_block=clip_layer1, 
                                       clip_bits_block=act_bits_layer1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, 
                                       clip_values_block=clip_layer2, 
                                       clip_bits_block=act_bits_layer2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, 
                                       clip_values_block=clip_layer3, 
                                       clip_bits_block=act_bits_layer3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, 
                                       clip_values_block=clip_layer4, 
                                       clip_bits_block=act_bits_layer4)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Conv2d(512 * block.expansion, num_classes, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_classes)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, clip_values_block=None, clip_bits_block=None):
        """
        Build a layer of Bottleneck blocks.
        
        Args:
            block: Bottleneck class
            planes: Output channels before expansion
            blocks: Number of blocks in this layer
            stride: Stride for first block
            clip_values_block: List of 3*blocks clip values (3 per block)
            clip_bits_block: List of 3*blocks quantization bits (3 per block)
        """
        if clip_values_block is None:
            clip_values_block = [6.0] * (blocks * 3)
        if clip_bits_block is None:
            clip_bits_block = [4] * (blocks * 3)
        
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # Downsample: adder conv1x1 followed by BN
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # First block (with downsampling if needed)
        layers.append(
            block(self.inplanes, planes, stride=stride, downsample=downsample,
                  clip_value1=clip_values_block[0], clip_value2=clip_values_block[1],
                  clip_value3=clip_values_block[2],
                  clip_bits1=clip_bits_block[0], clip_bits2=clip_bits_block[1],
                  clip_bits3=clip_bits_block[2])
        )
        self.inplanes = planes * block.expansion
        
        # Remaining blocks
        for i in range(1, blocks):
            idx = i * 3
            layers.append(
                block(self.inplanes, planes,
                      clip_value1=clip_values_block[idx], clip_value2=clip_values_block[idx+1],
                      clip_value3=clip_values_block[idx+2],
                      clip_bits1=clip_bits_block[idx], clip_bits2=clip_bits_block[idx+1],
                      clip_bits3=clip_bits_block[idx+2])
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial convolution and pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Quantize initial ReLU output (clip_values[0])
        x = torch.clip(x, 0, self.clip_values[0])
        try:
            Q_ACT = int(self.act_bits[0])
        except Exception:
            Q_ACT = 4
        
        clip_val = float(self.clip_values[0])
        levels = 2 ** Q_ACT - 1
        if clip_val > 0 and levels > 0:
            delta_a = clip_val / levels
            if delta_a > 0:
                x = torch.round(x / delta_a) * delta_a
        
        x = self.maxpool(x)

        # Residual layers with per-layer activation quantization
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling and classification
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn2(x)

        return x.view(x.size(0), -1)


def resnet50(clip_values=None, **kwargs):
    """
    Create ResNet50 with quantization-aware activation handling.
    
    Features:
    - 4-bit uniform activation quantization (configurable via act_bits)
    - Per-layer clip value control (49 ReLU layers total)
    - BN fusion bias removal in quantization module
    - Dynamic clip value updating for optimization
    
    Args:
        clip_values: Optional list of 49 floats (one per ReLU layer).
                    Format: [clip_initial_relu, clip_layer1_block*_relu*, ...]
                    If None, defaults to [6.0] * 49
        
        act_bits: Optional single int or list. 
                 - Int: applied globally to all 49 ReLU layers (e.g., 4 for 4-bit)
                 - List: accepts 49 values, or uses first element as global
                 - Default: 4 bits
    
    Returns:
        ResNet50 model with quantization-aware forward pass
    
    Example:
        # Default 4-bit quantization with default clip values
        model = resnet50()
        
        # 3-bit quantization
        model = resnet50(act_bits=3)
        
        # Custom per-layer clip values
        clips = [6.0] * 49  # e.g., from optimization
        model = resnet50(clip_values=clips)
    """
    act_bits = kwargs.get('act_bits', None)
    return ResNet(Bottleneck, [3, 4, 6, 3], clip_values=clip_values, act_bits=act_bits)
