import torch
import adder_copy as adder
import torch.nn as nn
# from devkit.sparse_ops import SparseConv

def conv3x3(in_planes, out_planes, stride=1):
    return adder.adder2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def quantize_activation(x, clip_val, q_bits):
    """Quantize tensor `x` into uniform levels within [0, clip_val].

    If `q_bits` cannot be parsed or is invalid, falls back to 4 bits.
    If `clip_val` <= 0 or levels <= 0, returns `x` unchanged.
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
            return torch.round(x / delta) * delta
    return x


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None, clip_value1=6.0, clip_value2=6.0,
                 clip_bits1=4, clip_bits2=4):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride = stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.clip_value1 = clip_value1
        self.clip_value2 = clip_value2
        self.clip_bits1 = clip_bits1
        self.clip_bits2 = clip_bits2

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = torch.clip(out, 0, self.clip_value1)
        out = quantize_activation(out, self.clip_value1, self.clip_bits1)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        out = self.relu(out)
        out = torch.clip(out, 0, self.clip_value2)
        out = quantize_activation(out, self.clip_value2, self.clip_bits2)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, clip_values=None, act_bits=None):
        super(ResNet, self).__init__()
        # Default clip values: 19 ReLU layers (1 initial + 18 in BasicBlocks)
        # Format: [clip_initial, clip_layer1_block1_relu1, clip_layer1_block1_relu2, ...]
        if clip_values is None:
            clip_values = [6.0] * 19
        # activation quant bits: accept a single int (global) or a list
        # If a single int is provided, use it for all 19 ReLU layers; if a list
        # is provided, use its first element as the global bitwidth.
        if act_bits is None:
            act_bits_value = 4
        else:
            try:
                # allow passing an int-like
                act_bits_value = int(act_bits)
            except Exception:
                try:
                    # if a list-like is passed, take the first element
                    act_bits_value = int(act_bits[0])
                except Exception:
                    act_bits_value = 4
        # expand to per-ReLU list internally (length 19)
        self.act_bits = [act_bits_value] * 19
        self.clip_values = clip_values
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # clip_values[0] for initial ReLU, [1-6] for layer1, [7-12] for layer2, [13-18] for layer3
        # slice corresponding act_bits for passing to blocks
        act_bits_block1 = self.act_bits[1:1+layers[0]*2] if len(self.act_bits) > 1 else [4]*6
        act_bits_block2 = self.act_bits[1+layers[0]*2:1+layers[0]*2+layers[1]*2] if len(self.act_bits) > 7 else [4]*6
        act_bits_block3 = self.act_bits[1+layers[0]*2+layers[1]*2:] if len(self.act_bits) > 13 else [4]*6

        self.layer1 = self._make_layer(block, 16, layers[0], 
                                         clip_values_block=clip_values[1:1+layers[0]*2] if len(clip_values) > 1 else [6.0]*6,
                                         clip_bits_block=act_bits_block1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, 
                                         clip_values_block=clip_values[1+layers[0]*2:1+layers[0]*2+layers[1]*2] if len(clip_values) > 7 else [6.0]*6,
                                         clip_bits_block=act_bits_block2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, 
                                         clip_values_block=clip_values[1+layers[0]*2+layers[1]*2:] if len(clip_values) > 13 else [6.0]*6,
                                         clip_bits_block=act_bits_block3)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Conv2d(64 * block.expansion, num_classes, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
         
    def _make_layer(self, block, planes, blocks, stride=1, clip_values_block=None, clip_bits_block=None):
        if clip_values_block is None:
            clip_values_block = [6.0] * (blocks * 2)
        if clip_bits_block is None:
            clip_bits_block = [4] * (blocks * 2)
        
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                adder.adder2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        # First block has downsample
        layers.append(block(inplanes = self.inplanes, planes = planes, stride = stride, 
                   downsample = downsample, 
                   clip_value1=clip_values_block[0], clip_value2=clip_values_block[1],
                   clip_bits1=clip_bits_block[0], clip_bits2=clip_bits_block[1]))
        self.inplanes = planes * block.expansion
        # Remaining blocks
        for i in range(1, blocks):
            idx = i * 2
            layers.append(block(inplanes = self.inplanes, planes = planes, 
                               clip_value1=clip_values_block[idx], clip_value2=clip_values_block[idx+1],
                               clip_bits1=clip_bits_block[idx], clip_bits2=clip_bits_block[idx+1]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # clip then quantize the initial ReLU output to match adder input quantization
        x = torch.clip(x, 0, self.clip_values[0])
        # Quantize activation to Q bits (symmetric levels in [0, clip_val])
        # Use configured act_bits[0] if available so the bitwidth is top-level configurable
        try:
            Q_ACT = int(self.act_bits[0])
        except Exception:
            Q_ACT = 4
        clip_val = float(self.clip_values[0])
        levels = 2 ** Q_ACT - 1
        if clip_val > 0 and levels > 0:
            delta_a = clip_val / levels
            # avoid unnecessary ops if delta_a is zero
            if delta_a > 0:
                x = torch.round(x / delta_a) * delta_a

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn2(x)

        return x.view(x.size(0), -1)


def resnet20(clip_values=None, **kwargs):
    """
    Create ResNet20 model with configurable clip values and activation bitwidth.

    Args:
        clip_values: Optional list of 19 floats, one clip value per ReLU layer.
            Order: [initial_relu,
                    layer1_block1_relu1, layer1_block1_relu2, ...,
                    layer2_block1_relu1, layer2_block1_relu2, ...,
                    layer3_block1_relu1, layer3_block1_relu2, ...]
            If None, defaults to [6.0] * 19.

        act_bits: Optional single int (recommended) or list-like. When an int
            is provided it is used as the global activation bitwidth for all 19
            ReLU layers. If a list-like is provided, its first element will be
            used as the global bitwidth. Internally the value is expanded to a
            length-19 list and applied uniformly to every ReLU.

    Example:
        # Use default clip values and 4-bit activations
        model = resnet20()

        # Use a global 3-bit activation quantization
        model = resnet20(act_bits=3)

        # Provide custom clip values per ReLU (19 values)
        model = resnet20(clip_values=[6.0, 4.0, 5.0, 6.0, 5.0, 6.0, 5.0, 6.0,
                                      5.0, 6.0, 5.0, 6.0, 5.0, 6.0, 5.0, 6.0,
                                      5.0, 6.0, 5.0])
    """
    act_bits = kwargs.get('act_bits', None)
    return ResNet(BasicBlock, [3, 3, 3], clip_values=clip_values, act_bits=act_bits)

