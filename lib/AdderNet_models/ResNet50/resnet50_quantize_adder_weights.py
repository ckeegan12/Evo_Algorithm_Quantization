import torch
import torch.nn as nn
import copy
import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_RESNET50_DIR = (_THIS_DIR.parent / 'AdderNet_models' / 'ResNet50').resolve()
if str(_RESNET50_DIR) not in sys.path:
    sys.path.insert(0, str(_RESNET50_DIR))

import resnet50_actQ as resnet50

# ResNet50 adder/BN lists are inferred from checkpoint dynamically.
ADDER_LAYER_NAMES = []
BN_LAYER_NAMES = []

# Add 'module.' prefix for DataParallel models (filled after infer)
ADDER_LAYER_NAMES_WITH_PREFIX = []
BN_LAYER_NAMES_WITH_PREFIX = []

# Quantization parameters
Q = 4
DEFAULT_CLIP_VALUE = 6.0


def quantize_conv1_weight(w):
    """
    Quantize conv1 weights to 8 bits.
    
    Formula:
        delta_conv = (maxx - minn) / (2**8 - 1)
        wq = torch.round(w / delta_conv) * delta_conv
    
    Args:
        w: Weight tensor for conv1 (out_channels, in_channels, kH, kW)
    
    Returns:
        wq: 8-bit quantized weights
        delta: Quantization step size
    """
    maxx = torch.max(w)
    minn = torch.min(w)
    delta_conv = (maxx - minn) / (2**8 - 1)
    wq = torch.round(w / delta_conv) * delta_conv
    
    return wq, delta_conv

CLIP_VALUES_ARRAY = []
CLIP_VALUES = {}

# Input/Output paths
MODEL_PATH = str((_RESNET50_DIR / 'ResNet50-AdderNet.pth').resolve())
OUTPUT_PATH = str((_RESNET50_DIR / 'ResNet50-AdderNet-quantized.pth').resolve())


def _extract_state_dict(loaded_obj):
    if isinstance(loaded_obj, dict) and 'state_dict' in loaded_obj:
        return loaded_obj['state_dict']
    return loaded_obj


def infer_adder_layer_names(state_dict):
    names = set()
    for key in state_dict.keys():
        if key.endswith('.adder'):
            clean = key[7:] if key.startswith('module.') else key
            names.add(clean)
    return sorted(list(names))


def infer_bn_layer_names(adder_names):
    names = []
    for adder_name in adder_names:
        try:
            bn_name = get_bn_name_from_adder(adder_name)
            if bn_name is not None and bn_name not in names:
                names.append(bn_name)
        except Exception:
            continue
    return names


def initialize_layer_lists_from_state_dict(state_dict):
    global ADDER_LAYER_NAMES, BN_LAYER_NAMES
    global ADDER_LAYER_NAMES_WITH_PREFIX, BN_LAYER_NAMES_WITH_PREFIX
    global CLIP_VALUES_ARRAY, CLIP_VALUES

    ADDER_LAYER_NAMES = infer_adder_layer_names(state_dict)
    BN_LAYER_NAMES = infer_bn_layer_names(ADDER_LAYER_NAMES)
    ADDER_LAYER_NAMES_WITH_PREFIX = ['module.' + name for name in ADDER_LAYER_NAMES]
    BN_LAYER_NAMES_WITH_PREFIX = ['module.' + name for name in BN_LAYER_NAMES]

    CLIP_VALUES_ARRAY = [DEFAULT_CLIP_VALUE] * len(ADDER_LAYER_NAMES)
    CLIP_VALUES = {name: CLIP_VALUES_ARRAY[i] for i, name in enumerate(ADDER_LAYER_NAMES)}


def clip_values_to_relu_format(clip_values_dict, default_clip):
    """
    Convert per-layer clip values to ReLU clip format (49 values for ResNet50).

    ResNet50 (Bottleneck [3,4,6,3]) has 49 ReLU outputs used by resnet50_actQ:
    - index 0: initial ReLU after stem conv1+bn1
    - then 16 blocks * 3 ReLU outputs = 48 values (indices 1..48)

    Mapping policy (input-activation driven):
    - block.conv1 clip <- previous block output ReLU (or index 0 for first block)
    - block.conv2 clip <- block relu1
    - block.conv3 clip <- block relu2

    Downsample adder layers are ignored for ReLU clip mapping.
    Args:
        clip_values_dict: Dict mapping adder layer name to clip value
        default_clip: Default clip value
    Returns:
        List of 49 clip values for ReLU layers
    """
    relu_clip_values = [default_clip] * 49
    blocks_per_stage = [3, 4, 6, 3]

    prev_relu_idx = 0
    relu_counter = 1

    for stage_idx, n_blocks in enumerate(blocks_per_stage, start=1):
        for block_idx in range(n_blocks):
            base = f'layer{stage_idx}.{block_idx}'
            conv1_name = f'{base}.conv1.adder'
            conv2_name = f'{base}.conv2.adder'
            conv3_name = f'{base}.conv3.adder'

            relu1_idx = relu_counter
            relu2_idx = relu_counter + 1
            relu3_idx = relu_counter + 2

            relu_clip_values[prev_relu_idx] = clip_values_dict.get(conv1_name, relu_clip_values[prev_relu_idx])
            relu_clip_values[relu1_idx] = clip_values_dict.get(conv2_name, relu_clip_values[relu1_idx])
            relu_clip_values[relu2_idx] = clip_values_dict.get(conv3_name, relu_clip_values[relu2_idx])

            prev_relu_idx = relu3_idx
            relu_counter += 3

    return relu_clip_values


def get_bn_name_from_adder(adder_name):
    """
    Get the corresponding BN layer name from a ResNet50 adder layer name.

    Supports:
    - layerX.Y.conv1.adder -> layerX.Y.bn1
    - layerX.Y.conv2.adder -> layerX.Y.bn2
    - layerX.Y.conv3.adder -> layerX.Y.bn3
    - layerX.Y.downsample.0.adder -> layerX.Y.downsample.1
    """
    if adder_name.endswith('.adder'):
        base_name = adder_name[:-6]  # Remove '.adder' (6 characters)
    else:
        base_name = adder_name

    if '.downsample.0' in base_name:
        return base_name.replace('.downsample.0', '.downsample.1')
    if '.conv1' in base_name:
        return base_name.replace('.conv1', '.bn1')
    if '.conv2' in base_name:
        return base_name.replace('.conv2', '.bn2')
    if '.conv3' in base_name:
        return base_name.replace('.conv3', '.bn3')

    raise ValueError(f"Invalid/unsupported adder name format: {adder_name}")


def quantize_adder_weight(w, clip_val, Q=4):
    """
    Quantize adder weights using the clip + quantize scheme.
    
    Args:
        w: Weight tensor (out_channels, in_channels, kH, kW)
        clip_val: Clip value for quantization
        Q: Number of bits for quantization
    
    Returns:
        wq_nn: Quantized weights
        bias_sum: Bias to be added to BN running_mean
    """
    # Step 1: Clipping
    w_nn = torch.clamp(w, min=0, max=clip_val)
    
    # Step 2: Bias Calculation
    bias_tensor = (w - w_nn).abs()
    bias_sum = torch.sum(bias_tensor, dim=(1, 2, 3))
    
    # Step 3: Quantization
    delta = clip_val / (2**Q - 1)
    wq_nn = w_nn.clone() if delta == 0 else torch.round(w_nn / delta) * delta
    
    return wq_nn, bias_sum


def apply_conv1_quantization(state_dict):
    """
    Apply 8-bit quantization to conv1 layer weights.
    
    Args:
        state_dict: Model state dictionary
    
    Returns:
        Updated state_dict with conv1 quantized
        conv1_delta: Quantization step size for conv1 (for reference)
    """
    conv1_weight_key = None
    
    # Try both with and without module prefix
    if 'conv1.weight' in state_dict:
        conv1_weight_key = 'conv1.weight'
    elif 'module.conv1.weight' in state_dict:
        conv1_weight_key = 'module.conv1.weight'
    
    if conv1_weight_key is None:
        print("Warning: conv1.weight not found in state dict")
        return state_dict, None
    
    w = state_dict[conv1_weight_key]
    wq, delta = quantize_conv1_weight(w)
    
    state_dict[conv1_weight_key] = wq
    
    maxx = torch.max(w)
    minn = torch.min(w)
    print(f"Conv1 8-bit quantization:")
    print(f"  Original range: [{minn:.4f}, {maxx:.4f}]")
    print(f"  Quantization delta: {delta:.6f}")
    print(f"  Weight shape: {w.shape}")
    print(f"  Unique quantized values: {len(wq.unique())}")
    
    return state_dict, delta


def apply_quantization_to_layer(state_dict, layer_name, clip_val, Q=4):
    """
    Apply quantization to a single adder layer and fuse with BN.
    
    Args:
        state_dict: Model state dictionary
        layer_name: Name of the adder layer
        clip_val: Clip value for quantization
        Q: Number of bits
    
    Returns:
        Updated state_dict
    """
    # Try both with and without module prefix
    actual_layer_name = None
    if layer_name in state_dict:
        actual_layer_name = layer_name
    elif "module." + layer_name in state_dict:
        actual_layer_name = "module." + layer_name
    
    if actual_layer_name is None:
        print(f"Warning: Layer {layer_name} not found in state dict")
        return state_dict
    
    # Get the weight
    w = state_dict[actual_layer_name]
    
    # Apply quantization
    wq_nn, bias_sum = quantize_adder_weight(w, clip_val, Q)
    
    # Update the weight
    state_dict[actual_layer_name] = wq_nn
    
    # Get corresponding BN layer and apply fusion (try both with and without prefix)
    # The BN fusion is applied to running_mean
    bn_name = get_bn_name_from_adder(layer_name)
    bn_running_mean_key = bn_name + ".running_mean"
    
    actual_bn_key = None
    if bn_running_mean_key in state_dict:
        actual_bn_key = bn_running_mean_key
    elif "module." + bn_running_mean_key in state_dict:
        actual_bn_key = "module." + bn_running_mean_key
    
    if actual_bn_key is not None:
        bn_mean = state_dict[actual_bn_key]
        bn_fusion = bn_mean + bias_sum
        state_dict[actual_bn_key] = bn_fusion
        print(f"  Quantized {layer_name} -> clip={clip_val:.2f}, fused bias into {bn_name}.running_mean")
    else:
        print(f"  Warning: BN layer {bn_name}.running_mean not found, bias not fused")
    
    return state_dict


def main():
    print("=" * 60)
    print("Adder Layer Weight Quantization for ResNet50-AdderNet")
    print("=" * 60)
    print(f"Quantization bits: {Q}")
    print(f"Default clip value: {DEFAULT_CLIP_VALUE}")
    print(f"Fine-grained clip control: Enabled (per-layer)")
    print(f"Input model: {MODEL_PATH}")
    print(f"Output model: {OUTPUT_PATH}")
    print()
    
    # Create model instance
    print("Loading model architecture...")
    model = resnet50.resnet50(act_bits=Q)

    # Load pretrained weights
    print(f"Loading pretrained weights from {MODEL_PATH}...")
    raw_loaded = torch.load(MODEL_PATH, map_location='cpu')
    state_dict = _extract_state_dict(raw_loaded)

    initialize_layer_lists_from_state_dict(state_dict)

    # Print all per-layer clip values
    print("Per-layer clip values:")
    for layer_name in ADDER_LAYER_NAMES:
        clip_val = CLIP_VALUES.get(layer_name, DEFAULT_CLIP_VALUE)
        print(f"  {layer_name}: {clip_val}")
    print()
    
    # Print some info about the state dict
    print(f"Total keys in state dict: {len(state_dict)}")
    
    # List all adder layers in the state dict
    adder_layers_in_model = [k for k in state_dict.keys() if '.adder' in k]
    print(f"Adder layers found in model: {len(adder_layers_in_model)}")
    for layer in adder_layers_in_model[:5]:
        print(f"  - {layer}")
    if len(adder_layers_in_model) > 5:
        print(f"  ... and {len(adder_layers_in_model) - 5} more")
    print()
    
    # Step 1: Apply Conv1 8-bit quantization
    print("=" * 60)
    print("Step 1: Applying Conv1 8-bit quantization...")
    print("=" * 60)
    state_dict, conv1_delta = apply_conv1_quantization(state_dict)
    print()
    
    # Step 2: Apply quantization to each adder layer
    print("=" * 60)
    print(f"Step 2: Applying quantization to {len(ADDER_LAYER_NAMES)} adder layers...")
    print("=" * 60)
    
    for i, layer_name in enumerate(ADDER_LAYER_NAMES):
        print(f"[{i}] Quantizing {layer_name}...")
        
        # Get clip value for this specific layer (fine-grained control)
        clip_val = CLIP_VALUES.get(layer_name, DEFAULT_CLIP_VALUE)
        
        # Apply quantization with the per-layer clip value
        state_dict = apply_quantization_to_layer(
            state_dict, 
            layer_name, 
            clip_val, 
            Q
        )
    
    print("=" * 60)
    print("Quantization complete!")
    print()
    
    # Convert per-layer clip values to ReLU clip format (49 values for ResNet50)
    relu_clip_values = clip_values_to_relu_format(CLIP_VALUES, DEFAULT_CLIP_VALUE)
    print("ReLU clip values (49 total):")
    print(f"  {relu_clip_values}")
    print()
    
    # Save the quantized model
    print(f"Saving quantized model to {OUTPUT_PATH}...")
    # Save state dict with clip values as metadata
    save_dict = {
        'state_dict': state_dict,
        'clip_values': CLIP_VALUES,
        'relu_clip_values': relu_clip_values,
        'default_clip': DEFAULT_CLIP_VALUE,
        'Q': Q,
        'conv1_quantized': True,
        'conv1_delta': conv1_delta if conv1_delta is not None else 0.0
    }
    torch.save(save_dict, OUTPUT_PATH)
    print("Done!")
    
    # Verify the saved model
    print("\nVerifying saved model...")
    # Use the load_quantized_model function to verify
    try:
        model = load_quantized_model(OUTPUT_PATH)
        print("Model loaded successfully with clip values!")
        
        # Print some statistics about the quantized weights
        print("\nQuantized weight statistics:")
        for i, layer_name in enumerate(ADDER_LAYER_NAMES[:3]):  # Show first 3
            if hasattr(model, layer_name.replace('.', '.')):
                w = eval(f"model.{layer_name.replace('.', '.')}.weight")
                print(f"  {layer_name}:")
                print(f"    Shape: {w.shape}")
                print(f"    Min: {w.min().item():.4f}")
                print(f"    Max: {w.max().item():.4f}")
                print(f"    Mean: {w.mean().item():.4f}")
                print(f"    Unique values: {len(w.unique())}")
    except Exception as e:
        print(f"Warning: Could not load model with clip values: {e}")
        print("Falling back to direct state dict load...")
        loaded = torch.load(OUTPUT_PATH, map_location='cpu')
        if isinstance(loaded, dict) and 'state_dict' in loaded:
            loaded_state = loaded['state_dict']
        else:
            loaded_state = loaded
        print(f"Keys in saved model: {len(loaded_state)}")
    
    print("\n" + "=" * 60)
    print("Quantization completed successfully!")
    print("=" * 60)


def load_quantized_model(model_path, clip_values_dict=None, default_clip=3.0):
    """
    Load quantized model with clip values for inference.
    
    Args:
        model_path: Path to the quantized model .pth file
        clip_values_dict: Dict of per-layer clip values (if None, will use default)
        default_clip: Default clip value if not provided
    
    Returns:
        model: ResNet50 model with clip values applied
    
    Example:
        # Load with custom clip values
        model = load_quantized_model(
            ".../ResNet50-AdderNet-quantized.pth",
            CLIP_VALUES,  # Your per-layer clip dict
            DEFAULT_CLIP_VALUE
        )
    
        # Or load with clip values saved in the model file
        model = load_quantized_model(".../ResNet50-AdderNet-quantized.pth")
    """
    # Load the saved model
    saved_data = torch.load(model_path, map_location='cpu')
    
    # Handle both old format (just state_dict) and new format (dict with metadata)
    if isinstance(saved_data, dict) and 'state_dict' in saved_data:
        state_dict = saved_data['state_dict']
        # Try to get clip_values from saved data
        if clip_values_dict is None and 'clip_values' in saved_data:
            clip_values_dict = saved_data['clip_values']
        if default_clip == 3.0 and 'default_clip' in saved_data:
            default_clip = saved_data['default_clip']
        # print(f"Loaded model metadata: clip_values={clip_values_dict}, default_clip={default_clip}")
        print(f"Loaded model metadata: default_clip={default_clip}")

    else:
        state_dict = saved_data
        print("Warning: Loading old format model (no clip metadata)")
    
    # If globals are not initialized yet, initialize from this checkpoint
    initialize_layer_lists_from_state_dict(state_dict)

    # Convert per-layer clip values to ReLU format (49 values)
    if clip_values_dict is None:
        clip_values_dict = CLIP_VALUES
    
    relu_clip_values = clip_values_to_relu_format(clip_values_dict, default_clip)
    
    print(f"Creating model with ReLU clip values: {relu_clip_values}")
    
    q_bits = Q
    if isinstance(saved_data, dict) and 'Q' in saved_data:
        try:
            q_bits = int(saved_data['Q'])
        except Exception:
            q_bits = Q

    # Create model with clip values
    model = resnet50.resnet50(clip_values=relu_clip_values, act_bits=q_bits)
    
    # Load state dict - handle module prefix
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        # Handle module prefix issue - strip 'module.' prefix from keys
        print("Detected module prefix mismatch, trying to fix...")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    
    return model


if __name__ == "__main__":
    main()
