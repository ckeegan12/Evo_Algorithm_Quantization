import time
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from zhwf_resnet20_actQ import resnet20
from zhwf_quantize_adder_weights import ADDER_LAYER_NAMES, apply_quantization_to_layer, clip_values_to_relu_format, DEFAULT_CLIP_VALUE


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(device, model_path="models/ResNet20-AdderNet.pth"):
    model = resnet20(act_bits=8).to(device)
    model = torch.nn.DataParallel(model)
    if not os.path.isfile(model_path):
        print(f"=> no checkpoint found at '{model_path}'")
        return model, False

    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    if not any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {('module.' + k): v for k, v in state_dict.items()}
    # Quantize conv1 weights to 8-bit if present in the checkpoint
    try:
        conv_key = None
        if 'module.conv1.weight' in state_dict:
            conv_key = 'module.conv1.weight'
        elif 'conv1.weight' in state_dict:
            conv_key = 'conv1.weight'

        if conv_key is not None and torch.is_tensor(state_dict[conv_key]):
            w = state_dict[conv_key].float()
            maxx = float(w.max())
            minn = float(w.min())
            levels = 2 ** 8 - 1
            delta_conv = (maxx - minn) / levels if levels > 0 else 0.0
            if delta_conv == 0.0:
                wq = w.clone()
            else:
                wq = torch.round(w / delta_conv) * delta_conv
            state_dict[conv_key] = wq
            print(f"=> conv1 quantized to 8-bit (delta={delta_conv:.6f})")
    except Exception as e:
        print(f"=> Warning: conv1 quantization skipped/failed: {e}")

    model.load_state_dict(state_dict, strict=False)
    print(f"=> loaded checkpoint '{model_path}'")
    return model, True


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0.0


def accuracy(output, target):
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()
        return (correct / target.size(0)) * 100.0


def validate(val_loader, model, device):
    meter = AverageMeter()
    model.eval()
    batch_size = getattr(val_loader, 'batch_size', 512)
    # warm-up to avoid first-batch overheads
    warmup_iters = 5
    with torch.no_grad():
        for _ in range(warmup_iters):
            x = torch.randn(batch_size, 3, 32, 32, device=device)
            _ = model(x)
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if device.type == 'cuda':
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                # use CUDA events for accurate GPU timing
                torch.cuda.synchronize()
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
                output = model(input)
                end_evt.record()
                torch.cuda.synchronize()
                batch_time = start_evt.elapsed_time(end_evt)
            else:
                input = input.to(device)
                target = target.to(device)
                start = time.perf_counter()
                output = model(input)
                batch_time = (time.perf_counter() - start) * 1000.0
            acc1 = accuracy(output, target)
            meter.update(acc1, input.size(0))
            print(f'Batch {i}: Avg Acc@1: {meter.avg:.3f}, Time: {batch_time:.2f}ms')
    return meter.avg


def main(q=4, clip_values_arg=None):
    device = get_device()
    print(f"=> device: {device}")

    # Load checkpoint state dict
    model_path = "models/ResNet20-AdderNet.pth"
    if not os.path.isfile(model_path):
        print(f"=> no checkpoint found at '{model_path}'")
        return

    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    if not any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {('module.' + k): v for k, v in state_dict.items()}

    # Conv1 8-bit quantization (if present)
    try:
        conv_key = None
        if 'module.conv1.weight' in state_dict:
            conv_key = 'module.conv1.weight'
        elif 'conv1.weight' in state_dict:
            conv_key = 'conv1.weight'

        if conv_key is not None and torch.is_tensor(state_dict[conv_key]):
            w = state_dict[conv_key].float()
            maxx = float(w.max())
            minn = float(w.min())
            levels = 2 ** 8 - 1
            delta_conv = (maxx - minn) / levels if levels > 0 else 0.0
            if delta_conv == 0.0:
                wq = w.clone()
            else:
                wq = torch.round(w / delta_conv) * delta_conv
            state_dict[conv_key] = wq
            print(f"=> conv1 quantized to 8-bit (delta={delta_conv:.6f})")
    except Exception as e:
        print(f"=> Warning: conv1 quantization skipped/failed: {e}")

    # Parse clip values argument: support single float, 18 adder-values, or 19 ReLU-values (csv)
    adder_clip_map = {name: DEFAULT_CLIP_VALUE for name in ADDER_LAYER_NAMES}
    if clip_values_arg:
        parts = [p.strip() for p in clip_values_arg.split(',') if p.strip()]
        try:
            nums = [float(p) for p in parts]
            if len(nums) == 1:
                # single value -> use for all adder layers
                for name in ADDER_LAYER_NAMES:
                    adder_clip_map[name] = nums[0]
            elif len(nums) == 18:
                # assume per-adder ordering
                for i, name in enumerate(ADDER_LAYER_NAMES):
                    adder_clip_map[name] = nums[i]
            elif len(nums) == 19:
                # interpret as ReLU 19-values; map back to adder layers
                relu_vals = nums
                # mapping adder -> relu index (as in quantize script)
                adder_to_relu = {
                    "layer1.0.conv1.adder": (0,),
                    "layer1.0.conv2.adder": (1,),
                    "layer1.1.conv1.adder": (2,),
                    "layer1.1.conv2.adder": (3,),
                    "layer1.2.conv1.adder": (4,),
                    "layer1.2.conv2.adder": (5,),
                    "layer2.0.conv1.adder": (6,),
                    "layer2.0.conv2.adder": (7,),
                    "layer2.1.conv1.adder": (8,),
                    "layer2.1.conv2.adder": (9,),
                    "layer2.2.conv1.adder": (10,),
                    "layer2.2.conv2.adder": (11,),
                    "layer3.0.conv1.adder": (12,),
                    "layer3.0.conv2.adder": (13,),
                    "layer3.1.conv1.adder": (14,),
                    "layer3.1.conv2.adder": (15,),
                    "layer3.2.conv1.adder": (16,),
                    "layer3.2.conv2.adder": (17,),
                }
                for name, relu_idxs in adder_to_relu.items():
                    # pick first associated relu idx
                    idx = relu_idxs[0]
                    if 0 <= idx < len(relu_vals):
                        adder_clip_map[name] = relu_vals[idx]
            else:
                print(f"Warning: --clip_values length {len(nums)} not recognized; using defaults")
        except Exception as e:
            print(f"Warning: failed to parse --clip_values: {e}; using defaults")

    # Apply quantization to all 18 adder layers using same Q
    print("=> Quantizing all adder layers with Q=", q)
    for layer_name in ADDER_LAYER_NAMES:
        clip_val = adder_clip_map.get(layer_name, DEFAULT_CLIP_VALUE)
        try:
            state_dict = apply_quantization_to_layer(state_dict, layer_name, clip_val, Q=q)
        except Exception as e:
            print(f"Warning: quantizing {layer_name} failed: {e}")

    # Convert per-adder clip map to 19-length ReLU clip list for model
    relu_clip_values = clip_values_to_relu_format(adder_clip_map, DEFAULT_CLIP_VALUE)

    # Build model with activation bits = q (same Q for activations)
    act_bits_list = [int(q)] * 19
    model = resnet20(clip_values=relu_clip_values, act_bits=act_bits_list)
    model = torch.nn.DataParallel(model).to(device)
    # Load quantized state_dict
    model.load_state_dict(state_dict, strict=False)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "./data_cifar10/",
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
        ),
        batch_size=512,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == 'cuda'),
    )

    print("=> validating")
    acc1 = validate(val_loader, model, device)
    print(f' * Final Acc@1 {acc1:.3f}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Validate ResNet20-AdderNet with quantized weights and activations')
    parser.add_argument('--q', type=int, default=4, help='Quantization bits for adder weights and activations')
    parser.add_argument('--clip_values', type=str, default=None,
                        help='Comma-separated clip values: single value, 18 adder-values, or 19 ReLU-values')
    args = parser.parse_args()

    main(q=args.q, clip_values_arg=args.clip_values)


