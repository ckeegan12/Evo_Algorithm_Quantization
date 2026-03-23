import argparse
import json
import os
import random
import time
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import optuna

from zhwf_resnet20_actQ import resnet20
from zhwf_quantize_adder_weights import ADDER_LAYER_NAMES, apply_quantization_to_layer, clip_values_to_relu_format, DEFAULT_CLIP_VALUE
from zhwf_stepbystep3_bo_search import load_checkpoint_state
from contextlib import contextmanager
import sys


def quantize_conv1_and_fc(state_dict):
    try:
        conv_key = 'module.conv1.weight' if 'module.conv1.weight' in state_dict else ('conv1.weight' if 'conv1.weight' in state_dict else None)
        if conv_key is not None and torch.is_tensor(state_dict[conv_key]):
            w = state_dict[conv_key].float()
            maxx = float(w.max())
            minn = float(w.min())
            levels = 2 ** 8 - 1
            delta_conv = (maxx - minn) / levels if levels > 0 else 0.0
            wq = w.clone() if delta_conv == 0.0 else torch.round(w / delta_conv) * delta_conv
            state_dict[conv_key] = wq
    except Exception:
        pass
    try:
        fc_key = 'module.fc.weight' if 'module.fc.weight' in state_dict else ('fc.weight' if 'fc.weight' in state_dict else None)
        if fc_key is not None and torch.is_tensor(state_dict[fc_key]):
            wf = state_dict[fc_key].float()
            maxf = float(wf.max())
            minf = float(wf.min())
            levels_f = 2 ** 8 - 1
            delta_fc = (maxf - minf) / levels_f if levels_f > 0 else 0.0
            wfq = wf.clone() if delta_fc == 0.0 else torch.round(wf / delta_fc) * delta_fc
            state_dict[fc_key] = wfq
    except Exception:
        pass


def quantize_bn_params(state_dict):
    for k in list(state_dict.keys()):
        if '.bn' in k and (k.endswith('.weight') or k.endswith('.bias') or k.endswith('.running_mean') or k.endswith('.running_var')):
            try:
                t = state_dict[k].float()
                maxv = float(t.max())
                minv = float(t.min())
                levels_bn = 2 ** 16 - 1
                delta_bn = (maxv - minv) / levels_bn if levels_bn > 0 else 0.0
                tq = t.clone() if delta_bn == 0.0 else torch.round(t / delta_bn) * delta_bn
                state_dict[k] = tq
            except Exception:
                continue


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


def validate(val_loader, model, device, max_batches=None, verbose=False):
    meter = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if max_batches is not None and i >= max_batches:
                break
            if device.type == 'cuda':
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
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
            if verbose:
                print(f'Batch {i}: Avg Acc@1: {meter.avg:.3f}, Time: {batch_time:.2f}ms')
    return meter.avg


def run_coordinate_descent(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            './data_cifar10/', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        ),
        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=(device.type=='cuda')
    )

    per_dim_pass = {name: [] for name in ADDER_LAYER_NAMES}
    per_dim_best = {name: (None, -1.0) for name in ADDER_LAYER_NAMES}  # (value, acc)

    values = np.arange(args.min_val, args.max_val + 1e-9, args.step)

    for idx, layer_name in enumerate(ADDER_LAYER_NAMES):
        print(f"CD: scanning layer {idx+1}/{len(ADDER_LAYER_NAMES)}: {layer_name}")
        for v in values:
            v_rounded = float(round(float(v), 1))
            try:
                state_dict = load_checkpoint_state(args.model_path)
            except FileNotFoundError:
                print('Model path not found:', args.model_path)
                return per_dim_pass, per_dim_best

            # apply quantization for this single layer
            try:
                state_dict = apply_quantization_to_layer(state_dict, layer_name, v_rounded, Q=args.q)
            except Exception:
                # skip if failing
                continue
            quantize_conv1_and_fc(state_dict)
            quantize_bn_params(state_dict)

            relu_clip_values = clip_values_to_relu_format({layer_name: v_rounded}, DEFAULT_CLIP_VALUE)
            act_bits_list = [int(args.q)] * 19
            model = resnet20(clip_values=relu_clip_values, act_bits=act_bits_list)
            model = torch.nn.DataParallel(model).to(device)
            model.load_state_dict(state_dict, strict=False)

            acc = validate(val_loader, model, device, max_batches=args.n_proxy_batches, verbose=False)
            print(f" layer {layer_name} val={v_rounded:.1f} -> proxy_acc={acc:.3f}")

            if acc >= args.threshold:
                per_dim_pass[layer_name].append(v_rounded)
            if acc > per_dim_best[layer_name][1]:
                per_dim_best[layer_name] = (v_rounded, acc)

    # save results
    out = {
        'per_dim_pass': per_dim_pass,
        'per_dim_best': {k: {'value': v[0], 'acc': v[1]} for k, v in per_dim_best.items()}
    }
    with open(args.output_json, 'w') as f:
        json.dump(out, f, indent=2)
    print('Coordinate descent done. Results saved to', args.output_json)
    return per_dim_pass, per_dim_best


def build_seeds(per_dim_pass, per_dim_best, n_seeds=5):
    seeds = []
    # seed 0: best per-dim
    seed0 = []
    for name in ADDER_LAYER_NAMES:
        best = per_dim_best.get(name, (None, -1.0))[0]
        if best is None:
            seed0.append(DEFAULT_CLIP_VALUE)
        else:
            seed0.append(best)
    seeds.append(seed0)

    # other seeds: random combination from passing sets (fallback to best or default)
    for s in range(1, n_seeds):
        seed = []
        for name in ADDER_LAYER_NAMES:
            vals = per_dim_pass.get(name, [])
            if vals:
                seed.append(float(random.choice(vals)))
            else:
                best = per_dim_best.get(name, (None, -1.0))[0]
                seed.append(best if best is not None else DEFAULT_CLIP_VALUE)
        seeds.append(seed)
    return seeds


def enqueue_seeds_and_run_cma(args, seeds):
    # create study with CMA-ES
    try:
        sampler = optuna.samplers.CmaEsSampler()
    except Exception as e:
        print('Warning: could not create CmaEsSampler():', e)
        sampler = None

    if args.storage:
        study = optuna.create_study(direction='maximize', study_name=args.study_name, storage=args.storage, load_if_exists=True, sampler=sampler)
    else:
        study = optuna.create_study(direction='maximize', study_name=args.study_name, sampler=sampler)

    # enqueue seeds
    for seed in seeds:
        params = {f'clip_{i}': float(seed[i]) for i in range(len(seed))}
        try:
            study.enqueue_trial(params)
        except Exception:
            pass

    # objective uses the same machinery as existing script: reuse its objective_factory
    from zhwf_stepbystep3_bo_search import objective_factory
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            './data_cifar10/', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        ),
        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=(device.type=='cuda')
    )
    objective = objective_factory(args.model_path, device, val_loader, args.q, args.n_proxy_batches)

    def _obj(trial):
        return objective(trial, verbose=False)

    study.optimize(_obj, n_trials=args.cma_trials)
    print('CMA-ES finished. Best trial:', study.best_trial.params, 'value:', study.best_value)


def main():
    parser = argparse.ArgumentParser(description='CD -> CMA-ES pipeline for clip search')
    parser.add_argument('--model_path', type=str, default='models/ResNet20-AdderNet.pth')
    parser.add_argument('--q', type=int, default=4)
    parser.add_argument('--n_proxy_batches', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--min_val', type=float, default=1.0)
    parser.add_argument('--max_val', type=float, default=3.5)
    parser.add_argument('--step', type=float, default=0.1)
    parser.add_argument('--threshold', type=float, default=90.0, help='proxy accuracy threshold (percent) to keep values per-dim')
    parser.add_argument('--output_json', type=str, default='cd_results.json')
    parser.add_argument('--storage', type=str, default=None, help='Optuna storage URI (sqlite:///...)')
    parser.add_argument('--study_name', type=str, default='cmaes_from_cd')
    parser.add_argument('--cma_trials', type=int, default=200)
    parser.add_argument('--seeds', type=int, default=5)
    args = parser.parse_args()
    # If CD results already exist and not forced, load them and skip CD
    if os.path.exists(args.output_json):
        try:
            with open(args.output_json, 'r') as f:
                data = json.load(f)
            per_dim_pass = data.get('per_dim_pass', {name: [] for name in ADDER_LAYER_NAMES})
            per_dim_best_raw = data.get('per_dim_best', {})
            per_dim_best = {name: (per_dim_best_raw.get(name, {}).get('value', None), per_dim_best_raw.get(name, {}).get('acc', -1.0)) for name in ADDER_LAYER_NAMES}
            print(f'Loaded existing CD results from {args.output_json}; skipping CD stage')
        except Exception as e:
            print('Failed to load existing CD results, running CD. Error:', e)
            per_dim_pass, per_dim_best = run_coordinate_descent(args)
    else:
        per_dim_pass, per_dim_best = run_coordinate_descent(args)
    seeds = build_seeds(per_dim_pass, per_dim_best, n_seeds=args.seeds)
    print('Generated seeds (first 3 shown):')
    for s in seeds[:3]:
        print(','.join([f'{v:.1f}' for v in s]))

    enqueue_seeds_and_run_cma(args, seeds)


if __name__ == '__main__':
    main()
