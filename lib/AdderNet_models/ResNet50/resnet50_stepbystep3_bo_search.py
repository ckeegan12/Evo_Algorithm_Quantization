import os
import time
import argparse
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import optuna

from resnet50_actQ import resnet50
from resnet50_quantize_adder_weights import ADDER_LAYER_NAMES, apply_quantization_to_layer, clip_values_to_relu_format, DEFAULT_CLIP_VALUE
from contextlib import contextmanager
import sys


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


def load_checkpoint_state(model_path):
    # normalize path and support legacy "models/" by redirecting to "lib/models/"
    resolved = model_path
    if not os.path.isfile(resolved):
        if model_path.startswith("models/"):
            candidate = os.path.join("lib", model_path)  # models/x -> lib/models/x
            if os.path.isfile(candidate):
                resolved = candidate

    if not os.path.isfile(resolved):
        raise FileNotFoundError(resolved)

    checkpoint = torch.load(resolved, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    # ensure module. prefix for consistency
    if not any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {('module.' + k): v for k, v in state_dict.items()}
    return state_dict


def quantize_conv1_and_fc(state_dict):
    # conv1
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

    # fc
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


def objective_factory(model_path, device, val_loader, q, n_proxy_batches):
    def objective(trial, verbose=True):
        # propose 18 per-adder clip values
        adder_vals = []
        # round sampled clip values to 2 decimals before using them
        for i, name in enumerate(ADDER_LAYER_NAMES):
            # search range changed to [0.1, 4.0]
            v = trial.suggest_float(f'clip_{i}', 1.0, 3.5, step=0.1)
            # round to 1 decimal place before using in quantization/model
            v = float(round(v, 1))
            adder_vals.append(v)

        # build adder_clip_map
        adder_clip_map = {name: adder_vals[i] for i, name in enumerate(ADDER_LAYER_NAMES)}

        # Print only the rounded clip values for this trial (suppress Optuna's high-precision params)
        try:
            rounded_str = ','.join([f"{v:.1f}" for v in adder_vals])
            print(f"Trial {trial.number} rounded_clips: {rounded_str}")
        except Exception:
            pass

        # load fresh state_dict per trial
        state_dict = load_checkpoint_state(model_path)

        # optionally suppress verbose prints from quantization routines
        @contextmanager
        def _suppress_stdout():
            old_stdout = sys.stdout
            try:
                sys.stdout = open(os.devnull, 'w')
                yield
            finally:
                try:
                    sys.stdout.close()
                except Exception:
                    pass
                sys.stdout = old_stdout

        if verbose:
            quantize_conv1_and_fc(state_dict)
            for layer_name in ADDER_LAYER_NAMES:
                clip_val = adder_clip_map.get(layer_name, DEFAULT_CLIP_VALUE)
                try:
                    state_dict = apply_quantization_to_layer(state_dict, layer_name, clip_val, Q=q)
                except Exception:
                    pass
            quantize_bn_params(state_dict)
        else:
            with _suppress_stdout():
                quantize_conv1_and_fc(state_dict)
                for layer_name in ADDER_LAYER_NAMES:
                    clip_val = adder_clip_map.get(layer_name, DEFAULT_CLIP_VALUE)
                    try:
                        state_dict = apply_quantization_to_layer(state_dict, layer_name, clip_val, Q=q)
                    except Exception:
                        pass
                quantize_bn_params(state_dict)

        # build model with relu clip values and act bits
        relu_clip_values = clip_values_to_relu_format(adder_clip_map, DEFAULT_CLIP_VALUE)
        act_bits_list = [int(q)] * 49
        model = resnet50(clip_values=relu_clip_values, act_bits=act_bits_list)
        model = torch.nn.DataParallel(model).to(device)
        model.load_state_dict(state_dict, strict=False)

        acc = validate(val_loader, model, device, max_batches=n_proxy_batches, verbose=False)
        # print proxy inference accuracy for this trial
        try:
            print(f"Trial {trial.number} proxy_acc: {acc:.3f}")
        except Exception:
            pass

        # print best-so-far proxy accuracy and corresponding trial (if available)
        try:
            study = getattr(trial, 'study', None)
            if study is not None:
                best = getattr(study, 'best_trial', None)
                if best is not None and best.value is not None:
                    print(f"Best so far proxy_acc: {best.value:.3f} (trial {best.number})")
        except Exception:
            pass
        # maximize accuracy
        return acc

    return objective


def main():
    parser = argparse.ArgumentParser(description='Optuna BO search for clip values')
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--q', type=int, default=4)
    parser.add_argument('--model_path', type=str, default='lib/models/ResNet50-AdderNet.pth')
    parser.add_argument('--n_proxy_batches', type=int, default=20, help='Number of validation batches per trial (proxy)')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--storage', type=str, default=None, help='Optuna storage URL (optional)')
    parser.add_argument('--study_name', type=str, default='clip_search')
    parser.add_argument('--sampler', type=str, default='cmaes', choices=['tpe', 'cmaes', 'nsga'], help='Sampler to use: tpe, cmaes or nsga (default: cmaes)')
    args = parser.parse_args()

    # reduce Optuna INFO logs so terminal does not print high-precision params
    try:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except Exception:
        pass

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

    parser.add_argument('--verbose', action='store_true', help='Enable verbose printing during quantization')
    args = parser.parse_args()

    # choose sampler
    sampler = None
    if args.sampler == 'cmaes':
        try:
            sampler = optuna.samplers.CmaEsSampler()
        except Exception as e:
            print(f"Warning: could not create CmaEsSampler() ({e}), falling back to default sampler.")
            sampler = None
    elif args.sampler == 'tpe':
        try:
            sampler = optuna.samplers.TPESampler()
        except Exception as e:
            print(f"Warning: could not create TPESampler() ({e}), falling back to default sampler.")
            sampler = None
    elif args.sampler == 'nsga':
        try:
            sampler = optuna.samplers.NSGAIISampler()
        except Exception as e:
            print(f"Warning: could not create NSGAIISampler() ({e}), falling back to default sampler.")
            sampler = None
    else:
        sampler = None  # default Optuna sampler (TPE)

    if args.storage:
        study = optuna.create_study(direction='maximize', study_name=args.study_name, storage=args.storage, load_if_exists=True, sampler=sampler)
    else:
        study = optuna.create_study(direction='maximize', study_name=args.study_name, sampler=sampler)

    # wrap objective to pass verbose
    def _obj(trial):
        return objective(trial, verbose=args.verbose)

    study.optimize(_obj, n_trials=args.trials)

    print('Best trial:')
    print(study.best_trial.params)
    print(f"Best value (acc): {study.best_value:.4f}")


if __name__ == '__main__':
    main()
