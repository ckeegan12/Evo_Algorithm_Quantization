"""
ResNet20-AdderNet: DE-Optimized Activation Clip Values
=======================================================
1. Loads the ResNet20-AdderNet checkpoint and validates a baseline accuracy
   (Q=4, default clip=6.0)
2. Defines an objective function that quantizes the model with a given set of
   19 per-ReLU clip values and returns CIFAR-10 validation accuracy
3. Runs Differential Evolution (Diff_evo.DE) to find clip values that maximise
   accuracy
4. Reports and saves convergence plots
"""

# ── path setup ─────────────────────────────────────────────────────────────────
import sys
import os

TEST_DE_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_DIR     = os.path.dirname(TEST_DE_DIR)

for p in [TEST_DE_DIR, LIB_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

print("TEST_DE_DIR :", TEST_DE_DIR)
print("LIB_DIR     :", LIB_DIR)

# ── standard imports ───────────────────────────────────────────────────────────
import time
import copy
import tempfile
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

# Project modules
from zhwf_resnet20_actQ import resnet20
from zhwf_quantize_adder_weights import (
    ADDER_LAYER_NAMES,
    apply_quantization_to_layer,
    clip_values_to_relu_format,
    DEFAULT_CLIP_VALUE,
)
from Diff_evo import DE

print("All imports OK")

# ── configuration ──────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(LIB_DIR, "models", "AdderNet_model.pth")
DATA_DIR    = tempfile.gettempdir()   # CIFAR-10 downloaded to system temp dir
BATCH_SIZE  = 512
NUM_WORKERS = 0        # set >0 on Linux/Mac for speed; keep 0 on Windows
Q_BITS      = 4        # adder-weight quantisation bits

# DE hyper-parameters
DE_POP_SIZE = 25       # population size
DE_MAX_ITER = 50       # number of generations
DE_F        = 0.8      # mutation factor
DE_CR       = 0.7      # crossover rate (prob_mut)
CLIP_LB     = 1.0      # lower bound for each clip value
CLIP_UB     = 12.0     # upper bound for each clip value
N_DIM       = 19       # 19 ReLU clip values per candidate solution

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device    :", DEVICE)
print("Model path:", MODEL_PATH, "| exists:", os.path.isfile(MODEL_PATH))

# ── load raw checkpoint ────────────────────────────────────────────────────────
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location='cpu')
BASE_SD    = checkpoint.get('state_dict', checkpoint)

# Ensure 'module.' prefix for DataParallel compatibility
if not any(k.startswith('module.') for k in BASE_SD.keys()):
    BASE_SD = {'module.' + k: v for k, v in BASE_SD.items()}

# Quantise conv1 to Q_BITS (currently 4-bit)
for conv_key in ('module.conv1.weight', 'conv1.weight'):
    if conv_key in BASE_SD and torch.is_tensor(BASE_SD[conv_key]):
        w      = BASE_SD[conv_key].float()
        levels = 2 ** Q_BITS - 1
        delta  = (w.max() - w.min()) / levels
        BASE_SD[conv_key] = torch.round(w / delta) * delta if delta > 0 else w
        print(f"conv1 quantised to {Q_BITS}-bit  (delta={float(delta):.6f})")
        break

print("Checkpoint loaded — keys:", len(BASE_SD))

# ── CIFAR-10 validation loader ─────────────────────────────────────────────────
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

val_dataset = datasets.CIFAR10(
    root=DATA_DIR, train=False, download=True, transform=val_transform
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == 'cuda'),
)
print(f"Val samples: {len(val_dataset)}  |  batches: {len(val_loader)}")

# ── helpers ────────────────────────────────────────────────────────────────────
def validate_model(model, loader, device):
    """Run the model over the full validation set and return average top-1 accuracy."""
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs  = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            pred    = outputs.argmax(dim=1)
            total_correct += pred.eq(targets).sum().item()
            total_samples += targets.size(0)
    return (total_correct / total_samples) * 100.0


def build_model_from_clip_values(clip_values_19, q=Q_BITS, base_sd=None):
    """
    Given 19 per-ReLU clip values, quantise the adder weights and build
    the full ResNet20-AdderNet, ready for inference.

    Parameters
    ----------
    clip_values_19 : array-like of length 19
        Per-ReLU activation clip values.
    q              : int
        Number of bits for adder-weight and activation quantisation.
    base_sd        : dict, optional
        Pre-loaded checkpoint state_dict. Defaults to the global BASE_SD.

    Returns
    -------
    model : nn.DataParallel
        Quantised model moved to DEVICE.
    """
    if base_sd is None:
        base_sd = BASE_SD

    sd = copy.deepcopy(base_sd)

    # Map 19-length ReLU clip list → per-adder clip map (18 layers)
    # ReLU index 0       → initial conv ReLU (not an adder layer)
    # ReLU indices 1..18 → adder layers in order
    adder_clip_map = {}
    for idx, name in enumerate(ADDER_LAYER_NAMES):
        relu_idx = idx + 1
        clip_val = float(clip_values_19[relu_idx]) if relu_idx < len(clip_values_19) else DEFAULT_CLIP_VALUE
        adder_clip_map[name] = clip_val

    # Quantise each adder layer's weights
    for layer_name in ADDER_LAYER_NAMES:
        clip_val = adder_clip_map.get(layer_name, DEFAULT_CLIP_VALUE)
        try:
            sd = apply_quantization_to_layer(sd, layer_name, clip_val, Q=q)
        except Exception as e:
            print(f"Warning: quantising {layer_name} failed: {e}")

    # Convert per-adder map → 19-length relu clip list that resnet20() expects
    relu_clip_values = clip_values_to_relu_format(adder_clip_map, DEFAULT_CLIP_VALUE)

    act_bits_list = [int(q)] * 19
    model = resnet20(clip_values=relu_clip_values, act_bits=act_bits_list)
    model = torch.nn.DataParallel(model).to(DEVICE)
    model.load_state_dict(sd, strict=False)
    return model


# ── baseline ───────────────────────────────────────────────────────────────────
default_clips = [DEFAULT_CLIP_VALUE] * N_DIM    # 19 × 6.0

print(f"\nBuilding baseline model  (Q={Q_BITS}, clip={DEFAULT_CLIP_VALUE}) …")
t0             = time.time()
baseline_model = build_model_from_clip_values(default_clips, q=Q_BITS)
baseline_acc   = validate_model(baseline_model, val_loader, DEVICE)
del baseline_model

print(f" Baseline Acc@1 = {baseline_acc:.3f}%   ({time.time() - t0:.1f}s)")

# ── DE objective function ──────────────────────────────────────────────────────
_eval_counter = [0]

def objective(clip_values_vec):
    """
    Objective function for DE.
    clip_values_vec : 1-D numpy array of length N_DIM (19 clip values)
    Returns         : float — CIFAR-10 top-1 accuracy (%)
    """
    _eval_counter[0] += 1
    clip_values_vec = np.clip(clip_values_vec, CLIP_LB, CLIP_UB)

    try:
        model = build_model_from_clip_values(clip_values_vec, q=Q_BITS)
        acc   = validate_model(model, val_loader, DEVICE)
        del model
    except Exception as e:
        print(f"  [eval #{_eval_counter[0]}] ERROR: {e}")
        acc = 0.0

    print(f"  [eval #{_eval_counter[0]:>4d}]  "
          f"clips = {np.round(clip_values_vec, 2).tolist()}  →  acc = {acc:.3f}%")
    return acc


# ── set up and run DE ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"DE config:  pop={DE_POP_SIZE}  gens={DE_MAX_ITER}  "
      f"F={DE_F}  CR={DE_CR}  dims={N_DIM}")
print(f"Clip bounds: [{CLIP_LB}, {CLIP_UB}]")
print("=" * 60)

de = DE(
    func     = objective,
    F        = DE_F,
    lb       = CLIP_LB,
    ub       = CLIP_UB,
    size_pop = DE_POP_SIZE,
    n_dim    = N_DIM,
    max_iter = DE_MAX_ITER,
    prob_mut = DE_CR,
)

t_start          = time.time()
best_x, best_acc = de.run()
elapsed          = time.time() - t_start

print("\n" + "=" * 60)
print(f"DE finished in {elapsed / 60:.1f} min  ({_eval_counter[0]} model evals)")
print(f"Best Acc@1 : {best_acc:.3f}%")
print(f"Best clips : {np.round(best_x, 3).tolist()}")
print(f"Baseline   : {baseline_acc:.3f}%")
print(f"Δ accuracy : {best_acc - baseline_acc:+.3f}%")

# ── convergence plot ───────────────────────────────────────────────────────────
gen_best = de.generation_best_Y
gens     = list(range(1, len(gen_best) + 1))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(gens, gen_best, marker='o', linewidth=2, label='DE best')
ax.axhline(baseline_acc, color='gray', linestyle='--',
           label=f'Baseline ({baseline_acc:.2f}%)')
ax.set_xlabel('Generation')
ax.set_ylabel('Top-1 Accuracy (%)')
ax.set_title('DE Convergence – Activation Clip Optimisation')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('de_convergence.png', dpi=150)
plt.close()
print("Plot saved to de_convergence.png")

# ── best clip values bar chart ─────────────────────────────────────────────────
labels = [f'ReLU {i}' for i in range(N_DIM)]

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(labels, best_x, color='steelblue', alpha=0.8)
ax.axhline(DEFAULT_CLIP_VALUE, color='tomato', linestyle='--',
           label=f'Default clip ({DEFAULT_CLIP_VALUE})')
ax.set_xlabel('ReLU Layer Index')
ax.set_ylabel('Clip Value')
ax.set_title('Optimised Per-ReLU Activation Clip Values')
ax.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('de_best_clips.png', dpi=150)
plt.close()
print("Bar chart saved to de_best_clips.png")

# ── summary table ──────────────────────────────────────────────────────────────
print("\n── Summary ─────────────────────────────────")
print(f"  Q bits          : {Q_BITS}")
print(f"  Baseline Acc@1  : {baseline_acc:.3f}%")
print(f"  DE Best  Acc@1  : {best_acc:.3f}%")
print(f"  Improvement     : {best_acc - baseline_acc:+.3f}%")
print(f"  Generations     : {DE_MAX_ITER}")
print(f"  Population size : {DE_POP_SIZE}")
print(f"  Total evals     : {_eval_counter[0]}")
print("─────────────────────────────────────────────")
print("Best clip values (19 ReLU layers):")
for lbl, v in zip(labels, best_x):
    print(f"  {lbl:<10s}  {v:.4f}")
