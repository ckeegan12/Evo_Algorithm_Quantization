# StepByStep4 Documentation (CD + CMA-ES Clip Search)

This document explains `zhwf_stepbystep4_bo_cd_search.py` in detail:

- end-to-end data flow
- quantization formulas used in this stage and imported modules
- optimization formulas (including Differential Evolution)
- how to implement a ResNet50 version of the same search pipeline

---

## 1) What StepByStep4 Does

`stepbystep4` runs a **two-stage search** for per-layer clip values used in adder-weight quantization:

1. **Coordinate Descent (CD) pre-search**
   - Scans each adder layer independently over a value grid (`min_val:step:max_val`)
   - Keeps values that pass a proxy accuracy threshold
   - Records best value per layer

2. **CMA-ES refinement (Optuna sampler)**
   - Builds multiple seed vectors from CD results
   - Enqueues them into Optuna
   - Optimizes full joint clip vector for maximum proxy top-1 accuracy

The script reuses `objective_factory` and checkpoint handling from `zhwf_stepbystep3_bo_search.py`.

---

## 2) Data Flow (Detailed)

### Inputs

- model checkpoint (`--model_path`)
- quantization bitwidth (`--q`, default 4)
- proxy eval budget (`--n_proxy_batches`)
- CD search range (`--min_val`, `--max_val`, `--step`)
- acceptance threshold (`--threshold`)
- Optuna settings (`--storage`, `--study_name`, `--cma_trials`)

### Stage A: Coordinate Descent scan

For each layer in `ADDER_LAYER_NAMES`:

1. load fresh checkpoint state
2. quantize only current layer with candidate `clip=v`
3. quantize `conv1` and `fc` to 8-bit
4. quantize BN parameters to 16-bit
5. build quantized `resnet20` with matching ReLU clip list
6. evaluate proxy top-1 on CIFAR-10 validation loader
7. save:
   - passing values: `acc >= threshold`
   - best value by highest proxy accuracy

Outputs saved to JSON (`--output_json`):

- `per_dim_pass`: layer -> list of passing values
- `per_dim_best`: layer -> `{value, acc}`

### Stage B: Seed construction

Build seed vectors for CMA-ES:

- Seed 0: all per-layer best values from CD
- Other seeds: random picks from each layer’s passing set
- fallback order if empty set: `best -> DEFAULT_CLIP_VALUE`

### Stage C: CMA-ES joint optimization

1. create Optuna study (sampler=`CmaEsSampler` when available)
2. enqueue CD-derived seeds
3. objective samples clip parameters (`clip_0 ... clip_17`)
4. objective applies quantization + proxy validation
5. optimize for max proxy accuracy

---

## 3) Quantization Formulas Used

The following formulas are used directly in StepByStep3/4 and `zhwf_quantize_adder_weights.py`.

### 3.1 Conv1 / FC uniform quantization (8-bit)

For tensor $w$:

$$
\Delta = \frac{\max(w)-\min(w)}{2^8-1}
$$

$$
w_q = \operatorname{round}\!\left(\frac{w}{\Delta}\right)\Delta
$$

If $\Delta=0$, code keeps original tensor.

### 3.2 BN parameter quantization (16-bit)

For BN tensors (`weight`, `bias`, `running_mean`, `running_var`):

$$
\Delta_{bn} = \frac{\max(t)-\min(t)}{2^{16}-1}, \quad
t_q = \operatorname{round}\!\left(\frac{t}{\Delta_{bn}}\right)\Delta_{bn}
$$

### 3.3 Adder weight quantization (per-layer clip)

Given adder weight tensor $w$, layer clip $c$, bitwidth $Q$:

1. **clip non-negative range**

$$
w_{nn}=\operatorname{clip}(w,0,c)
$$

2. **bias/fusion term**

$$
b = \sum_{i,j,k}\left|w - w_{nn}\right|
$$

(`sum` over input-channel and kernel dims, producing per-output-channel bias)

3. **uniform quantization**

$$
\Delta = \frac{c}{2^Q-1}, \quad
w_q = \operatorname{round}\!\left(\frac{w_{nn}}{\Delta}\right)\Delta
$$

4. **BN fusion**

$$
\mu_{bn}^{new} = \mu_{bn}^{old} + b
$$

### 3.4 Activation quantization (ResNet20 actQ)

For activation tensor $x$ after ReLU and clipping to $[0,c_a]$:

$$
\Delta_a = \frac{c_a}{2^{Q_a}-1}, \quad
x_q = \operatorname{round}\!\left(\frac{x}{\Delta_a}\right)\Delta_a
$$

Where `Q_a` is `act_bits` (global in current implementation).

---

## 4) Optimization Formulas in StepByStep4

### 4.1 Coordinate Descent acceptance rule

For each layer $d$ and grid value $v$:

$$
\text{acc}_{d,v} = \text{proxy\_validate}(\text{quantize with layer }d\leftarrow v)
$$

Keep in pass set if:

$$
\text{acc}_{d,v} \ge \tau
$$

where $\tau$ is `--threshold`.

Best per layer:

$$
v_d^* = \arg\max_v \text{acc}_{d,v}
$$

### 4.2 CMA-ES objective

For clip vector $\mathbf{c}=[c_1,\dots,c_{18}]$:

$$
f(\mathbf{c}) = \text{proxy\_top1\_accuracy}(\mathbf{c})
$$

Optimization target:

$$
\max_{\mathbf{c}} f(\mathbf{c})
$$

Parameter domain in code is discretized with step $0.1$ in `[1.0, 3.5]`.

---

## 5) Differential Evolution (from `Diff_evo.py`)

Although StepByStep4 uses CD + CMA-ES, your repo also includes DE. Core DE formulas:

### 5.1 Mutation

For target index $j$, pick distinct $a,b,c \neq j$:

$$
\mathbf{v}_j = \mathbf{x}_a + F(\mathbf{x}_b - \mathbf{x}_c)
$$

Then apply bounds:

$$
\mathbf{v}_j \leftarrow \operatorname{clip}(\mathbf{v}_j, \mathbf{lb}, \mathbf{ub})
$$

### 5.2 Binomial crossover

For each dimension $k$:

$$
u_{j,k}=
\begin{cases}
v_{j,k}, & \text{if } r_k \le CR \text{ or } k=j_{rand}\\
x_{j,k}, & \text{otherwise}
\end{cases}
$$

where $r_k\sim U(0,1)$ and $j_{rand}$ forces at least one mutant dimension.

### 5.3 Selection (maximization)

$$
\mathbf{x}_j \leftarrow
\begin{cases}
\mathbf{u}_j, & \text{if } f(\mathbf{u}_j) > f(\mathbf{x}_j)\\
\mathbf{x}_j, & \text{otherwise}
\end{cases}
$$

In your implementation:

- `F` = mutation factor
- `prob_mut` = crossover rate `CR`
- objective returns accuracy, so larger is better.

---

## 6) How to Implement a ResNet50 Version of StepByStep4

Current StepByStep4 is tightly coupled to ResNet20/CIFAR10 and a fixed 18-layer adder list. For ResNet50 you should generalize these points.

## 6.1 Model + dataset replacement

1. Replace model import:
   - from `zhwf_resnet20_actQ import resnet20`
   - to a quantization-aware ResNet50 constructor (recommended new file: `zhwf_resnet50_actQ.py`)
2. Replace CIFAR-10 loader with ImageNet-style loader and transforms.
3. Update `num_classes` if your checkpoint is not ImageNet-1000.

## 6.2 Build dynamic adder layer list

Do **not** hardcode 18 names. Build search dimensions from checkpoint keys:

- include keys ending with `.adder`
- keep deterministic ordering (sorted or model traversal order)

Then every place using `ADDER_LAYER_NAMES` becomes dynamic over this list.

## 6.3 Clip-to-activation mapping for ResNet50

For ResNet50 Bottleneck blocks, each block has 3 conv/adder ops and multiple ReLU sites.

Implement a mapping function equivalent to `clip_values_to_relu_format`, but for ResNet50 topology:

- define the exact ReLU index order expected by your `resnet50_actQ`
- map each searched adder layer clip to the ReLU(s) feeding that adder
- provide default clip for unmapped entries

## 6.4 Quantization utility generalization

Generalize these helpers so they work for both ResNet20/ResNet50:

- `quantize_conv1_and_fc(state_dict)`
- `quantize_bn_params(state_dict)`
- `apply_quantization_to_layer(...)`

Important for ResNet50:

- robust adder->BN mapping for Bottleneck (`conv1/bn1`, `conv2/bn2`, `conv3/bn3`)
- support optional `module.` prefixes

## 6.5 Objective and search-space updates

In `objective_factory`:

- sample one clip parameter per dynamic adder layer
- keep clipping range and step configurable (same `[1.0,3.5], step=0.1` or retune)
- use proxy evaluation batches to keep runtime tractable

CD stage:

- scan each ResNet50 adder layer independently (same CD logic)
- write `per_dim_pass` and `per_dim_best` with dynamic keys

## 6.6 Suggested implementation sequence

1. create `zhwf_resnet50_actQ.py` with clip-aware activation quantization
2. add dynamic adder/BN mapping utilities for ResNet50
3. clone step4 script into `zhwf_stepbystep4_resnet50_bo_cd_search.py`
4. swap dataset/model/objective wiring
5. test with tiny proxy setup (`n_proxy_batches=1`, small batch)
6. run short CD and short CMA smoke test

---

## 7) Practical Notes

- StepByStep4 resumes from existing CD JSON if present.
- Proxy accuracy is only a fast surrogate; always run full validation on final best clips.
- CMA-ES runtime can be high on ResNet50; start with fewer layers or narrower clip range for early debugging.
