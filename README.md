# LatEq

**Laterality-Equivariant Learning for Anatomical Image Classification**

A minimal PyTorch implementation of the method in the LatEq manuscript.
One module implements the anatomical label action, ActionTrain objective,
and aligned probability projection. It plugs into an existing classifier
and adds no trainable parameters.

This release contains method code and documentation: no model weights,
dataset files, training loops, evaluation scripts, or experiment infrastructure.

![LatEq: anatomical label action, action-aware training, and aligned probability projection](assets/overview.svg)

## Method

A reflection that exchanges anatomical laterality also exchanges the
corresponding left/right labels. LatEq applies the same known class action
during supervision and prediction alignment.

Let $F$ denote the input reflection and $P$ the class permutation, with
$F^2=P^2=I$. For classifier logits $f_\theta$ and probabilities
$p=\operatorname{softmax}(f_\theta)$:

**Action-aware training — Fig. 1(b), Eq. (3)**

$$
\mathcal{L}_{\mathrm{act}} =
\frac{1}{2}\operatorname{CE}(f_\theta(x),y) +
\frac{1}{2}\operatorname{CE}(f_\theta(Fx),Py).
$$

**Aligned probability projection — Fig. 1(c), Eq. (4)**

$$
q(x) = \frac{1}{2}\left[p(x)+Pp(Fx)\right].
$$

For a deterministic predictor, this readout satisfies $q(Fx)=Pq(x)$.
Projection also makes pair-averaged cross entropy non-increasing. Accuracy
is measured empirically. The implementation averages aligned **probabilities**;
each view is passed through softmax first.

## Files

```text
ICASSP_LATEQ/
├── lateq.py              # Method implementation
├── requirements.txt     # PyTorch dependency
├── README.md            # Method, usage, and manuscript results
└── assets/
    └── overview.svg     # Original manuscript method figure
```

## Usage

Use Python 3.10 or later with PyTorch:

```bash
pip install -r requirements.txt
```

Validated with Python 3.12.2 and PyTorch 2.14.0 on CPU using synthetic tensors:
2-D/3-D reflection, hard/soft-target gradients, probability equivariance,
projection idempotence, and the pair-averaged cross-entropy inequality.

The snippets below integrate with an existing `model`, input batch `x`,
target batch `y`, and `device`. The model returns logits of shape `[N, K]`.
Use the same classifier for the original and reflected inputs.

```python
from lateq import LatEq

# OrganAMNIST / OrganCMNIST class order; x has shape [N, C, H, W].
permutation = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 10]
action = LatEq(permutation, flip_dim=2).to(device)

# Fig. 1(b): paired supervision, ready for the caller's optimizer.
model.train()
logits = model(x)
reflected_logits = model(action.reflect(x))
loss = action.loss(logits, reflected_logits, y)
```

The projected readout uses only the two sets of logits:

```python
import torch

# Fig. 1(c): deterministic classifier evaluation, with no labels required.
model.eval()
with torch.no_grad():
    probabilities = action(model(x), model(action.reflect(x)))
    predictions = probabilities.argmax(dim=1)
```

`LatEq` leaves classifier execution, device placement and optimizer control
with the caller. Place `model`, `x`, `y`, and `action` on the same device.
Evaluation mode disables standard dropout and fixes BatchNorm statistics;
the equivariance relation assumes the classifier evaluates deterministically.

### Hard and soft targets

`action.loss(...)` accepts either:

- Hard labels: `[N]` integer class IDs, `dtype=torch.long`.
- Soft targets: `[N, K]` floating-point probabilities, nonnegative with each
  row summing to one.

For a mixed target $t=\lambda y_i+(1-\lambda)y_j$, linearity gives
$Pt=\lambda Py_i+(1-\lambda)Py_j$. Pass the mixed images and their soft targets
through the same interface; their mixing coefficients are preserved.

### Anatomical action and tensor axes

`permutation[c]` is the reflected class ID for original class `c`.
Classes unchanged by the action map to themselves. Initialization checks
that the mapping is a bijection and that applying it twice gives the identity.

The manuscript uses the following task-specific mappings:

| Task | Swapped class pairs | Permutation |
|---|---|---|
| OrganAMNIST / OrganCMNIST | Femur `(1, 2)`, kidney `(4, 5)`, lung `(7, 8)` | `[0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 10]` |
| OrganMNIST3D | Kidney `(1, 2)`, femur `(3, 4)`, lung `(7, 8)` | `[0, 2, 1, 4, 3, 5, 6, 8, 7, 9, 10]` |

For the manuscript's stored input order, the reflection uses `flip_dim=2`:
the first spatial dimension of `[N, C, H, W]` or `[N, C, D, H, W]`.
For another input convention, specify the spatial dimension representing
the intended anatomical action. A tensor's horizontal display direction
alone does not determine its anatomical left/right axis.

The manuscript checks this choice using training class means. For known
left/right class pairs $\mathcal S$ and candidate spatial axis $j$:

$$
d_j=\frac{1}{|\mathcal S|}\sum_{(l,r)\in\mathcal S}
\operatorname{MSE}(F_j\mu_l,\mu_r).
$$

The selected axis minimizes $d_j$ and improves on the unreflected discrepancy.
Sagittal inputs favor no in-plane reflection and are excluded from the main
experiments. This diagnostic checks a specified anatomical coordinate convention.

### API and paper correspondence

| API | Output | Manuscript |
|---|---|---|
| `action.reflect(x)` | Reflected images or volumes, same shape as `x` | Fig. 1(a) |
| `action.transform_targets(y)` | Reflected hard or soft targets | Fig. 1(a) |
| `action.loss(logits, reflected_logits, y)` | Scalar ActionTrain loss | Fig. 1(b), Eq. (3) |
| `action(logits, reflected_logits)` | Aligned probabilities `[N, K]` | Fig. 1(c), Eq. (4) |

The module implements the primary method, **ActionTrain + group averaging**.
The additional symmetric-KL penalty is an ablation in the manuscript.

## Results from the manuscript

The tables summarize the current manuscript's frozen experimental results.
All paper experiments use NVIDIA A800 GPUs.

### Training benefit under a common readout

Reflection-balanced accuracy is the average accuracy over original inputs
with targets $y$ and reflected inputs with targets $Py$. Both methods below
use the same group-averaged readout. Accuracy is reported as mean ± sample
standard deviation over three seeds; gains are in percentage points (pp).

| Task / backbone | ERM + GA (%) | LatEq (%) | Gain (pp), paired 95% CI |
|---|---:|---:|---:|
| OrganMNIST3D / Small3D | 90.22 ± 3.08 | 94.86 ± 1.12 | +4.64 [3.22, 6.17] |
| OrganAMNIST / Small2D | 89.93 ± 0.08 | 91.41 ± 1.04 | +1.48 [1.23, 1.72] |
| OrganAMNIST / ResNet-18 | 91.71 ± 0.46 | 92.99 ± 0.69 | +1.28 [1.04, 1.52] |
| OrganCMNIST / Small2D | 85.98 ± 1.15 | 90.32 ± 0.27 | +4.34 [3.91, 4.78] |
| OrganCMNIST / ResNet-18 | 86.86 ± 1.40 | 91.39 ± 0.38 | +4.53 [4.09, 4.96] |

Source: manuscript Table 1 and the paired statistical report. Confidence
intervals use 10,000 paired resamples of released benchmark samples after
averaging sample-level accuracy over seeds, conditional on the fitted models.

### Recent-method comparison

Reflection-balanced accuracy (%) on the two 2-D tasks with ResNet-18,
using the same splits, optimizer, 30 epochs and three seeds.

| Method adaptation | OrganAMNIST | OrganCMNIST |
|---|---:|---:|
| RegMixup + GA | 89.18 ± 0.76 | 86.38 ± 0.63 |
| HMix + GA | 88.16 ± 0.91 | 85.80 ± 2.45 |
| GMix + GA | 90.90 ± 0.77 | 86.78 ± 0.84 |
| PRIME + GA | 87.47 ± 1.72 | 86.60 ± 0.51 |
| SoftAug + GA | 92.83 ± 0.51 | 89.09 ± 1.17 |
| CC-Flip + GA | 91.62 ± 0.92 | 90.54 ± 0.09 |
| EquiZero | 91.74 ± 0.41 | 86.85 ± 1.43 |
| ERM + GA | 91.71 ± 0.46 | 86.86 ± 1.40 |
| TSE-Z2 + GA | 90.27 ± 0.90 | 86.34 ± 2.77 |
| **LatEq** | **92.99 ± 0.69** | **91.39 ± 0.38** |

Source: manuscript Table 2. These are task adaptations under the paper's
grayscale protocol. Generic competitors share the known-permutation GA
readout; EquiZero selects the minimum-entropy aligned orbit member.
CC-Flip reflects only fixed classes, and TSE-Z2 uses reflected-kernel softness
0.5. ERM + GA also represents the equivalent two-element
Equi-Tuning-style / minimal-frame averaging readout.

On OrganAMNIST, the difference from SoftAug is +0.16 pp with paired CI
[-0.12, 0.43]; SoftAug leads macro-F1 and ERM + GA leads paired-class accuracy.
On OrganCMNIST, the gain over CC-Flip is +0.86 pp with CI [0.58, 1.13],
and LatEq leads all four metrics reported in the paper. Differences are
computed before rounding.

### Mixed-target compatibility

On axial ResNet-18, applying the correct target action with the same GA readout
raises MixUp accuracy from 84.79% to 90.97% (+6.18 pp) and CutMix from 83.50%
to 93.48% (+9.98 pp). These are the one-seed compatibility experiments in
manuscript Table 5.

## Figure attribution

The overview is the manuscript's original horizontal method figure. Its
medical image is an OrganCMNIST+ 224-pixel training crop from
[MedMNIST+ v3.0](https://zenodo.org/records/10519652), distributed by the MedMNIST
authors under CC BY 4.0. Display uses a transpose and an exact reflection of
the same sample. Contours, organ icons and probability bars are schematic.
The paper experiments use 28-pixel inputs. The figure is included as a
documentation asset; no separate medical sample arrays are distributed here.

MedMNIST reference: Jiancheng Yang et al., *MedMNIST v2 — A large-scale
lightweight benchmark for 2D and 3D biomedical image classification*,
Scientific Data 10, 41 (2023).
