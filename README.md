# MedHE: Communication-Efficient Privacy-Preserving Federated Learning for Healthcare

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.3](https://img.shields.io/badge/PyTorch-2.3.0-orange.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.09043-b31b1b.svg)](https://arxiv.org/abs/2511.09043)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of:

> Yesmin, F. (2026). *MedHE: Communication-Efficient Privacy-Preserving Federated
> Learning with Adaptive Gradient Sparsification for Healthcare.*
> Under review, CIBB 2026.
> Preprint: [arXiv:2511.09043](https://arxiv.org/abs/2511.09043)

Part of the [FairHealth](https://github.com/Farjana-Yesmin/fairhealth) library —
`pip install fairhealth`

---

## What MedHE Does

MedHE co-designs **adaptive gradient sparsification** with **CKKS homomorphic
encryption** for privacy-preserving healthcare NLP. Applied to binary drug
effectiveness classification using DistilBERT across 5 federated clients, MedHE:

- Reduces communication from **1,277 MB → 32 MB** (97.5% reduction)
- Maintains **macro-F1 = 0.950 ± 0.005** (statistically equivalent to standard FL, p=0.32)
- Achieves **MIA resistance of 51.1%** (near-random; ideal = 50%)
- Satisfies **differential privacy** with ε ≤ 1.0, δ = 1×10⁻⁵

---

## Results

### Performance Comparison (5 independent trials, mean ± std)

| Method | Macro-F1 | Accuracy | Communication (MB) |
|---|---|---|---|
| Standard FedAvg | 0.944 ± 0.006 | 89.9 ± 0.7% | 1,277 |
| HE-only FL | 0.714 ± 0.126 | 90.4 ± 2.2% | 1,490 |
| **MedHE (ours)** | **0.950 ± 0.005** | **89.5 ± 0.8%** | **32** |

MedHE vs Standard FedAvg: p=0.32 (not significant) — accuracy preserved.
HE-only FL collapses to majority-class bias (F1=0.714) without sparsification.

### Privacy Evaluation (Membership Inference Attacks)

| Method | Worst-case MIA Rate | Privacy Level |
|---|---|---|
| Random guessing (ideal) | 50.0% | Perfect |
| **MedHE (ours)** | **51.1%** | **Strong** |
| Standard FedAvg | 56.3% | Weak |

### Ablation Study

| Configuration | Macro-F1 | Accuracy | Comm (MB) | MIA |
|---|---|---|---|---|
| Full MedHE | 0.950 | 89.5% | 32 | 51.1% |
| − Error feedback | 0.763 | 91.2% | 32 | 51.1% |
| − Adaptive threshold | 0.773 | 91.2% | 32 | 51.1% |
| − Batch packing | 0.950 | 89.5% | 415 | 51.1% |
| − HE (sparsity only) | 0.755 | 90.9% | 26 | 56.3% |

Error feedback and adaptive thresholding are essential (F1 drops ~19% when either
is removed). Removing HE eliminates privacy while saving only 6MB.

### Sparsity Sensitivity

Optimal sparsity = 0.9 (90%):
- s < 0.8: communication savings < 80%, no meaningful privacy benefit
- s = 0.9: accuracy preserved, 97.5% comm reduction ✓
- s > 0.95: accuracy drops below 85%

---

## MedHE Framework

### Three-Component Architecture
Client gradient G ∈ ℝᵈ
↓
Step 1: Adaptive Sparsification (top-10% by magnitude)
+ Error-feedback compensation (residual added next round)
+ EMA threshold τₜ = α·τₜ₋₁ + (1-α)·τcurrent, α=0.9
↓
Step 2: CKKS Batch Packing
64 gradient values per ciphertext slot
13 ciphertexts total (vs 800+ naive)
Each ciphertext ≈ 0.47 MB
Ring dimension N=8,192, 128-bit security (RLWE)
↓
Step 3: Differential Privacy
Gaussian noise N(0, σ²I)
ε ≤ 1.0, δ = 1×10⁻⁵ across T=3 rounds
90% sparsity amplifies privacy by factor (1-s)=0.1
↓
Encrypted upload to aggregation server
Server: ciphertext-domain weighted averaging → broadcast
### Convergence

Both MedHE and Standard FedAvg converge by round 3 of 10.
Training time: 8.7 min (MedHE) vs 7.2 min (baseline) — only 21% overhead.

---

## Dataset

**UCI Drug Review Dataset** (ID 461)
- ~4,140 patient reviews of pharmaceutical drugs
- Task: binary classification (effective if rating ≥ 3)
- 86% majority class → macro-F1 is the primary metric (accuracy misleading)
- Federated setup: 5 clients, equal partition (~660 samples each)
- 3 communication rounds, 2 local epochs per round

```python
from ucimlrepo import fetch_ucirepo
drug_review = fetch_ucirepo(id=461)
```

---

## Quick Start

```bash
pip install transformers==4.41.2 torch==2.3.0 tenseal==0.3.16
pip install scikit-learn pandas ucimlrepo
```

```python
# Run the main notebook
# CIBB'26 MedHE / notebook on Kaggle
```

**Key hyperparameters:**
```python
config = {
    'num_clients': 5,
    'fl_rounds': 3,
    'epochs_per_client': 2,
    'learning_rate': 1e-4,
    'batch_size': 8,
    'sparsity': 0.9,           # 90% gradient sparsification
    'ckks_poly_degree': 8192,  # ring dimension
    'dp_epsilon': 1.0,         # privacy budget
    'dp_delta': 1e-5,
}
```

---

## Use with FairHealth

```python
from fairhealth.federated.privacy import (
    clip_weights,
    add_gaussian_noise,
    sparsify,
    dp_fedavg_aggregate,
)

# 97.5% communication reduction
sparse_w, rate = sparsify(weights, sparsity=0.975)

# Differential privacy
noisy_w = add_gaussian_noise(clip_weights(sparse_w), epsilon=1.0)
```

---

## Citation

```bibtex
@misc{yesmin2026medhe,
  author = {Yesmin, Farjana},
  title  = {MedHE: Communication-Efficient Privacy-Preserving Federated
            Learning with Adaptive Gradient Sparsification for Healthcare},
  note   = {Under review, CIBB 2026. Preprint: arXiv:2511.09043},
  year   = {2026},
  url    = {https://arxiv.org/abs/2511.09043}
}
```

---

**Author:** Farjana Yesmin · [farjana-yesmin.github.io](https://farjana-yesmin.github.io)
· MIT License
