# MedHE: Communication-Efficient Privacy-Preserving Federated Learning for Healthcare

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.3](https://img.shields.io/badge/PyTorch-2.3.0-orange.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.09043-b31b1b.svg)](https://arxiv.org/abs/2511.09043)
[![CIBB 2026](https://img.shields.io/badge/CIBB_2026-Oral_Presentation-brightgreen.svg)](https://cibb2026.teralab.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of:

> Yesmin, F. (2026). *MedHE: Communication-Efficient Privacy-Preserving Federated
> Learning for Healthcare Text Classification.*
> **Accepted for Oral Presentation at CIBB 2026** (21st International Conference on
> Computational Intelligence Methods for Bioinformatics and Biostatistics),
> Sapienza University of Rome, Italy, September 2–4, 2026.
> Preprint: [arXiv:2511.09043](https://arxiv.org/abs/2511.09043)

Part of the [FairHealth](https://github.com/Farjana-Yesmin/fairhealth) library —
`pip install fairhealth`

---

## What MedHE Does

MedHE co-designs **adaptive gradient sparsification** with **CKKS homomorphic
encryption** for privacy-preserving healthcare NLP. Applied to binary drug
effectiveness classification using DistilBERT across 5 federated clients, MedHE:

- Reduces communication from **1,277 MB → 32 MB** (97.5% reduction)
- Maintains **macro-F1 = 0.950 ± 0.005** (no statistically significant difference from standard FL, p=0.32)
- Achieves **MIA resistance of 51.1%** (near-random; ideal = 50%)
- Satisfies **differential privacy** with ε ≤ 1.0, δ = 1×10⁻⁵ (Δ₂=1.0, σ=0.12, moments accountant)

---

## Results

### Performance Comparison (5 independent trials, mean ± std)

| Method | Macro-F1 | Accuracy | Communication (MB) |
|---|---|---|---|
| Standard FedAvg | 0.944 ± 0.006 | 89.9 ± 0.7% | 1,277 |
| HE-only FL | 0.714 ± 0.126 | 90.4 ± 2.2% | 1,490 |
| **MedHE (ours)** | **0.950 ± 0.005** | **89.5 ± 0.8%** | **32** |

MedHE vs Standard FedAvg: p=0.32 (no statistically significant difference detected).  
HE-only FL collapses to majority-class bias (F1=0.714) without sparsification — the paper's key finding.

### Privacy Evaluation (Membership Inference Attacks — 4 attack strategies)

| Method | Worst-case MIA Rate | Privacy Level |
|---|---|---|
| Random guessing (ideal) | 50.0% | Perfect |
| **MedHE (ours)** | **51.1%** | **Strong** |
| Standard FedAvg | 56.3% | Weak |

MIA rates measured experimentally on trained models across four attack strategies: confidence threshold, entropy threshold, modified entropy, and ML classifier attack.

### Ablation Study

| Configuration | Macro-F1 | Accuracy | Comm (MB) | MIA |
|---|---|---|---|---|
| Full MedHE | **0.950** | 89.5% | 32 | 51.1% |
| − Error feedback | 0.763 | 91.2% | 32 | 51.1% |
| − Adaptive threshold | 0.773 | 91.2% | 32 | 51.1% |
| − Batch packing | 0.950 | 89.5% | 415 | 51.1% |
| − HE (sparsity only) | 0.755 | 90.9% | 26 | 56.3% |

Error feedback and adaptive thresholding are essential: F1 drops ~19% when either is removed, despite raw accuracy appearing to rise (majority-class bias). Removing HE eliminates privacy without meaningful communication savings.

### Sparsity Sensitivity

Optimal sparsity s = 0.9:
- s < 0.8: communication savings < 80%, insufficient benefit
- **s = 0.9: accuracy preserved, 97.5% communication reduction ✓**
- s > 0.95: accuracy drops below 85%

---

## MedHE Framework

### Three-Component Architecture

```
Client gradient G ∈ ℝᵈ
↓
Step 1: Adaptive Sparsification (top-10% by magnitude)
  + Error-feedback: residual eₜ = G - G_sparse added to next round
  + EMA threshold: τₜ = α·τₜ₋₁ + (1-α)·τcurrent, α=0.9
  + Gradient clip: L2 norm Δ₂ = 1.0
↓
Step 2: CKKS Batch Packing (TenSEAL)
  64 gradient values per ciphertext slot
  N×64 = 524,288 effective capacity per ciphertext
  ⌈6,695,501 / 524,288⌉ = 13 ciphertexts (vs 800+ naïve)
  Each ciphertext ≈ 0.47 MB → 6.1 MB per client per round
  Ring dimension N=8,192, 128-bit security (RLWE)
↓
Step 3: Differential Privacy
  Gaussian noise N(0, σ²I), σ=0.12
  ε ≤ 1.0, δ = 1×10⁻⁵ across T=3 rounds (moments accountant)
  90% sparsity amplifies privacy by factor (1-s)=0.1
  Client-level guarantee (all 5 clients participate each round)
↓
Encrypted upload to aggregation server
Server: ciphertext-domain weighted averaging → broadcast
```

### Convergence

Both MedHE and Standard FedAvg converge by round 3 of 10. The error-feedback mechanism prevents gradient bias accumulation and divergence across rounds.

---

## Dataset

**UCI Drug Review Dataset — Druglib.com** (ID 461)  
DOI: [10.24432/C55G6J](https://doi.org/10.24432/C55G6J)

- Patient-written reviews grouped by benefits, side effects, and overall comment
- Task: binary effectiveness classification (effective if rating ≥ 3 on 5-point scale)
- 86% majority class → **macro-F1 is the primary metric** (accuracy alone is misleading)
- Labels are self-reported patient ratings, not clinically validated outcomes
- Federated setup: 5 clients, equal IID partition (~660 samples each)
- 3 communication rounds, 2 local epochs per round, batch size 8

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

Open the Kaggle notebook in `CIBB'26 MedHE/` and run all cells. All experiments (FL training, MIA evaluation, ablation study) are self-contained.

**Key hyperparameters:**
```python
config = {
    'num_clients': 5,
    'fl_rounds': 3,
    'epochs_per_client': 2,
    'learning_rate': 1e-4,
    'batch_size': 8,
    'sparsity': 0.9,            # 90% gradient sparsification
    'ckks_poly_degree': 8192,   # ring dimension N
    'ckks_coeff_mod_bits': 240, # 4 × 60-bit primes
    'ckks_scale': 2**40,        # scale factor Δ
    'batch_pack_size': 64,      # gradients per slot
    'dp_clip_norm': 1.0,        # L2 sensitivity Δ₂
    'dp_noise_multiplier': 0.12,# σ
    'dp_epsilon': 1.0,          # ε budget
    'dp_delta': 1e-5,           # δ
    'ema_alpha': 0.9,           # EMA threshold rate
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
sparse_w, rate = sparsify(weights, sparsity=0.9)

# Differential privacy (ε ≤ 1.0)
noisy_w = add_gaussian_noise(clip_weights(sparse_w, norm=1.0), epsilon=1.0)
```

---

## Citation

```bibtex
@inproceedings{yesmin2026medhe,
  author    = {Yesmin, Farjana},
  title     = {MedHE: Communication-Efficient Privacy-Preserving Federated
               Learning for Healthcare Text Classification},
  booktitle = {Proceedings of CIBB 2026 -- 21st International Conference on
               Computational Intelligence Methods for Bioinformatics
               and Biostatistics},
  year      = {2026},
  address   = {Sapienza University of Rome, Italy},
  note      = {Oral presentation. Preprint: arXiv:2511.09043},
  url       = {https://arxiv.org/abs/2511.09043}
}
```

---

**Author:** Farjana Yesmin · [farjana-yesmin.github.io](https://farjana-yesmin.github.io) · MIT License
