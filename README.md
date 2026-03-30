# Adaptive Prompt Learning (APL) for Privacy-Preserving Scene Understanding

> **Paper:** *Adaptive Prompt Learning for Privacy-Preserving via Vision-Language Models*  
> **Authors:** Abdolazim Rezaei, Mahboobeh Haghparast, Ahmad Patoghy, Mehdi Sookhak  
> **Venue:** IEEE Transactions on Big Data

---

## Overview

APL is a privacy-preserving scene understanding framework for Intelligent Transportation Systems (ITS). Instead of transmitting raw visual data (which exposes Personally Identifiable Information such as faces and license plates), APL converts traffic scene images into **structured, multi-perspective textual abstractions** that can support downstream tasks without leaking sensitive visual content.

```
Raw Image → Vision-Language Encoding → Adaptive Prompt Learning
         → Evidence Extraction → Scenario Bank Matching
         → Iterative Refinement → Rule-Based Decision
```

---

## Architecture

The framework consists of **five decoupled modules**:

| Module | Description |
|--------|-------------|
| `VisionLanguageEncoder` | Extracts global + local visual features via pretrained VLM (CLIP) |
| `AdaptivePromptLearning` | Generates multi-perspective text features via cross-attention Feature Generator |
| `EvidenceExtractor` | Fuses text + visual features into structured scene descriptions |
| `ScenarioBankMatcher` | Matches descriptions to DMV-aligned violation scenarios via differentiable soft-logic |
| `RuleEngine` | Makes legally transparent violation decisions from accumulated evidence |

### Three Feature Branches in APL

1. **Global-Invariant (GI):** Stable, rule-aligned semantic features — regularized toward zero-shot representations
2. **Instance-Dependent (ID):** Scene-adaptive features conditioned on the global visual embedding
3. **Fine-Grained (FG):** Localized evidentiary features selected via top-k spatial similarity

---

## Installation

```bash
git clone https://github.com/<your-username>/APL_framework.git
cd APL_framework
pip install -r requirements.txt
```

### Requirements
- Python ≥ 3.8
- PyTorch ≥ 1.12
- `transformers`, `open_clip_torch`, `numpy`, `pillow`, `tqdm`, `pyyaml`, `pytest`

---

## Project Structure

```
APL_framework/
├── models/
│   ├── __init__.py
│   ├── vision_language_encoder.py   # CLIP-based visual + text encoder
│   ├── feature_generator.py         # Cross-attention SFG module
│   ├── adaptive_prompt_learning.py  # APL: GI + ID + FG branches
│   ├── evidence_extractor.py        # Structured evidence extraction
│   ├── scenario_bank.py             # DMV scenario bank + soft-logic matcher
│   └── rule_engine.py               # Deterministic violation decision
├── utils/
│   ├── __init__.py
│   ├── privacy_metrics.py           # SSIM, PSNR, MSE, SRRA computation
│   └── text_quality.py              # BLEU, ROUGE, CIDEr computation
├── configs/
│   └── default.yaml                 # All hyperparameters
├── tests/
│   ├── __init__.py
│   ├── test_models.py               # Unit tests for all modules
│   └── test_pipeline.py             # End-to-end pipeline test
├── scripts/
│   └── run_inference.py             # Inference entry point
├── requirements.txt
└── README.md
```

---

## Quick Start

### Inference on a single image

```python
from scripts.run_inference import run_inference
from PIL import Image

image = Image.open("path/to/traffic_scene.jpg")
result = run_inference(image)
print(result)
# {'decision': 'violation', 'confidence': 0.91, 'description': '...', 'evidence': {...}}
```

### Running tests (CPU-only, no GPU required)

```bash
pytest tests/ -v
```

---

## Configuration

Edit `configs/default.yaml` to adjust hyperparameters:

```yaml
model:
  feature_dim: 512
  num_fine_grained: 4
  eta: 0.5               # multi-scale factor for spatial grounding
  temperature: 0.07

training:
  lambda_gi: 1.0
  lambda_id: 1.0
  lambda_fg: 1.0
  max_iterations: 3
  convergence_eps: 0.01

rule_engine:
  gamma_strict: 0.75
  gamma_loose: 0.40
```

---

## Privacy Preservation

APL achieves privacy by design — raw pixels are **never transmitted or stored**. The textual abstraction pipeline:
- Irreversibly destroys pixel-level spatial relationships (high MSE, low SSIM/PSNR)
- Resists adversarial image reconstruction attacks (low SRRA)
- Mitigates Membership Inference Attacks (low MIA risk)

---

## Citation

```bibtex
@article{rezaei2025apl,
  title={Adaptive Prompt Learning for Privacy-Preserving via Vision-Language Models},
  author={Rezaei, Abdolazim and Haghparast, Mahboobeh and Patoghy, Ahmad and Sookhak, Mehdi},
  journal={IEEE Transactions on Big Data},
  year={2025}
}
```

---

## License

This project is released under the MIT License.
