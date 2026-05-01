# RadVQA

**Visual Question Answering for Radiology Images — Medical AI Research Project**

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![Status](https://img.shields.io/badge/status-work%20in%20progress-orange)
![Domain](https://img.shields.io/badge/domain-Radiology%20·%20NLP%20·%20CV-purple)

---

## Overview

**RadVQA** is a research project exploring **Medical Visual Question Answering (Med-VQA)** applied to radiology images. Given a radiology image (X-ray, CT, MRI) and a clinical question in natural language, the model produces an accurate and interpretable answer.

This repository provides a modular pipeline covering data preprocessing, visual and text encoding, multimodal fusion, and answer generation — evaluated on standard benchmarks (VQA-RAD, SLAKE).

---

## Features

- **Multimodal architecture** — visual encoder (ViT / ResNet) + text encoder (BioBERT)
- **Closed & open-ended answering** — classification and free-text generation modes
- **Benchmark-ready** — evaluation scripts for VQA-RAD and SLAKE out of the box
- **Modular design** — swap encoders, fusion strategies, and datasets with minimal config changes
- **Experiment tracking** — integrated with Weights & Biases

---

## Project Structure

```
radvqa/
├── data/
│   ├── vqa_rad/          # VQA-RAD dataset (download separately)
│   ├── slake/            # SLAKE dataset (download separately)
│   └── preprocessing/    # tokenizers, image transforms
├── models/
│   ├── visual_encoder.py # ViT / ResNet visual backbone
│   ├── text_encoder.py   # BioBERT / ClinicalBERT
│   └── fusion.py         # Cross-attention multimodal fusion
├── train.py              # Training loop
├── evaluate.py           # Evaluation on benchmarks
├── config/
│   └── default.yaml      # Hyperparameters & dataset paths
├── notebooks/
│   └── exploration.ipynb # EDA & visualization
├── requirements.txt
└── README.md
```

---

## Getting Started

### Installation

```bash
git clone https://github.com/your-username/radvqa.git
cd radvqa
pip install -r requirements.txt
```

### Download Datasets

- **VQA-RAD** — [osf.io/89kps](https://osf.io/89kps) → place in `data/vqa_rad/`
- **SLAKE** — [github.com/med-vl/SLAKE](https://github.com/med-vl/SLAKE) → place in `data/slake/`

### Train

```bash
python train.py --config config/default.yaml --dataset vqa_rad
```

### Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best_model.pt --dataset slake
```

---

## Datasets Used

| Dataset | Type | Size | Access |
|---|---|---|---|
| VQA-RAD | VQA (clinical Q&A) | 3 515 pairs | [osf.io/89kps](https://osf.io/89kps) |
| SLAKE | VQA bilingual EN/ZH | 14 000 pairs | [GitHub](https://github.com/med-vl/SLAKE) |
| NIH ChestX-ray14 | Classification (pre-training) | 112 000+ images | [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) |
| MIMIC-CXR | Report generation | 227 835 studies | [PhysioNet](https://physionet.org/content/mimic-cxr) |

---

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐
│  Radiology Image│     │  Clinical Question│
│  (X-ray/CT/MRI) │     │  (natural language│
└────────┬────────┘     └────────┬─────────┘
         │                       │
         ▼                       ▼
  ┌─────────────┐        ┌─────────────┐
  │Visual Encoder│       │ Text Encoder │
  │ViT / ResNet │        │  BioBERT    │
  └──────┬──────┘        └──────┬──────┘
         │                      │
         └──────────┬───────────┘
                    ▼
          ┌──────────────────┐
          │  Fusion Module   │
          │ Cross-Attention  │
          └────────┬─────────┘
                   ▼
         ┌─────────────────┐
         │  Answer Module  │
         │  Classif / Gen  │
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │    Answer       │
         │  Yes/No/Label   │
         │  or free text   │
         └─────────────────┘
```

---

## Roadmap

- [ ] Baseline model (BioBERT + ResNet + concat fusion)
- [ ] Cross-attention fusion module
- [ ] Evaluation on VQA-RAD and SLAKE
- [ ] Grad-CAM visual explanations
- [ ] Fine-tuning LLaVA-Med adapter
- [ ] Demo Gradio interface

---

## References

- Lau et al. (2018) — [VQA-RAD](https://osf.io/89kps)
- Liu et al. (2021) — [SLAKE](https://arxiv.org/abs/2107.04803)
- Johnson et al. (2019) — [MIMIC-CXR](https://physionet.org/content/mimic-cxr)
- Wang et al. (2017) — [ChestX-ray14](https://arxiv.org/abs/1705.02315)

---

## Citation

```bibtex
@misc{radvqa2025,
  title  = {RadVQA: Visual Question Answering for Radiology Images},
  author = {Your Name},
  year   = {2025},
  url    = {https://github.com/your-username/radvqa}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
