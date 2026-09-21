# 🔬 MedVQA N6 — Medical Visual Question Answering & Visual Explainability

**Deep Learning Multimodal System for Automated Clinical Question Answering and Visual Attention Mapping on Radiology Images (X-Ray, CT, MRI).**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg)](https://huggingface.co/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed.svg)](https://www.docker.com/)

---

## 📌 Overview

**MedVQA N6** is an end-to-end multimodal medical artificial intelligence framework designed to assist radiologists and clinicians in diagnostic decision-making. Given an input radiological image (X-ray, CT scan, or MRI) along with a free-form clinical question in natural language, the system:
1. **Predicts accurate clinical answers** across both closed (binary/yes-no/abnormality) and open-ended (organ/diagnosis/modality) queries.
2. **Estimates calibrated diagnostic confidence scores**.
3. **Generates visual explainability heatmaps** using **Grad-CAM** attention maps overlaid directly onto the source radiological scans.

---

## 🧠 System Architecture

```
                          ┌────────────────────────┐
                          │    Radiology Image     │
                          │     (X-Ray/CT/MRI)     │
                          └───────────┬────────────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │     Vision Encoder     │
                          │   ViT-Base/16 (224)    │
                          └───────────┬────────────┘
                                      │ (768-dim)
                                      ▼
                          ┌────────────────────────┐
                          │  Visual Projection FC  │
                          └───────────┬────────────┘
                                      │ (512-dim)
                                      │
┌────────────────────────┐            │
│   Clinical Question    │            │
│   (Natural Language)   │            │
└───────────┬────────────┘            │
            │                         │
            ▼                         │
┌────────────────────────┐            │
│      Text Encoder      │            │
│       BiomedBERT       │            │
└───────────┬────────────┘            │
            │ (768-dim)               │
            ▼                         │
┌────────────────────────┐            │
│   Text Projection FC   │            │
└───────────┬────────────┘            │
            │ (512-dim)               │
            └───────────┬─────────────┘
                        ▼
            ┌────────────────────────┐
            │   Multimodal Fusion    │
            │  Concat + BN + Dropout │
            └───────────┬────────────┘
                        │
            ┌───────────┴────────────┐
            ▼                        ▼
┌────────────────────────┐ ┌────────────────────────┐
│  Head Closed (Linear)  │ │   Head Open (Linear)   │
│  Yes/No/Normal/Bilateral│ │  Organs & Diagnoses    │
└────────────────────────┘ └────────────────────────┘
```

- **Visual Backbone**: `Vision Transformer (ViT-Base/16, 224x224 input)`
- **Textual Backbone**: `BiomedBERT` (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`)
- **Multimodal Fusion**: Dedicated 512-d linear projections, feature concatenation, `BatchNorm1d`, and `Dropout(0.3)` regularization.
- **Visual Explainability**: Grad-CAM attention hook on the final normalization layer of the Vision Transformer with vectorized Jet colormap projection.

---

## 🗂️ Repository Structure

```
.
├── src/                               # Modular Python Source Package
│   ├── models/
│   │   └── medvqa.py                  # MedVQA_N6 Dual-Head Architecture
│   ├── interpretability/
│   │   └── gradcam.py                 # Vision Transformer Grad-CAM Module
│   ├── utils/
│   │   ├── colormap.py                # Vectorized Jet Colormap Generator
│   │   └── download_weights.py        # Automated Weights Download Utility
│   └── pipeline.py                    # Unified Inference & Explainability Pipeline
├── model/
│   ├── vocab_closed.json              # Closed-ended Answers Vocabulary
│   └── vocab_open.json                # Open-ended Answers Vocabulary
├── notebooks/                         # Complete Research Benchmarking (N1 -> N6)
│   ├── Baseline1_and_2.ipynb          # Baselines N1 (Majority/BERT) & N2 (CNN+LSTM)
│   ├── MedVQA_Baseline_N3_BAN8.ipynb  # Baseline N3 (Bilinear Attention Network)
│   ├── baseline4.ipynb                # Baseline N4 (ViLBERT / Co-attention)
│   ├── MedVQA_Baseline_N5.ipynb       # Baseline N5 (ViT + BiomedBERT baseline)
│   └── MedVQA_N6_Final.ipynb          # ⭐ Official Final N6 Model (Training & Export)
├── templates/
│   └── index.html                     # Clinical Diagnostic Web UI (HTML5)
├── static/
│   ├── style.css                      # UI Theme / Glassmorphism / Dark Mode
│   └── app.js                         # Interactive Logic, Grad-CAM slider, PDF Export
├── app.py                             # Flask Production Web Server & REST API
├── gradio_app.py                      # Gradio Web Demo (Hugging Face Spaces)
├── Dockerfile                         # Containerization Recipe
├── requirements.txt                   # Python Dependencies
├── LICENSE                            # MIT License
└── README.md
```

---

## 🚀 Quickstart Guide (Local Setup)

### 1. Clone Repository & Install Dependencies

```bash
git clone https://github.com/Yousra-khallou/Visual-Question-Answering-for-Radiology-Images-Medical-AI-Research-Project.git
cd Visual-Question-Answering-for-Radiology-Images-Medical-AI-Research-Project

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\Activate
# Linux / macOS:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Download Model Weights (`best_n6.pth`)

Place your trained `best_n6.pth` checkpoint (from [`notebooks/MedVQA_N6_Final.ipynb`](notebooks/MedVQA_N6_Final.ipynb)) inside the `model/` folder, or download it automatically:

```bash
# Via Google Drive:
python -m src.utils.download_weights --gdrive-id <YOUR_GDRIVE_FILE_ID>

# Via Hugging Face Hub:
python -m src.utils.download_weights --hf-repo <USERNAME/REPO_NAME>
```

### 3. Launch Flask Web Application

```bash
python app.py
```
👉 Open your browser at: **`http://localhost:5000`**

### 4. Launch Gradio Demo Interface

```bash
python gradio_app.py
```
👉 Open your browser at: **`http://localhost:7860`**

---

## 🐳 Docker Containerization & Cloud Deployment

This project includes a production-ready `Dockerfile` powered by **Gunicorn** WSGI:

### 1. Local Containerized Execution
Run the entire application in an isolated environment without local PyTorch configuration:

```bash
# Build Docker image
docker build -t medvqa-n6 .

# Run container locally
docker run -d -p 5000:5000 --name medvqa-app medvqa-n6
```
👉 The application will be accessible at **`http://localhost:5000`**.

### 2. Public Cloud Deployment (Optional)
This repository is pre-configured for one-click deployment:
- **Hugging Face Spaces**: Instant deployment via `gradio_app.py` (Free 16 GB RAM CPU tier).
- **Render / Railway / AWS / GCP**: Deploy the containerized Flask app via `Dockerfile`.

---

## 📊 Datasets & Research Benchmarks

The models are trained and evaluated on gold-standard clinical radiology benchmarks:
- **VQA-RAD**: 3,515 clinician-validated image-question pairs across various radiology modalities.
- **SLAKE**: Semantically annotated bilingual medical Visual Question Answering benchmark.

---

## 👥 Authors & Collaboration

This research project was developed in collaborative partnership by:
- **Yousra Khallou** — [@Yousra-khallou](https://github.com/Yousra-khallou)
- **Ilham Elmattichi** — [@ilhameelma](https://github.com/ilhameelma)

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.  
Copyright (c) 2026 Yousra Khallou & Ilham Elmattichi.
