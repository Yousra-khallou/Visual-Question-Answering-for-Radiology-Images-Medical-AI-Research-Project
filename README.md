# 🔬 MedVQA N6 — Medical Visual Question Answering & Explainability

**Système d'Intelligence Artificielle pour le Diagnostic et l'Analyse Visuelle de Radiologies Médicales.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg)](https://huggingface.co/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed.svg)](https://www.docker.com/)

---

## 📌 Présentation

**MedVQA N6** est un système multimodal d'aide au diagnostic médical combinant vision par ordinateur et traitement du langage naturel clinique. À partir d'un cliché d'imagerie médicale (Radiographie X, Scanner CT, IRM) et d'une question clinique formulée en langage naturel, le modèle :
1. **Prédit la réponse diagnostique** (questions binaires/fermées ou questions cliniques ouvertes).
2. **Estime la certitude diagnostique** via un score de confiance calibré.
3. **Fournit une explication visuelle interprétable** grâce à des cartes de chaleur **Grad-CAM** superposées à l'image d'origine.

---

## 🧠 Architecture Multimodale

```
                          ┌────────────────────────┐
                          │   Radiologie Médicale  │
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
│    Question Clinique   │            │
│   (Langage Naturel)    │            │
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
            │   Fusion Multimodale   │
            │  Concat + BN + Dropout │
            └───────────┬────────────┘
                        │
            ┌───────────┴────────────┐
            ▼                        ▼
┌────────────────────────┐ ┌────────────────────────┐
│  Head Closed (Linear)  │ │   Head Open (Linear)   │
│  Oui/Non/Normal/Bilateral│ │  Diagnostics & Organes │
└────────────────────────┘ └────────────────────────┘
```

- **Backbone Visuel** : `Vision Transformer (ViT-Base/16, patch 224)`
- **Backbone Textuel** : `BiomedBERT` (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`)
- **Mécanisme de Fusion** : Projections denses 512-d, concaténation, BatchNorm et régularisation Dropout.
- **Interprétabilité** : Hook Grad-CAM sur la couche de normalisation du ViT avec projection sur l'image source.

---

## 🗂️ Structure du Projet

```
.
├── src/                               # Modules Python réutilisables
│   ├── models/
│   │   └── medvqa.py                  # Architecture du modèle MedVQA_N6
│   ├── interpretability/
│   │   └── gradcam.py                 # Algorithme Grad-CAM pour ViT
│   ├── utils/
│   │   ├── colormap.py                # Colormap Jet optimisée
│   │   └── download_weights.py        # Téléchargement automatique des poids
│   └── pipeline.py                    # Pipeline unifié inférence & explicabilité
├── model/
│   ├── vocab_closed.json              # Dictionnaire des réponses fermées
│   └── vocab_open.json                # Dictionnaire des réponses ouvertes
├── notebooks/                         # Démarche expérimentale complète (N1 -> N6)
│   ├── Baseline1_and_2.ipynb          # Baselines N1 (Majority/BERT) & N2 (CNN+LSTM)
│   ├── MedVQA_Baseline_N3_BAN8.ipynb  # Baseline N3 (Bilinear Attention Network)
│   ├── baseline4.ipynb                # Baseline N4 (ViLBERT / Co-attention)
│   ├── MedVQA_Baseline_N5.ipynb       # Baseline N5 (ViT + BiomedBERT minimal)
│   └── MedVQA_N6_Final.ipynb          # ⭐ Modèle FINAL N6 (Entraînement & Export)
├── templates/
│   └── index.html                     # Interface Web Médicale (HTML5)
├── static/
│   ├── style.css                      # Thème UI / Dark Mode / Glassmorphism
│   └── app.js                         # Logique interactive, export PDF, historique
├── app.py                             # Serveur Web Flask & API REST
├── gradio_app.py                      # Démo Gradio (Prêt pour Hugging Face Spaces)
├── Dockerfile                         # Recette de conteneurisation Docker
├── requirements.txt                   # Dépendances Python
└── README.md
```

---

## 🚀 Démarrage Rapide (Local)

### 1. Cloner le Répertoire & Installer les Dépendances

```bash
git clone https://github.com/Yousra-khallou/Visual-Question-Answering-for-Radiology-Images-Medical-AI-Research-Project.git
cd Visual-Question-Answering-for-Radiology-Images-Medical-AI-Research-Project

python -m venv venv
# Windows :
venv\Scripts\activate
# Linux/macOS :
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Télécharger ou Placer les Poids du Modèle (`best_n6.pth`)

Placez le fichier `best_n6.pth` (issu de l'entraînement dans [`notebooks/MedVQA_N6_Final.ipynb`](notebooks/MedVQA_N6_Final.ipynb)) dans le dossier `model/`, ou téléchargez-le automatiquement :

```bash
# Via Google Drive :
python -m src.utils.download_weights --gdrive-id <VOTRE_GDRIVE_FILE_ID>

# Via Hugging Face Hub :
python -m src.utils.download_weights --hf-repo <USERNAME/REPO_NAME>
```

### 3. Lancer l'Application Web Flask

```bash
python app.py
```
👉 Accédez à l'interface dans votre navigateur : **`http://localhost:5000`**

### 4. Lancer l'Interface Gradio

```bash
python gradio_app.py
```
👉 Accédez à l'interface Gradio : **`http://localhost:7860`**

---

## 🐳 Conteneurisation Docker & Prêt pour le Cloud

Le projet inclut un fichier `Dockerfile` configuré avec le serveur de production WSGI **Gunicorn**. Il permet deux usages :

### 1. Exécution Locale Isolée (dans un conteneur sur votre machine)
Permet d'exécuter l'application dans un environnement hermétique sans installer Python ou PyTorch directement sur votre système :

```bash
# Construction de l'image Docker
docker build -t medvqa-n6 .

# Lancement du conteneur en local
docker run -d -p 5000:5000 --name medvqa-app medvqa-n6
```
👉 L'application tourne alors sur **`http://localhost:5000`**.

### 2. Déploiement Public sur le Cloud (Optionnel)
Ce même conteneur Docker ou le fichier `gradio_app.py` peut être déployé en 1 clic sur les plateformes Cloud pour donner une URL publique accessible à tous :
- **Hugging Face Spaces** : Via `gradio_app.py` (Gratuit, 16 Go RAM).
- **Render / Railway / AWS / GCP** : En connectant simplement ce dépôt GitHub et son `Dockerfile`.

---

## 📊 Datasets & Entraînement

Le modèle est entraîné et évalué sur les benchmarks cliniques de référence :
- **VQA-RAD** : 3 515 paires image-question validées par des cliniciens.
- **SLAKE** : Dataset bilingue annoté sémantiquement pour la radiologie.

---


---

## 👥 Auteurs & Collaboration

Ce projet de recherche a été développé en collaboration par :
- **Yousra Khallou** — [@Yousra-khallou](https://github.com/Yousra-khallou)
- **Ilham Elmattichi** — [@ilhameelma](https://github.com/ilhameelma)

---

## 📜 Licence & Droits d'Auteur

Ce projet est distribué sous licence MIT. Consultez le fichier [LICENSE](LICENSE) pour plus de détails.  
Copyright (c) 2026 Yousra Khallou & Ilham Elmattichi.
