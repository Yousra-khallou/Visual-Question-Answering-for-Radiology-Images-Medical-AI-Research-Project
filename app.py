"""
MedVQA N6 — Serveur Flask
Lance avec : python app.py
Accès : http://localhost:5000
"""

import os, io, json, gc, warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import torchvision.transforms as transforms
import timm
from transformers import AutoModel, AutoTokenizer
import numpy as np
import base64

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =============================================================================
# CONFIGURATION
# =============================================================================
BIOMEDBERT  = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH  = 'model/best_n6.pth'
VOCAB_C     = 'model/vocab_closed.json'
VOCAB_O     = 'model/vocab_open.json'

MEAN = [0.5, 0.5, 0.5]
STD  = [0.5, 0.5, 0.5]

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

CLOSED_KEYWORDS = {
    'yes', 'no', 'normal', 'abnormal', 'present', 'absent',
    'left', 'right', 'bilateral', 'central', 'peripheral',
    'true', 'false', 'positive', 'negative',
}

# =============================================================================
# ARCHITECTURE MODÈLE (identique à MedVQA_N6_Final.py)
# =============================================================================
class MedVQA_N6(nn.Module):
    def __init__(self, num_closed, num_open, dim=768, proj_dim=512):
        super().__init__()
        self.visual_encoder = timm.create_model(
            'vit_base_patch16_224', pretrained=False, num_classes=0)
        self.text_encoder = AutoModel.from_pretrained(BIOMEDBERT)
        self.v_proj = nn.Sequential(nn.Linear(dim, proj_dim), nn.GELU(), nn.Dropout(0.1))
        self.q_proj = nn.Sequential(nn.Linear(dim, proj_dim), nn.GELU(), nn.Dropout(0.1))
        self.fusion = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(proj_dim, proj_dim // 2), nn.GELU(), nn.Dropout(0.3),
        )
        self.bn = nn.BatchNorm1d(proj_dim // 2)
        self.head_closed = nn.Linear(proj_dim // 2, num_closed)
        self.head_open   = nn.Sequential(
            nn.Linear(proj_dim // 2, proj_dim // 4), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(proj_dim // 4, num_open),
        )

    def forward(self, img, input_ids, attn_mask):
        v = self.visual_encoder.forward_features(img).mean(dim=1)
        q = self.text_encoder(
            input_ids=input_ids, attention_mask=attn_mask
        ).last_hidden_state[:, 0, :]
        fused = self.fusion(torch.cat([self.v_proj(v), self.q_proj(q)], dim=-1))
        fused = self.bn(fused)
        return self.head_closed(fused), self.head_open(fused)


# =============================================================================
# GRAD-CAM
# =============================================================================
class GradCAMViT:
    def __init__(self, model):
        self.model = model
        self.grads, self.acts = [], []
        self.hook_f = model.visual_encoder.norm.register_forward_hook(
            lambda m, i, o: self.acts.append(o))
        self.hook_b = model.visual_encoder.norm.register_full_backward_hook(
            lambda m, gi, go: self.grads.append(go[0]))

    def remove_hooks(self):
        self.hook_f.remove(); self.hook_b.remove()

    def generate(self, img_t, input_ids, attn_mask, answer_type=0):
        self.model.eval()
        self.grads, self.acts = [], []
        lc, lo = self.model(img_t, input_ids, attn_mask)
        score  = lc.max() if answer_type == 0 else lo.max()
        self.model.zero_grad()
        score.backward(retain_graph=True)
        weights = self.grads[0].mean(dim=1, keepdim=True)
        cam = F.relu((weights * self.acts[0]).sum(dim=-1)[:, 1:])
        B   = cam.size(0)
        cam = cam.view(B, 14, 14)
        cam = cam - cam.view(B, -1).min(dim=1)[0].view(B, 1, 1)
        cam = cam / (cam.view(B, -1).max(dim=1)[0].view(B, 1, 1) + 1e-8)
        return cam.detach()


# =============================================================================
# CHARGEMENT DU MODÈLE
# =============================================================================
model_instance = None
tokenizer_inst  = None
closed2idx, idx2closed = {}, {}
open2idx,   idx2open   = {}, {}

def load_model():
    global model_instance, tokenizer_inst, closed2idx, idx2closed, open2idx, idx2open

    if model_instance is not None:
        return True, "Modèle déjà chargé"

    # Vérifier les fichiers
    missing = []
    for f in [MODEL_PATH, VOCAB_C, VOCAB_O]:
        if not os.path.exists(f):
            missing.append(f)
    if missing:
        return False, f"Fichiers manquants : {missing}"

    try:
        print("Chargement des vocabulaires...")
        with open(VOCAB_C) as f:
            vc = json.load(f)
            closed2idx = vc['closed2idx']
            idx2closed = {int(k): v for k, v in vc['idx2closed'].items()}

        with open(VOCAB_O) as f:
            vo = json.load(f)
            open2idx = vo['open2idx']
            idx2open = {int(k): v for k, v in vo['idx2open'].items()}

        print(f"Vocabulaires — Closed: {len(closed2idx)} | Open: {len(open2idx)}")

        print("Chargement du tokenizer BiomedBERT...")
        tokenizer_inst = AutoTokenizer.from_pretrained(BIOMEDBERT)

        print("Chargement du modèle N6...")
        model_instance = MedVQA_N6(
            num_closed=len(closed2idx),
            num_open=len(open2idx)
        ).to(DEVICE)

        ck = torch.load(MODEL_PATH, map_location=DEVICE)
        model_instance.load_state_dict(ck['model_state_dict'])
        model_instance.eval()

        print(f"✅ Modèle chargé sur {DEVICE}")
        return True, f"Modèle chargé (epoch {ck.get('epoch','?')}, Val Combined {ck.get('val_combined',0):.1f}%)"

    except Exception as e:
        import traceback
        return False, f"Erreur chargement : {e}\n{traceback.format_exc()}"


def predict(image: Image.Image, question: str, generate_cam: bool = True):
    """Inférence + Grad-CAM optionnel."""
    if model_instance is None:
        return {"error": "Modèle non chargé"}

    # Preprocessing image
    img_t = val_transform(image.convert('RGB')).unsqueeze(0).to(DEVICE)

    # Tokenisation
    tok = tokenizer_inst(
        [question], padding=True, truncation=True,
        max_length=64, return_tensors='pt'
    )
    input_ids = tok['input_ids'].to(DEVICE)
    attn_mask = tok['attention_mask'].to(DEVICE)

    with torch.no_grad():
        logits_c, logits_o = model_instance(img_t, input_ids, attn_mask)

    # Top-3 Closed
    probs_c = torch.softmax(logits_c, dim=-1)[0]
    top3_c  = probs_c.topk(3)
    top3_closed = [
        {"answer": idx2closed.get(i.item(), "?"), "confidence": round(p.item() * 100, 1)}
        for p, i in zip(top3_c.values, top3_c.indices)
    ]

    # Top-3 Open
    probs_o = torch.softmax(logits_o, dim=-1)[0]
    top3_o  = probs_o.topk(3)
    top3_open = [
        {"answer": idx2open.get(i.item(), "?"), "confidence": round(p.item() * 100, 1)}
        for p, i in zip(top3_o.values, top3_o.indices)
    ]

    best_closed_conf = top3_closed[0]['confidence']
    best_open_conf   = top3_open[0]['confidence']

    # Décision : closed si confiance > 60% ET réponse dans keywords
    ans_closed  = top3_closed[0]['answer']
    is_closed = (best_closed_conf > 60.0) and (ans_closed in CLOSED_KEYWORDS)
    answer_type = 0 if is_closed else 1
    best_answer = top3_closed[0]['answer'] if is_closed else top3_open[0]['answer']
    confidence  = best_closed_conf if is_closed else best_open_conf

    result = {
        "answer":       best_answer,
        "answer_type":  "Closed (oui/non/binaire)" if is_closed else "Open (médical)",
        "confidence":   confidence,
        "top3_closed":  top3_closed,
        "top3_open":    top3_open,
        "cam_base64":   None,
    }

    # Grad-CAM
    if generate_cam:
        try:
            gcam = GradCAMViT(model_instance)
            cam  = gcam.generate(img_t, input_ids, attn_mask, answer_type)
            gcam.remove_hooks()

            cam_np = cam[0].cpu().numpy()
            cam_up = np.array(
                Image.fromarray((cam_np * 255).astype(np.uint8))
                     .resize((224, 224), Image.BILINEAR)
            ) / 255.0

            # Overlay sur image originale
            img_small = image.convert('RGB').resize((224, 224))
            img_np = np.array(img_small) / 255.0

            # Colormap jet manuel
            cmap = plt_jet(cam_up)
            overlay = np.clip(img_np * 0.6 + cmap * 0.4, 0, 1)
            overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))

            buf = io.BytesIO()
            overlay_img.save(buf, format='JPEG', quality=90)
            result['cam_base64'] = base64.b64encode(buf.getvalue()).decode()

        except Exception as e:
            result['cam_error'] = str(e)

    return result


def plt_jet(x):
    """Colormap jet sans matplotlib (pour éviter l'import)."""
    x = np.clip(x, 0, 1)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


# =============================================================================
# ROUTES FLASK
# =============================================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/status')
def status():
    ok, msg = load_model()
    return jsonify({
        "loaded": model_instance is not None,
        "device": str(DEVICE),
        "message": msg,
        "vocab": {
            "closed": len(closed2idx),
            "open":   len(open2idx),
        } if model_instance else {}
    })


@app.route('/predict', methods=['POST'])
def predict_route():
    if model_instance is None:
        ok, msg = load_model()
        if not ok:
            return jsonify({"error": msg}), 500

    if 'image' not in request.files:
        return jsonify({"error": "Aucune image fournie"}), 400
    question = request.form.get('question', '').strip()
    if not question:
        return jsonify({"error": "Aucune question fournie"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Fichier vide"}), 400

    try:
        image = Image.open(file.stream).convert('RGB')
        import time
        t0 = time.time()
        result = predict(image, question, generate_cam=True)
        result['processing_time_ms'] = round((time.time() - t0) * 1000)
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route('/load_model', methods=['POST'])
def load_model_route():
    ok, msg = load_model()
    return jsonify({"success": ok, "message": msg})


if __name__ == '__main__':
    print(f"\n{'='*60}")
    print("  MedVQA N6 — Serveur Web")
    print(f"  Device : {DEVICE}")
    print(f"  URL    : http://localhost:5000")
    print(f"{'='*60}\n")
    # Pré-charger le modèle au démarrage
    ok, msg = load_model()
    print(f"Modèle : {msg}")
    app.run(debug=False, host='0.0.0.0', port=5000)
