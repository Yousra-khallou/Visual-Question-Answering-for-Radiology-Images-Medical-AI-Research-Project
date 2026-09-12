"""
MedVQA N6 — Interface Gradio
Alternative à Flask, idéale pour Hugging Face Spaces (gratuit).

Lancer avec : python gradio_app.py
Accès : http://localhost:7860
"""

import os, io, json, warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import timm
from transformers import AutoModel, AutoTokenizer
import gradio as gr
import base64

warnings.filterwarnings('ignore')

BIOMEDBERT = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'model/best_n6.pth'
MEAN = [0.5, 0.5, 0.5]
STD  = [0.5, 0.5, 0.5]

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

CLOSED_KEYWORDS = {
    'yes','no','normal','abnormal','present','absent',
    'left','right','bilateral','central','peripheral',
    'true','false','positive','negative',
}


class MedVQA_N6(nn.Module):
    def __init__(self, num_closed, num_open, dim=768, proj_dim=512):
        super().__init__()
        self.visual_encoder = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        self.text_encoder   = AutoModel.from_pretrained(BIOMEDBERT)
        self.v_proj = nn.Sequential(nn.Linear(dim, proj_dim), nn.GELU(), nn.Dropout(0.1))
        self.q_proj = nn.Sequential(nn.Linear(dim, proj_dim), nn.GELU(), nn.Dropout(0.1))
        self.fusion = nn.Sequential(
            nn.Linear(proj_dim*2, proj_dim), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(proj_dim, proj_dim//2), nn.GELU(), nn.Dropout(0.3),
        )
        self.bn         = nn.BatchNorm1d(proj_dim//2)
        self.head_closed = nn.Linear(proj_dim//2, num_closed)
        self.head_open   = nn.Sequential(
            nn.Linear(proj_dim//2, proj_dim//4), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(proj_dim//4, num_open),
        )

    def forward(self, img, input_ids, attn_mask):
        v = self.visual_encoder.forward_features(img).mean(dim=1)
        q = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state[:, 0, :]
        fused = self.fusion(torch.cat([self.v_proj(v), self.q_proj(q)], dim=-1))
        return self.head_closed(self.bn(fused)), self.head_open(self.bn(fused))


# Chargement modèle
print("Chargement du modèle...")
with open('model/vocab_closed.json') as f:
    vc = json.load(f)
    closed2idx = vc['closed2idx']
    idx2closed = {int(k): v for k, v in vc['idx2closed'].items()}

with open('model/vocab_open.json') as f:
    vo = json.load(f)
    open2idx = vo['open2idx']
    idx2open = {int(k): v for k, v in vo['idx2open'].items()}

tokenizer = AutoTokenizer.from_pretrained(BIOMEDBERT)
model = MedVQA_N6(num_closed=len(closed2idx), num_open=len(open2idx)).to(DEVICE)
ck = torch.load('model/best_n6.pth', map_location=DEVICE)
model.load_state_dict(ck['model_state_dict'])
model.eval()
print(f"✅ Modèle chargé sur {DEVICE}")


def plt_jet(x):
    x = np.clip(x, 0, 1)
    r = np.clip(1.5 - np.abs(4*x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4*x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4*x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def analyze(image: Image.Image, question: str):
    if image is None:
        return "❌ Aucune image fournie", None, "", ""

    img_t = val_transform(image.convert('RGB')).unsqueeze(0).to(DEVICE)
    tok   = tokenizer([question], padding=True, truncation=True, max_length=64, return_tensors='pt')
    iids  = tok['input_ids'].to(DEVICE)
    amask = tok['attention_mask'].to(DEVICE)

    with torch.no_grad():
        lc, lo = model(img_t, iids, amask)

    probs_c = torch.softmax(lc, dim=-1)[0]
    probs_o = torch.softmax(lo, dim=-1)[0]
    top3_c  = probs_c.topk(3)
    top3_o  = probs_o.topk(3)

    best_c = idx2closed.get(top3_c.indices[0].item(), '?')
    best_o = idx2open.get(top3_o.indices[0].item(), '?')
    conf_c = top3_c.values[0].item() * 100
    conf_o = top3_o.values[0].item() * 100

    is_closed   = conf_c > conf_o and best_c in CLOSED_KEYWORDS
    answer_type = 0 if is_closed else 1
    best_ans    = best_c if is_closed else best_o
    confidence  = conf_c if is_closed else conf_o
    atype_str   = "Closed (binaire/numérique)" if is_closed else "Open (médical)"

    # Grad-CAM
    cam_img = None
    try:
        grads, acts = [], []
        hf = model.visual_encoder.norm.register_forward_hook(lambda m,i,o: acts.append(o))
        hb = model.visual_encoder.norm.register_full_backward_hook(lambda m,gi,go: grads.append(go[0]))
        lc2, lo2 = model(img_t, iids, amask)
        score = lc2.max() if answer_type == 0 else lo2.max()
        model.zero_grad(); score.backward(retain_graph=True)
        hf.remove(); hb.remove()
        weights = grads[0].mean(dim=1, keepdim=True)
        cam = F.relu((weights * acts[0]).sum(dim=-1)[:, 1:]).view(1, 14, 14)
        cam = cam - cam.view(1,-1).min(dim=1)[0].view(1,1,1)
        cam = cam / (cam.view(1,-1).max(dim=1)[0].view(1,1,1) + 1e-8)
        cam_np = cam[0].detach().cpu().numpy()
        cam_up = np.array(Image.fromarray((cam_np*255).astype(np.uint8)).resize((224,224),Image.BILINEAR)) / 255.0
        img_np = np.array(image.convert('RGB').resize((224,224))) / 255.0
        cmap   = plt_jet(cam_up)
        overlay = np.clip(img_np * 0.6 + cmap * 0.4, 0, 1)
        cam_img = Image.fromarray((overlay * 255).astype(np.uint8))
    except Exception as e:
        print(f"Grad-CAM erreur : {e}")

    # Top-3 text
    top3_txt = "**Top-3 Closed :**\n"
    for i, (p, idx) in enumerate(zip(top3_c.values, top3_c.indices)):
        top3_txt += f"{i+1}. {idx2closed.get(idx.item(),'?')} — {p.item()*100:.1f}%\n"
    top3_txt += "\n**Top-3 Open :**\n"
    for i, (p, idx) in enumerate(zip(top3_o.values, top3_o.indices)):
        top3_txt += f"{i+1}. {idx2open.get(idx.item(),'?')} — {p.item()*100:.1f}%\n"

    result_md = f"## {best_ans.upper()}\n\n**Type :** {atype_str}\n\n**Confiance :** {confidence:.1f}%"
    return result_md, cam_img, top3_txt, f"Device: {DEVICE}"


# Interface Gradio
with gr.Blocks(title="MedVQA N6", theme=gr.themes.Base()) as demo:
    gr.Markdown("# 🔬 MedVQA N6 — Visual Question Answering Médical")
    gr.Markdown("*ViT-B/16 + BiomedBERT + Fusion concat — Entraîné sur VQA-RAD + SLAKE*")

    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(type="pil", label="Image médicale")
            question_in = gr.Textbox(
                label="Question",
                placeholder="Is there a pleural effusion? / What organ is shown? / Is this normal?",
                lines=2
            )
            gr.Examples(
                examples=[
                    ["Is there a pleural effusion?"],
                    ["Is this normal?"],
                    ["What organ is visible?"],
                    ["Is there cardiomegaly?"],
                    ["Is the lesion bilateral?"],
                    ["What is the diagnosis?"],
                ],
                inputs=question_in,
            )
            submit_btn = gr.Button("⚡ Analyser", variant="primary")

        with gr.Column(scale=1):
            result_out = gr.Markdown(label="Réponse")
            cam_out    = gr.Image(type="pil", label="Grad-CAM (zones d'attention)")
            top3_out   = gr.Markdown(label="Top-3 candidats")
            info_out   = gr.Textbox(label="Info", interactive=False)

    submit_btn.click(
        fn=analyze,
        inputs=[image_in, question_in],
        outputs=[result_out, cam_out, top3_out, info_out]
    )

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860, share=False)
