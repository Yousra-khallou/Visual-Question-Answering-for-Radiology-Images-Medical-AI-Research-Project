import os
import io
import json
import base64
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import AutoTokenizer

from src.models.medvqa import MedVQA_N6, DEFAULT_BIOMEDBERT
from src.interpretability.gradcam import GradCAMViT
from src.utils.colormap import plt_jet

CLOSED_KEYWORDS = {
    'yes', 'no', 'normal', 'abnormal', 'present', 'absent',
    'left', 'right', 'bilateral', 'central', 'peripheral',
    'true', 'false', 'positive', 'negative',
}

class MedVQAPipeline:
    """
    Unified Inference and Explainability Pipeline for MedVQA N6.
    Shared across Flask Web API and Gradio UI.
    """
    def __init__(
        self,
        model_path: str = 'model/best_n6.pth',
        vocab_closed_path: str = 'model/vocab_closed.json',
        vocab_open_path: str = 'model/vocab_open.json',
        text_backbone: str = DEFAULT_BIOMEDBERT,
        device: torch.device = None
    ):
        self.model_path = model_path
        self.vocab_closed_path = vocab_closed_path
        self.vocab_open_path = vocab_open_path
        self.text_backbone = text_backbone
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.model = None
        self.tokenizer = None
        self.closed2idx, self.idx2closed = {}, {}
        self.open2idx, self.idx2open = {}, {}
        self.is_loaded = False
        self.load_message = "Model not loaded"

    def load(self, force_reload: bool = False):
        if self.is_loaded and not force_reload:
            return True, self.load_message

        missing_files = []
        for path in [self.vocab_closed_path, self.vocab_open_path]:
            if not os.path.exists(path):
                missing_files.append(path)

        if missing_files:
            self.is_loaded = False
            self.load_message = f"Missing vocabulary files: {missing_files}"
            return False, self.load_message

        try:
            # 1. Load Vocabularies
            with open(self.vocab_closed_path, 'r', encoding='utf-8') as f:
                vc = json.load(f)
                self.closed2idx = vc['closed2idx']
                self.idx2closed = {int(k): v for k, v in vc['idx2closed'].items()}

            with open(self.vocab_open_path, 'r', encoding='utf-8') as f:
                vo = json.load(f)
                self.open2idx = vo['open2idx']
                self.idx2open = {int(k): v for k, v in vo['idx2open'].items()}

            # 2. Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_backbone)

            # 3. Model Architecture
            self.model = MedVQA_N6(
                num_closed=len(self.closed2idx),
                num_open=len(self.open2idx),
                text_backbone=self.text_backbone
            ).to(self.device)

            # 4. Model Checkpoint Weights
            if not os.path.exists(self.model_path):
                self.is_loaded = False
                self.load_message = f"Model checkpoint not found: '{self.model_path}'"
                return False, self.load_message

            checkpoint = torch.load(self.model_path, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.model.load_state_dict(state_dict)
            self.model.eval()

            epoch = checkpoint.get('epoch', '?') if isinstance(checkpoint, dict) else '?'
            val_score = checkpoint.get('val_combined', 0.0) if isinstance(checkpoint, dict) else 0.0

            self.is_loaded = True
            self.load_message = f"Model successfully loaded on {self.device} (Epoch: {epoch}, Val Score: {val_score:.1f}%)"
            return True, self.load_message

        except Exception as e:
            self.is_loaded = False
            self.load_message = f"Error loading model: {str(e)}"
            return False, self.load_message

    def predict(self, image: Image.Image, question: str, generate_cam: bool = True) -> dict:
        if not self.is_loaded:
            ok, msg = self.load()
            if not ok:
                return {"error": msg}

        if image is None:
            return {"error": "No image provided."}

        question = (question or "").strip()
        if not question:
            return {"error": "No question provided."}

        # 1. Image Preprocessing
        rgb_image = image.convert('RGB')
        img_t = self.transform(rgb_image).unsqueeze(0).to(self.device)

        # 2. Text Tokenization
        tok = self.tokenizer(
            [question],
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )
        input_ids = tok['input_ids'].to(self.device)
        attn_mask = tok['attention_mask'].to(self.device)

        # 3. Model Forward Pass
        with torch.no_grad():
            logits_c, logits_o = self.model(img_t, input_ids, attn_mask)

        probs_c = torch.softmax(logits_c, dim=-1)[0]
        probs_o = torch.softmax(logits_o, dim=-1)[0]

        top3_c = probs_c.topk(min(3, len(probs_c)))
        top3_o = probs_o.topk(min(3, len(probs_o)))

        top3_closed = [
            {"answer": self.idx2closed.get(idx.item(), "?"), "confidence": round(p.item() * 100, 1)}
            for p, idx in zip(top3_c.values, top3_c.indices)
        ]

        top3_open = [
            {"answer": self.idx2open.get(idx.item(), "?"), "confidence": round(p.item() * 100, 1)}
            for p, idx in zip(top3_o.values, top3_o.indices)
        ]

        best_closed_ans = top3_closed[0]['answer'] if top3_closed else "?"
        best_closed_conf = top3_closed[0]['confidence'] if top3_closed else 0.0
        best_open_ans = top3_open[0]['answer'] if top3_open else "?"
        best_open_conf = top3_open[0]['confidence'] if top3_open else 0.0

        # Closed vs Open Selection Heuristic
        is_closed = (best_closed_conf > 60.0) and (best_closed_ans.lower() in CLOSED_KEYWORDS)
        answer_type_idx = 0 if is_closed else 1
        best_answer = best_closed_ans if is_closed else best_open_ans
        confidence = best_closed_conf if is_closed else best_open_conf

        result = {
            "answer": best_answer,
            "answer_type": "Closed (binary/yes-no)" if is_closed else "Open (clinical entity)",
            "confidence": confidence,
            "top3_closed": top3_closed,
            "top3_open": top3_open,
            "cam_base64": None,
            "cam_pil": None,
        }

        # 4. Grad-CAM Explainability
        if generate_cam:
            try:
                gcam = GradCAMViT(self.model)
                cam = gcam.generate(img_t, input_ids, attn_mask, answer_type=answer_type_idx)
                gcam.remove_hooks()

                cam_np = cam[0].cpu().numpy()
                cam_resized = np.array(
                    Image.fromarray((cam_np * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
                ) / 255.0

                img_resized = np.array(rgb_image.resize((224, 224))) / 255.0
                heatmap = plt_jet(cam_resized)
                overlay = np.clip(img_resized * 0.6 + heatmap * 0.4, 0, 1)
                overlay_pil = Image.fromarray((overlay * 255).astype(np.uint8))

                buf = io.BytesIO()
                overlay_pil.save(buf, format='JPEG', quality=90)
                result['cam_base64'] = base64.b64encode(buf.getvalue()).decode('utf-8')
                result['cam_pil'] = overlay_pil

            except Exception as e:
                result['cam_error'] = str(e)

        return result
