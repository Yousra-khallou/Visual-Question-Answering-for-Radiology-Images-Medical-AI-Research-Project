import torch
import torch.nn as nn
import timm
from transformers import AutoModel

DEFAULT_BIOMEDBERT = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"

class MedVQA_N6(nn.Module):
    """
    MedVQA N6 Multimodal Neural Architecture:
    - Visual Encoder : Vision Transformer (ViT-Base/16-224)
    - Text Encoder   : BiomedBERT (BiomedNLP-BiomedBERT-base-uncased-abstract)
    - Multimodal Fusion: Linear Projections + Concatenation + BatchNorm + Dropout
    - Dual Prediction Heads: Closed (binary/yes-no) and Open (clinical entities)
    """
    def __init__(
        self,
        num_closed: int,
        num_open: int,
        dim: int = 768,
        proj_dim: int = 512,
        text_backbone: str = DEFAULT_BIOMEDBERT
    ):
        super().__init__()
        self.visual_encoder = timm.create_model(
            'vit_base_patch16_224', pretrained=False, num_classes=0
        )
        self.text_encoder = AutoModel.from_pretrained(text_backbone)
        self.v_proj = nn.Sequential(
            nn.Linear(dim, proj_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.q_proj = nn.Sequential(
            nn.Linear(dim, proj_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.fusion = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(proj_dim, proj_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.bn = nn.BatchNorm1d(proj_dim // 2)
        self.head_closed = nn.Linear(proj_dim // 2, num_closed)
        self.head_open = nn.Sequential(
            nn.Linear(proj_dim // 2, proj_dim // 4),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(proj_dim // 4, num_open),
        )

    def forward(self, img: torch.Tensor, input_ids: torch.Tensor, attn_mask: torch.Tensor):
        v = self.visual_encoder.forward_features(img).mean(dim=1)
        q = self.text_encoder(
            input_ids=input_ids, attention_mask=attn_mask
        ).last_hidden_state[:, 0, :]
        
        fused = self.fusion(torch.cat([self.v_proj(v), self.q_proj(q)], dim=-1))
        fused = self.bn(fused)
        return self.head_closed(fused), self.head_open(fused)
