import torch
import torch.nn.functional as F

class GradCAMViT:
    """
    Grad-CAM Visual Explainability Module for Vision Transformers (ViT).
    Captures layer activations and backward gradients at the final normalization layer.
    """
    def __init__(self, model):
        self.model = model
        self.grads = []
        self.acts = []
        self.hook_f = model.visual_encoder.norm.register_forward_hook(
            lambda m, i, o: self.acts.append(o)
        )
        self.hook_b = model.visual_encoder.norm.register_full_backward_hook(
            lambda m, gi, go: self.grads.append(go[0])
        )

    def remove_hooks(self):
        """Detach forward and backward hooks to release memory."""
        if hasattr(self, 'hook_f') and self.hook_f:
            self.hook_f.remove()
        if hasattr(self, 'hook_b') and self.hook_b:
            self.hook_b.remove()

    def generate(
        self,
        img_t: torch.Tensor,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        answer_type: int = 0
    ) -> torch.Tensor:
        """
        Generates normalized Grad-CAM attention heatmap [B, 14, 14].
        answer_type: 0 for 'closed', 1 for 'open'.
        """
        self.model.eval()
        self.grads.clear()
        self.acts.clear()
        
        logits_c, logits_o = self.model(img_t, input_ids, attn_mask)
        score = logits_c.max() if answer_type == 0 else logits_o.max()
        
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        if not self.grads or not self.acts:
            raise RuntimeError("Unable to extract gradients or activations for Grad-CAM.")

        weights = self.grads[0].mean(dim=1, keepdim=True)
        cam = F.relu((weights * self.acts[0]).sum(dim=-1)[:, 1:])
        B = cam.size(0)
        cam = cam.view(B, 14, 14)
        
        # Min-Max Normalization per batch item
        cam_min = cam.view(B, -1).min(dim=1)[0].view(B, 1, 1)
        cam_max = cam.view(B, -1).max(dim=1)[0].view(B, 1, 1)
        cam = (cam - cam_min) / (cam_max + 1e-8)
        
        return cam.detach()
