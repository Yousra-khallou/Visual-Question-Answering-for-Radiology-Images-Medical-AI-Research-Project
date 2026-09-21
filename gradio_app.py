"""
MedVQA N6 — Gradio Interface (Hugging Face Spaces)
Run with : python gradio_app.py
Access   : http://localhost:7860
"""

import warnings
from PIL import Image
import gradio as gr
from src.pipeline import MedVQAPipeline

warnings.filterwarnings('ignore')

# Shared inference pipeline
pipeline = MedVQAPipeline()

def analyze(image: Image.Image, question: str):
    """Handler function for Gradio interface."""
    if image is None:
        return "### ❌ Error\nNo image provided.", None, "", f"Device: {pipeline.device}"

    if not question or not question.strip():
        return "### ❌ Error\nPlease provide a clinical question.", None, "", f"Device: {pipeline.device}"

    result = pipeline.predict(image, question, generate_cam=True)

    if "error" in result:
        return f"### ⚠️ Model Error\n{result['error']}", None, "", f"Device: {pipeline.device}"

    # Markdown format for main diagnostic answer
    result_md = (
        f"# 🩺 {result['answer'].upper()}\n\n"
        f"**Answer Type:** {result['answer_type']}\n\n"
        f"**Diagnostic Confidence:** `{result['confidence']:.1f}%`"
    )

    # Top-3 candidates list
    top3_txt = "### 📋 Top Candidates\n\n"
    top3_txt += "**Closed-Ended Questions:**\n"
    for i, item in enumerate(result.get('top3_closed', []), 1):
        top3_txt += f"- **{i}.** {item['answer']} — `{item['confidence']}%`\n"

    top3_txt += "\n**Open-Ended Questions:**\n"
    for i, item in enumerate(result.get('top3_open', []), 1):
        top3_txt += f"- **{i}.** {item['answer']} — `{item['confidence']}%`\n"

    info_txt = f"Device: {pipeline.device} | Status: Ready"
    cam_img = result.get('cam_pil')

    return result_md, cam_img, top3_txt, info_txt

# Gradio Blocks Layout
with gr.Blocks(title="MedVQA N6 — Medical AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔬 MedVQA N6 — Medical Visual Question Answering
        *Multimodal Deep Learning Architecture: ViT-Base/16 + BiomedBERT + Concatenation Fusion + Grad-CAM*
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(type="pil", label="Radiology Image (X-Ray / CT / MRI)")
            question_in = gr.Textbox(
                label="Clinical Question",
                placeholder="e.g., Is there a pleural effusion? / What organ is visible? / Is this normal?",
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
            submit_btn = gr.Button("⚡ Analyze Radiology", variant="primary")

        with gr.Column(scale=1):
            result_out = gr.Markdown(label="Predicted Diagnosis")
            cam_out = gr.Image(type="pil", label="Visual Explainability (Grad-CAM)")
            top3_out = gr.Markdown(label="Candidate Probability Distribution")
            info_out = gr.Textbox(label="System Information", interactive=False)

    submit_btn.click(
        fn=analyze,
        inputs=[image_in, question_in],
        outputs=[result_out, cam_out, top3_out, info_out]
    )

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860, share=False)
