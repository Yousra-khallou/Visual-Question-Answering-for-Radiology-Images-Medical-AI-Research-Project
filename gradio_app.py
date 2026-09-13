"""
MedVQA N6 — Interface Gradio (Hugging Face Spaces)
Lancement : python gradio_app.py
Accès     : http://localhost:7860
"""

import warnings
from PIL import Image
import gradio as gr
from src.pipeline import MedVQAPipeline

warnings.filterwarnings('ignore')

# Initialisation du pipeline partagé
pipeline = MedVQAPipeline()

def analyze(image: Image.Image, question: str):
    """Fonction de traitement pour l'interface Gradio."""
    if image is None:
        return "### ❌ Erreur\nAucune image fournie.", None, "", f"Device: {pipeline.device}"

    if not question or not question.strip():
        return "### ❌ Erreur\nVeuillez poser une question clinique.", None, "", f"Device: {pipeline.device}"

    result = pipeline.predict(image, question, generate_cam=True)

    if "error" in result:
        return f"### ⚠️ Erreur Modèle\n{result['error']}", None, "", f"Device: {pipeline.device}"

    # Formatage Markdown de la réponse principale
    result_md = (
        f"# 🩺 {result['answer'].upper()}\n\n"
        f"**Type de réponse :** {result['answer_type']}\n\n"
        f"**Niveau de Confiance :** `{result['confidence']:.1f}%`"
    )

    # Formatage du Top-3
    top3_txt = "### 📋 Meilleurs Candidats\n\n"
    top3_txt += "**Questions Fermées (Closed) :**\n"
    for i, item in enumerate(result.get('top3_closed', []), 1):
        top3_txt += f"- **{i}.** {item['answer']} — `{item['confidence']}%`\n"

    top3_txt += "\n**Questions Ouvertes (Open) :**\n"
    for i, item in enumerate(result.get('top3_open', []), 1):
        top3_txt += f"- **{i}.** {item['answer']} — `{item['confidence']}%`\n"

    info_txt = f"Device: {pipeline.device} | Statut: Prêt"
    cam_img = result.get('cam_pil')

    return result_md, cam_img, top3_txt, info_txt

# Construction de l'interface Gradio
with gr.Blocks(title="MedVQA N6 — Medical AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔬 MedVQA N6 — Visual Question Answering Médical
        *Architecture Multimodale : ViT-Base/16 + BiomedBERT + Fusion par Concaténation + Grad-CAM*
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(type="pil", label="Radiologie Médicale (X-Ray / CT / MRI)")
            question_in = gr.Textbox(
                label="Question Clinique",
                placeholder="Ex : Is there a pleural effusion? / What organ is visible? / Is this normal?",
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
            submit_btn = gr.Button("⚡ Analyser la Radiologie", variant="primary")

        with gr.Column(scale=1):
            result_out = gr.Markdown(label="Diagnostic Prédit")
            cam_out = gr.Image(type="pil", label="Explicabilité Visuelle (Grad-CAM)")
            top3_out = gr.Markdown(label="Distribution des Probabilités")
            info_out = gr.Textbox(label="Informations Système", interactive=False)

    submit_btn.click(
        fn=analyze,
        inputs=[image_in, question_in],
        outputs=[result_out, cam_out, top3_out, info_out]
    )

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860, share=False)
