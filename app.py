"""
MedVQA N6 — Serveur Web Flask
Lancement : python app.py
Accès     : http://localhost:5000
"""

import os
import time
import warnings
from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch

from src.pipeline import MedVQAPipeline

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 Mo max
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialisation du pipeline d'inférence
pipeline = MedVQAPipeline()

@app.route('/')
def index():
    """Page d'accueil du dashboard médical."""
    return render_template('index.html')

@app.route('/status')
def status():
    """État du serveur et du modèle."""
    ok, msg = pipeline.load()
    return jsonify({
        "loaded": pipeline.is_loaded,
        "device": str(pipeline.device),
        "message": msg,
        "vocab": {
            "closed": len(pipeline.closed2idx),
            "open": len(pipeline.open2idx),
        } if pipeline.is_loaded else {}
    })

@app.route('/predict', methods=['POST'])
def predict_route():
    """Endpoint d'inférence multimodale et Grad-CAM."""
    if not pipeline.is_loaded:
        ok, msg = pipeline.load()
        if not ok:
            return jsonify({"error": msg}), 500

    if 'image' not in request.files:
        return jsonify({"error": "Aucune image fournie"}), 400

    question = request.form.get('question', '').strip()
    if not question:
        return jsonify({"error": "Aucune question fournie"}), 400

    file = request.files['image']
    if not file or file.filename == '':
        return jsonify({"error": "Fichier image vide"}), 400

    try:
        image = Image.open(file.stream).convert('RGB')
        t0 = time.time()
        result = pipeline.predict(image, question, generate_cam=True)
        result['processing_time_ms'] = round((time.time() - t0) * 1000)

        # Nettoyer l'objet PIL non sérialisable en JSON
        if 'cam_pil' in result:
            del result['cam_pil']

        if "error" in result:
            return jsonify(result), 500

        return jsonify(result)

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route('/load_model', methods=['POST'])
def load_model_route():
    """Recharge manuellement le modèle."""
    ok, msg = pipeline.load(force_reload=True)
    return jsonify({"success": ok, "message": msg})

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  MedVQA N6 — Serveur Web Médical")
    print(f"  Device  : {pipeline.device}")
    print("  URL     : http://localhost:5000")
    print("=" * 60 + "\n")
    
    # Pré-chargement silencieux au démarrage si les fichiers existent
    ok, msg = pipeline.load()
    print(f"Statut modèle : {msg}")
    
    app.run(debug=False, host='0.0.0.0', port=5000)
