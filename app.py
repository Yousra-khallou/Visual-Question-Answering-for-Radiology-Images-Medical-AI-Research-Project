"""
MedVQA N6 — Flask Production Web Server & API
Run with : python app.py
Access   : http://localhost:5000
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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload limit
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize inference pipeline
pipeline = MedVQAPipeline()

@app.route('/')
def index():
    """Diagnostic Web Dashboard."""
    return render_template('index.html')

@app.route('/status')
def status():
    """System health & model status endpoint."""
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
    """Multimodal inference and Grad-CAM generation endpoint."""
    if not pipeline.is_loaded:
        ok, msg = pipeline.load()
        if not ok:
            return jsonify({"error": msg}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    question = request.form.get('question', '').strip()
    if not question:
        return jsonify({"error": "No question provided."}), 400

    file = request.files['image']
    if not file or file.filename == '':
        return jsonify({"error": "Empty image file."}), 400

    try:
        image = Image.open(file.stream).convert('RGB')
        t0 = time.time()
        result = pipeline.predict(image, question, generate_cam=True)
        result['processing_time_ms'] = round((time.time() - t0) * 1000)

        # Remove non-serializable PIL image before JSON serialization
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
    """Manually triggers model reload."""
    ok, msg = pipeline.load(force_reload=True)
    return jsonify({"success": ok, "message": msg})

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  MedVQA N6 — Clinical AI Diagnostic Web Server")
    print(f"  Device  : {pipeline.device}")
    print("  URL     : http://localhost:5000")
    print("=" * 60 + "\n")
    
    # Pre-load model if checkpoint is present
    ok, msg = pipeline.load()
    print(f"Model status: {msg}")
    
    app.run(debug=False, host='0.0.0.0', port=5000)
