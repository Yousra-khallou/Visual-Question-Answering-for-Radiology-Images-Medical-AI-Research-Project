# ═══════════════════════════════════════════════
# MedVQA N6 — Dockerfile
# Build : docker build -t medvqa-n6 .
# Run   : docker run -p 5000:5000 medvqa-n6
# ═══════════════════════════════════════════════

FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY app.py .
COPY templates/ templates/
COPY static/ static/

# Model files (doivent être présents avant docker build)
# Ou monter avec -v $(pwd)/model:/app/model
COPY model/ model/

# Non-root user (sécurité)
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 5000

CMD ["gunicorn", "-w", "1", "--timeout", "120", "-b", "0.0.0.0:5000", "app:app"]
