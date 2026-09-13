# ═══════════════════════════════════════════════
# MedVQA N6 — Production Dockerfile
# Build : docker build -t medvqa-n6 .
# Run   : docker run -p 5000:5000 medvqa-n6
# ═══════════════════════════════════════════════

FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application source code
COPY src/ src/
COPY app.py gradio_app.py ./
COPY templates/ templates/
COPY static/ static/
COPY model/ model/

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 5000

CMD ["gunicorn", "-w", "1", "--timeout", "120", "-b", "0.0.0.0:5000", "app:app"]
