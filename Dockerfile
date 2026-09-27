# SatyaCheck v2 inference server. CPU-only (ORT backend), models pulled from HF Hub.
# Build: docker build -t satya-api .   Run: docker run -p 8000:8000 satya-api
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SATYA_RUNTIME=ort \
    SATYA_VAR=/data \
    HF_HUB_OFFLINE=0

# ffmpeg (probe/encode/resync/normalise) + libmagic + weasyprint system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libmagic1 \
    libpango-1.0-0 libpangoft2-1.0-0 libgdk-pixbuf-2.0-0 libffi-dev shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-server.txt .
RUN pip install --upgrade pip && pip install -r requirements-server.txt

# code only (data/, runs/, venv excluded via .dockerignore; reports ship for /v1/benchmarks)
COPY server/ ./server/
COPY satya_core/ ./satya_core/
COPY ml/ ./ml/
COPY contracts/ ./contracts/

# serving models (HF Hub, blueprint §4.12 distribution path)
ARG HF_REPO=Abhishek1445/satyacheck-v2
RUN mkdir -p models && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('$HF_REPO', local_dir='models', allow_patterns=['*.onnx','*.pt','*.txt','*.json'])" && \
    ls -la models/ && \
    python -c "import json; r=json.load(open('models/registry.json')); print('registry serving:', r.get('serving'))"

RUN mkdir -p /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/v1/health',timeout=8).status==200 else 1)"
CMD ["python", "-m", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
