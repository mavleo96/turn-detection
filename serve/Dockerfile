# CPU inference image for the EoT model API (serve/). Build and run from the repo root:
#   docker build -f serve/Dockerfile -t eot-serve .
#   docker run --rm -p 8000:8000 -v "$PWD/artifacts/models/both-8s:/model:ro" eot-serve
# The model directory (a checkpoint such as <run>/best, with model.onnx from scripts/export_onnx.py) is mounted
# at /model, not baked in. No torch: the model runs in ONNX Runtime.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY serve/requirements.txt serve/requirements.txt
RUN pip install -r serve/requirements.txt

# The turn_detection package without its dependencies: serve imports only turn_detection.data (numpy, soxr).
COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-deps .

COPY serve/ serve/

RUN useradd --create-home --uid 1000 app
USER app

# Note: numpy's OpenBLAS threads (the feature extractor's mel matmul) busy-wait after each call and take the cores
# ONNX Runtime needs: 2 ONNX Runtime threads on 2 cores, preprocessing + model 87 ms with them, 42 ms without
ENV MODEL_DIR=/model \
    HOST=0.0.0.0 \
    PORT=8000 \
    OPENBLAS_NUM_THREADS=1
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["python", "-m", "serve"]
