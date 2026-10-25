# CPU inference image for the EoT model API (serve/). Build and run from the repo root:
#   docker build -f serve/Dockerfile -t eot-serve .
#   docker run --rm -p 8000:8000 -v "$PWD/artifacts/models/both-8s:/model:ro" eot-serve
# The model directory (a save_pretrained dir such as <run>/best) is mounted at /model, not baked in.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# CPU-only torch first: the default index would pull the CUDA build (several GB more).
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

COPY serve/requirements.txt serve/requirements.txt
RUN pip install -r serve/requirements.txt

# The turn_detection package, core only (model + audio preprocessing); its dependencies are pinned above.
COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-deps .

COPY serve/ serve/

RUN useradd --create-home --uid 1000 app
USER app

ENV MODEL_DIR=/model \
    HOST=0.0.0.0 \
    PORT=8000 \
    DEVICE=cpu
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["python", "-m", "serve"]
