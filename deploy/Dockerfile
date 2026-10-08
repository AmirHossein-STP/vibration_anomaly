# One image per service, built from one file:
#   docker build -f deploy/Dockerfile --target bearing   -t ad-bearing:dev   .
#   docker build -f deploy/Dockerfile --target vibration -t ad-vibration:dev .
#   docker build -f deploy/Dockerfile --target hydraulic -t ad-hydraulic:dev .
# Run (see deploy/docker-compose.yml and deploy/ad):  docker compose run --rm bearing infer
#
# CPU only. For a GPU later, add a `-gpu` target that installs the CUDA torch wheel and
# is used for `train` only; nothing else changes.
FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src SERVING_DATA_ROOT=/data SERVING_MODELS_ROOT=/models
# numpy/scipy/OpenBLAS threads: keep inference polite on shared servers
ENV OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
RUN useradd --create-home --uid 1000 app && mkdir -p /data /models && chown app:app /data /models
WORKDIR /app
COPY deploy/requirements-base.txt /tmp/requirements-base.txt
RUN pip install -r /tmp/requirements-base.txt
# only the code the services import (research notebooks, data and reports stay out)
COPY src/serving_core /app/src/serving_core
COPY src/vibration_ad /app/src/vibration_ad
COPY src/hydraulic_fusion_ad /app/src/hydraulic_fusion_ad
COPY src/domain_adaptation /app/src/domain_adaptation
ARG SERVICE_IMAGE_TAG=dev
ENV SERVICE_IMAGE_TAG=${SERVICE_IMAGE_TAG}
USER app
VOLUME ["/data", "/models"]

# ---- numpy/scipy only ----------------------------------------------------
FROM base AS bearing
ENTRYPOINT ["python", "-m", "serving_core", "bearing"]
CMD ["--help"]

# ---- + PyTorch (CPU) -----------------------------------------------------
FROM base AS torch-base
USER root
COPY deploy/requirements-torch.txt /tmp/requirements-torch.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements-torch.txt
USER app

FROM torch-base AS vibration
ENTRYPOINT ["python", "-m", "serving_core", "vibration"]
CMD ["--help"]

FROM torch-base AS hydraulic
ENTRYPOINT ["python", "-m", "serving_core", "hydraulic"]
CMD ["--help"]
