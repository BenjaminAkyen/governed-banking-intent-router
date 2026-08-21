# syntax=docker/dockerfile:1.7
# Supply an organisation-approved, digest-pinned Python 3.12 Linux image.
ARG PYTHON_CPU_IMAGE
FROM ${PYTHON_CPU_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    GOVERNED_BANKING_CONTAINER=1 \
    GOVERNED_BANKING_DEPLOYMENT_PROFILE=configs/deployment/linux-cpu.yaml

WORKDIR /app

RUN groupadd --system --gid 10001 router \
    && useradd --system --uid 10001 --gid router --home-dir /nonexistent router \
    && mkdir -p /app/artifacts/audit \
    && chown -R router:router /app

COPY --chown=router:router LICENSE README.md pyproject.toml ./
COPY --chown=router:router src ./src
RUN python -m pip install --no-cache-dir .

COPY --chown=router:router configs ./configs
COPY --chown=router:router data/manifests ./data/manifests
COPY --chown=router:router reports/calibration ./reports/calibration

USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=300s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2).read()"]

CMD ["uvicorn", "governed_banking.deployment_service:create_app_from_environment", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log", "--timeout-graceful-shutdown", "45"]
