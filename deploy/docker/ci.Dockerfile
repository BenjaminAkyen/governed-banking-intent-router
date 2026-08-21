# syntax=docker/dockerfile:1.7
# Module 16 passes a digest-pinned Python image; no mutable default is allowed.
ARG PYTHON_CI_IMAGE
FROM ${PYTHON_CI_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY LICENSE README.md pyproject.toml ./
COPY src ./src
RUN python -m pip install --no-cache-dir --no-deps .

USER 65532:65532

CMD ["python", "-c", "import importlib.metadata; print(importlib.metadata.version('governed-banking-intent-router'))"]
