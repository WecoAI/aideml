# Prebaked OpenRLHF + vLLM image for the AIDE RLHF pipeline (axia H200 cluster).
#
# Mirrors the official OpenRLHF dockerfile (nvcr.io/nvidia/pytorch base,
# conflicting packages removed, vllm==0.19.1 + prebuilt flash-attn wheel)
# and adds the aideml project dependencies so jobs only need a fast
# `pip install -e . --no-deps` at runtime.
#
# Build & push (from repo root):
#   docker build -f docker/openrlhf/Dockerfile -t <dockerhub-user>/aide-openrlhf:0.1 .
#   docker push <dockerhub-user>/aide-openrlhf:0.1
#
# NOTE: build on an x86_64 machine (or use `docker buildx build --platform
# linux/amd64`) — Apple Silicon defaults to arm64, which won't run on the cluster.

FROM nvcr.io/nvidia/pytorch:25.11-py3

WORKDIR /app

# Base image ships packages that conflict with vLLM / flash-attn.
# nvtx: its DummyDomain API breaks deepspeed's profiling wrapper
# (TypeError in push_range); deepspeed falls back to torch.cuda.nvtx without it.
RUN pip uninstall -y xgboost transformer_engine flash_attn pynvml opencv-python-headless nvtx || true

# Same pins as the official OpenRLHF dockerfile (torch 2.10 / cu130 / py312).
RUN pip install --no-cache-dir vllm==0.19.1 && \
    pip install --no-cache-dir \
      https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu130torch2.10-cp312-cp312-linux_x86_64.whl

RUN pip install --no-cache-dir "openrlhf[vllm]" --no-build-isolation

# aideml runtime deps (the package itself is uploaded by Polyaxon and
# installed at runtime with `pip install -e .`; keep this list in sync with
# pyproject.toml [project.dependencies] so that install is a no-op).
RUN pip install --no-cache-dir \
      anthropic \
      backoff \
      black \
      coolname \
      dataclasses-json \
      funcy \
      genson \
      humanize \
      igraph \
      ipython \
      jsonschema \
      omegaconf \
      openai \
      python-dotenv \
      rich \
      shutup \
      tqdm \
      redelex \
      relbench \
      huggingface_hub \
      pyarrow

# Sanity check at build time so a broken image fails fast.
RUN python -c "import torch, vllm, openrlhf, ray; print('torch', torch.__version__, '| vllm', vllm.__version__, '| ray', ray.__version__)"
