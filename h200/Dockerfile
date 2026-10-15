# Draft-OPD on one H200: paper-faithful training and Table 1 evaluation.
#
# Build from the harness root (the context needs patches/ and repro/):
#   docker build -f h200/Dockerfile -t draft-opd-h200 .
#
# Run (models, data, checkpoints and logs all go to the /workspace volume):
#   docker run --gpus '"device=0"' --shm-size=64g --ulimit memlock=-1 \
#     -v /path/on/host:/workspace -it draft-opd-h200
#   # then, inside:  bash h200/run_all.sh
#
# Base: the SGLang release the DFlash fork is cut from (0.5.8), CUDA 12.9 devel
# (nvcc for flash-attn), Python 3.12, torch 2.9.1+cu129. The verl project's own
# SGLang image uses the same lineage (lmsysorg/sglang:v0.5.9).
FROM lmsysorg/sglang:v0.5.8-cu129-amd64

ARG UPSTREAM_COMMIT=d6f0ec859321959785c65b59eb311fd25f042158
ARG MAX_JOBS=16

ENV PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    UPSTREAM_COMMIT=${UPSTREAM_COMMIT} \
    UPSTREAM_DIR=/opt/Draft-OPD \
    WORKSPACE=/workspace \
    HF_HOME=/workspace/hf \
    PYTHONUNBUFFERED=1 \
    RAY_DEDUP_LOGS=0

WORKDIR /opt/draft-opd-repro
COPY h200/ h200/
COPY patches/ patches/
COPY repro/make_val_jsonl.py repro/make_val_jsonl.py

# Code is baked into /opt so the editable installs survive any /workspace mount.
RUN WORKSPACE=/tmp/ws bash h200/00_get_upstream.sh \
 && WORKSPACE=/tmp/ws MAX_JOBS=${MAX_JOBS} bash h200/01_install.sh \
 && rm -rf /tmp/ws /root/.cache/pip

VOLUME ["/workspace"]
CMD ["/bin/bash"]
