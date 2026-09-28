# ═══════════════════════════════════════════════════════════════════════════
# Stage 0 — Gnina Builder (Blackwell SM 12.x GPU support)
# ═══════════════════════════════════════════════════════════════════════════
FROM nvidia/cuda:12.9.0-devel-ubuntu24.04 AS gnina-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget ca-certificates \
    libboost-all-dev libeigen3-dev libgoogle-glog-dev \
    libprotobuf-dev protobuf-compiler libhdf5-dev libatlas-base-dev \
    libopenbabel-dev swig python3-dev python3-pip python3-numpy \
    libjsoncpp-dev python3-pytest \
    cuda-nvtx-12-9 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch v1.3.2 \
    https://github.com/gnina/gnina.git /gnina-src \
    && mkdir -p /gnina-src/build

WORKDIR /gnina-src/build
# pip-installed torch includes libtorch C++ headers/libs with proper CUDA
# support. Point cmake to it so FetchContent doesn't re-download.
# --break-system-packages: needed for Ubuntu 24.04 PEP 668.
RUN pip install --break-system-packages --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu129 \
    && rm -rf build/* \
    && cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES="50;52;60;61;70;75;80;86;89;90;120" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
    && make -j$(nproc)

RUN mkdir -p /gnina-install && make install DESTDIR=/gnina-install

# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 — ESMFold2 Production Image
# ═══════════════════════════════════════════════════════════════════════════
FROM ubuntu:24.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    DEBIAN_FRONTEND=noninteractive

# USTC mirror disabled (unreachable 2026-06-03) — using default Ubuntu archives
# RUN sed -i 's|ports.ubuntu.com|mirrors.ustc.edu.cn|g' /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null; \
#     sed -i 's|archive.ubuntu.com|mirrors.ustc.edu.cn|g' /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null; \
#     sed -i 's|security.ubuntu.com|mirrors.ustc.edu.cn|g' /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null

# ── Step 1: Minimal system tools ────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates gnupg openssl \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    git build-essential cmake \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

# ── Step 2: CUDA 12.9 ────────────────────────────────────────────────────
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update && apt-get install -y --no-install-recommends \
    cuda-runtime-12-9 libcublas-12-9 libcudnn9-cuda-12 libnccl2 cuda-nvtx-12-9 libcupti-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/local/cuda-12/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:/opt/conda/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-} \
    CUDA_HOME=/usr/local/cuda-12

# ── Step 3: OpenBabel + SWIG (ligand/receptor format conversion) ────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    openbabel libopenbabel-dev swig \
    && rm -rf /var/lib/apt/lists/*

# ═══════════════════════════════════════════════════════════════════════════
# ── Step 4: Miniforge → mamba → AmberTools  ★ MUST come before all pip ───
#            so every Python package lands in the conda environment.
# ═══════════════════════════════════════════════════════════════════════════
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh
ENV PATH=/opt/conda/bin:${PATH}

RUN conda config --set channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/; \
    conda config --set show_channel_urls yes; \
    conda install -y mamba -c conda-forge

# Downgrade Python first (simpler solve), then ambertools (huge but now Python is fixed)
RUN mamba install -y python=3.12

RUN mamba install -y ambertools \
    && ln -sf /opt/conda/bin/antechamber /usr/local/bin/antechamber \
    && ln -sf /opt/conda/bin/tleap /usr/local/bin/tleap \
    && ln -sf /opt/conda/bin/sander /usr/local/bin/sander \
    && ln -sf /opt/conda/bin/cpptraj /usr/local/bin/cpptraj \
    && ln -sf /opt/conda/bin/pdb4amber /usr/local/bin/pdb4amber \
    && ln -sf /opt/conda/bin/MMPBSA.py /usr/local/bin/MMPBSA.py \
    && ln -sf /opt/conda/bin/parmchk2 /usr/local/bin/parmchk2

# ── Step 5: Python package manager config ───────────────────────────────
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || true

WORKDIR /app

# ── Step 6: PyTorch with CUDA 12.9 ──────────────────────────────────────
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# ── Step 7: Transformers + HuggingFace ──────────────────────────────────
RUN pip install --no-cache-dir transformers huggingface_hub

# ── Step 8: Core Python requirements ────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt; \
    pip install --no-cache-dir esm@git+https://github.com/Biohub/esm.git@main

# ── Step 9: fpocket (pocket detection) ──────────────────────────────────
RUN cd /tmp && \
    git clone --depth 1 https://github.com/Discngine/fpocket.git && \
    cd fpocket && make && \
    cp bin/fpocket /usr/local/bin/ && \
    rm -rf /tmp/fpocket

COPY Pocket-Detection-of-DTWG-main/TMalign /usr/local/bin/TMalign
RUN chmod +x /usr/local/bin/TMalign

# ═══════════════════════════════════════════════════════════════════════════
# Cascade pipeline dependencies (now in conda environment)
# ═══════════════════════════════════════════════════════════════════════════

# ── Step 10: Gnina (self-built, Blackwell SM 12.x GPU support) ─────────────
COPY --from=gnina-builder /gnina-install/usr/local/bin/gnina /usr/local/bin/gnina
COPY --from=gnina-builder /gnina-install/usr/local/lib/ /usr/local/lib/
# System deps for gnina runtime (uses system openbabel + boost from apt)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 libboost-program-options1.83.0 libboost-filesystem1.83.0 \
    libboost-system1.83.0 libboost-iostreams1.83.0 libboost-serialization1.83.0 \
    libboost-thread1.83.0 libboost-chrono1.83.0 libboost-atomic1.83.0 \
    libboost-timer1.83.0 libboost-regex1.83.0 libjsoncpp25 libnuma1 \
    && rm -rf /var/lib/apt/lists/* && ldconfig

# ── Step 11: OpenMM (CUDA via conda), MDTraj, Dimorphite, PoseBusters, xtb ──
# xtb: GFN2-xTB semi-empirical, ~2000× faster than sqm AM1-BCC for charges
RUN mamba install -y -c conda-forge openmm xtb \
    && pip install --no-cache-dir \
    mdtraj dimorphite_dl posebusters pdb2pqr

# ── Step 12: OpenFF + NAGL/Espaloma (GNN charges) ──────────────────────
# .conda packages contain pkg-*.tar.zst; extract manually to avoid CPU pytorch dep
COPY docker/extract_openff.py /tmp/extract_openff.py
RUN mamba install -y --download-only \
    openff-toolkit openff-nagl openff-nagl-models espaloma_charge; \
    python /tmp/extract_openff.py; \
    rm /tmp/extract_openff.py; \
    pip install --no-cache-dir dgl openff-units openff-utilities xmltodict || true

# Boltz-2 scoring venv (isolated: numpy 1.x vs main env 2.x)
RUN python -m venv /opt/boltz-venv && \
    /opt/boltz-venv/bin/pip install --no-cache-dir boltz[cuda]

# Nesso-1 affinity scoring venv (isolated: numpy 2.x vs boltz 1.x)
COPY paper/nesso/nesso-main /tmp/nesso-src
RUN python -m venv /opt/nesso-venv && \
    /opt/nesso-venv/bin/pip install --no-cache-dir -e /tmp/nesso-src && \
    rm -rf /tmp/nesso-src

# ═══════════════════════════════════════════════════════════════════════════

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

COPY app/ ./app/
COPY validation/ ./validation/
COPY drugclip/ ./drugclip/

RUN mkdir -p /cache/huggingface /cache/torch /output

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["api"]
