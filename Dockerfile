# Reproducible environment for pcdiff: Python 3.8, CUDA 11.1, torch 1.9.0,
# MinkowskiEngine 0.5.4. Only an NVIDIA driver (>= 455) and the NVIDIA
# Container Toolkit are needed on the host.
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip git build-essential ninja-build \
        libopenblas-dev libgl1 libglib2.0-0 libgomp1 libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/local/bin/python

RUN python3 -m pip install --upgrade "pip<24.1" "setuptools==59.5.0" wheel

RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
        -f https://download.pytorch.org/whl/torch_stable.html

# Build MinkowskiEngine for all common GPU architectures (Pascal to Ampere,
# plus PTX so newer GPUs can JIT-compile), independent of the build machine.
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    FORCE_CUDA=1 \
    MAX_JOBS=4
RUN pip install numpy==1.23.5 \
    && git clone --depth 1 --branch v0.5.4 https://github.com/NVIDIA/MinkowskiEngine.git /tmp/ME \
    && cd /tmp/ME \
    && python3 setup.py install --force_cuda --blas=openblas \
    && cd / && rm -rf /tmp/ME

COPY docker/requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install -r /tmp/requirements-docker.txt

# The code is mounted at /workspace at runtime; train.py calls git.
RUN git config --system --add safe.directory '*'
ENV PYTHONPATH=/workspace
WORKDIR /workspace/pcdiff

CMD ["bash"]
