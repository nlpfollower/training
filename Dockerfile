FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    git \
    nvidia-utils-525 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Set up conda
RUN conda init bash

# Copy the environment.yaml file
COPY environment.yaml /tmp/environment.yaml

# Create conda environment
RUN conda env create -f /tmp/environment.yaml

# Set up shell to use the new environment by default
SHELL ["conda", "run", "-n", "llama-3", "/bin/bash", "-c"]

# Print installed packages
RUN pip list && conda list

RUN pip install \
    llama-models \
    loguru \
    pydevd-pycharm \
    sentence-transformers \
    tf-keras

# Set working directory
WORKDIR /workspace

# Copy your project files
COPY . /workspace

# Set the entrypoint to activate conda environment
ENTRYPOINT ["conda", "run", "-n", "llama-3"]

# Set the default command
CMD ["python", "-m", "scripts.train", "--model", "pythia-160m", "--method", "dpo", "--nnodes", "1", "--nproc_per_node", "1"]