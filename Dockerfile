FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install mamba for faster environment creation
RUN conda install mamba -n base -c conda-forge

# Copy only the environment file
COPY environment.yaml /tmp/environment.yaml

# Create conda environment
RUN mamba env create -f /tmp/environment.yaml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "llama-3", "/bin/bash", "-c"]

# Install additional packages
RUN pip install llama-models loguru pydevd-pycharm sentence-transformers tf-keras fire python-dotenv tensorflow_hub nltk \
    openai accelerate bitsandbytes

# Set up conda environment activation
RUN echo "source activate llama-3" > ~/.bashrc

# Set the entrypoint to activate conda environment
ENTRYPOINT ["/bin/bash", "-c"]

# Set a default command that just starts a bash shell
CMD ["/bin/bash"]