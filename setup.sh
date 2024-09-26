#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

apt-get update
apt-get install -y --no-install-recommends wget git
rm -rf /var/lib/apt/lists/*
conda install mamba -n base -c conda-forge
mamba env create -f /tmp/environment.yaml
conda init bash
echo 'conda activate llama-3' >> ~/.bashrc
/opt/conda/envs/llama-3/bin/pip install llama-models loguru pydevd-pycharm sentence-transformers tf-keras fire python-dotenv tensorflow_hub nltk openai accelerate bitsandbytes fastapi uvicorn flash-attn

# Create entrypoint script
cat << EOF > /usr/local/bin/entrypoint.sh
#!/bin/bash
source ~/.bashrc
conda activate llama-3
exec "\$@"
EOF

chmod +x /usr/local/bin/entrypoint.sh

# Verify environment creation
conda env list