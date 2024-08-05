#!/bin/bash
apt-get update
apt-get install -y --no-install-recommends wget git
rm -rf /var/lib/apt/lists/*
conda install mamba -n base -c conda-forge
mamba env create -f /tmp/environment.yaml
conda init bash
echo 'source activate llama-3' >> ~/.bashrc
/opt/conda/envs/llama-3/bin/pip install llama-models loguru pydevd-pycharm sentence-transformers tf-keras fire python-dotenv tensorflow_hub nltk openai accelerate bitsandbytes
echo '#!/bin/bash' > /usr/local/bin/entrypoint.sh
echo 'source ~/.bashrc' >> /usr/local/bin/entrypoint.sh
echo 'exec "$@"' >> /usr/local/bin/entrypoint.sh
chmod +x /usr/local/bin/entrypoint.sh
