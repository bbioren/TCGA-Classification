# Base image with CUDA and Ubuntu
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

# Clean update and run
RUN rm -rf /var/lib/apt/lists/* \
    && apt-get update -o Acquire::CompressionTypes::Order::=gz \
    && apt-get install -y python3.12 python3.12-venv python3-pip git curl wget unzip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install core ML dependencies
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install transformers sentence-transformers einops pandas scikit-learn tqdm

# Jina embeddings
RUN pip install "jinaai/jina-embeddings-v3"

# Set working directory
WORKDIR /workspace

# Copy your fine-tune code and data folder
COPY fine_tune.py /workspace/
COPY data/ /workspace/data/

# Optional: set default command
CMD ["python3", "fine_tune.py", "--data", "/workspace/data/train.csv", "--text_col", "text", "--label_col", "OS"]
