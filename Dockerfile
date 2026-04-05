# Supply Chain Inventory Rebalancer — Hugging Face Spaces Deployment
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY models.py .
COPY environment.py .
COPY baseline.py .
COPY inference.py .
COPY openenv.yaml .

# HuggingFace token (passed at build or runtime via --build-arg or -e)
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# Environment variables for LLM API
ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

# Hugging Face Spaces port
EXPOSE 7860

LABEL org.opencontainers.image.title="Supply Chain Inventory Rebalancer"
LABEL org.opencontainers.image.description="OpenEnv RL benchmark — LLM planning over warehouse logistics"
LABEL space_sdk="docker"

# Run inference script (all 3 tasks)
CMD ["python", "inference.py"]
