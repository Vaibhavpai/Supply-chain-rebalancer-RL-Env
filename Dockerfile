# Supply Chain Inventory Rebalancer — Hugging Face Spaces Deployment
FROM python:3.11-slim

# Set up a new user named "user" with user ID 1000 (Required by HF)
RUN useradd -m -u 1000 user
USER user

# Set home to the user's home directory and add it to PATH
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Install dependencies FIRST (best practice for Docker caching)
# Note: Pip runs as 'user', avoiding the root warning
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all remaining project files
# The --chown flag ensures 'user' owns all these files
COPY --chown=user models.py .
COPY --chown=user environment.py .
COPY --chown=user baseline.py .
COPY --chown=user inference.py .
COPY --chown=user openenv.yaml .
COPY --chown=user app.py .

# HuggingFace token
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

# Run inference script
# Launch the FastAPI heartbeat server to keep the Space awake
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]