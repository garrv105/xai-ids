FROM python:3.11-slim

LABEL maintainer="XAI-IDS"
LABEL description="Explainable AI Intrusion Detection System"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

ARG INSTALL_EXTRAS=cpu

WORKDIR /app

# Install CPU-only PyTorch first to avoid large CUDA pulls
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .
RUN pip install --no-cache-dir -e .

# Create output directories
RUN mkdir -p models outputs logs

EXPOSE 8000

ENTRYPOINT ["uvicorn", "xai_ids.api.server:app"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
