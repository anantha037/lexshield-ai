FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    tesseract-ocr \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install gcloud CLI
RUN curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts
ENV PATH="/root/google-cloud-sdk/bin:${PATH}"

# CPU-only torch first — prevents pip pulling 2GB CUDA wheel
RUN pip install --no-cache-dir \
    torch==2.12.0+cpu \
    torchvision==0.27.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# spaCy models
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download en_core_web_lg

# Sentence transformer model
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"

# Copy project code
COPY . .
RUN chmod +x startup.sh

CMD ["./startup.sh"]