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

# Legal NER (production — active only when USE_LEGAL_NER=true)
# RUN pip install https://huggingface.co/opennyaiorg/en_legal_ner_trf/resolve/main/en_legal_ner_trf-any-py3-none-any.whl

# Copy entire project including data/chroma_db and models/saved/
COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]