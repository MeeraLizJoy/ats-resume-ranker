FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- CRITICAL CHANGE ---
# 1. Create the directories FIRST
RUN mkdir -p /app/output/model-last /app/models

# 2. Copy the download script
COPY download_models.py .

# 3. Run the download script to fill the directories
RUN python download_models.py

# 4. NOW copy the rest of the app code
COPY . .

# Download spaCy base model
RUN python -m spacy download en_core_web_sm

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]