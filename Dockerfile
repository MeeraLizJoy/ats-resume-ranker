# 1. Use a lightweight Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy ONLY requirements first (for better caching)
COPY requirements.txt .

# 5. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the entire project into the container
# We do this NOW so that download_models.py has access to the folder structure
COPY . .

# 7. Ensure the model output directory exists to prevent spacy [E050]
RUN mkdir -p output/model-last

# 8. Run the download script to bake the AI models into the image
# This script will now create the files inside the existing output directory
RUN python download_models.py

# 9. Download the standard spaCy model as a fallback
RUN python -m spacy download en_core_web_sm

# 10. Hugging Face Spaces uses port 7860 by default
EXPOSE 7860

# Final stable command for Hugging Face
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false", "--server.enableCORS=false"]