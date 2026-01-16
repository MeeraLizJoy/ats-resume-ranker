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

# 4. Copy only requirements first (to use Docker caching)
COPY requirements.txt .

# 5. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the download script to bake the model into the image
RUN python download_model.py

# 6. Download the base spaCy model (used as a fallback)
RUN python -m spacy download en_core_web_sm

# 7. Copy the rest of your project code
COPY . .

# 8. Expose the port Streamlit runs on
EXPOSE 8501

# 9. Define how to run the app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]