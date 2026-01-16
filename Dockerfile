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

# 6. NOW copy the rest of your project code 
# This MUST happen before running download_models.py
COPY . .

# 7. Run the download script to bake the AI models into the image
RUN python download_models.py

# 8. Download the spaCy model
RUN python -m spacy download en_core_web_sm

# 9. Expose the port Streamlit runs on
EXPOSE 8501

# 10. Define how to run the app
# Using CMD instead of ENTRYPOINT is often more flexible on Render
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]