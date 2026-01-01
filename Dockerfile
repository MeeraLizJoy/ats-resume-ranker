FROM python:3.11-slim

# Install gcc for spaCy compilation if needed
RUN apt-get update && apt-get install -y gcc g++ wget && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Install spaCy + model first
RUN pip install --no-cache-dir "spacy[cpu]==3.7.6" \
    && python -m spacy download en_core_web_sm

# Then install the rest (including pinned blis)
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 10000
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=10000", "--server.address=0.0.0.0"]