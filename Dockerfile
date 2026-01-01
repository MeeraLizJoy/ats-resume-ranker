FROM python:3.11-slim

# Install gcc for spacy compilation
RUN apt-get update && apt-get install -y gcc g++ wget && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Install FULL requirements + spacy model
RUN pip install --no-cache-dir spacy[cpu]==3.7.6 \
    && python -m spacy download en_core_web_sm \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 10000
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=10000", "--server.address=0.0.0.0"]