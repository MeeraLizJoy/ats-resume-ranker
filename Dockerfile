FROM python:3.11-slim
RUN apt-get update && apt-get install -y gcc g++ wget && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE $PORT
CMD streamlit run app/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true