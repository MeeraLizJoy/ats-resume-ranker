import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


def generate_embeddings(input_csv, embeddings_path, model_name='all-MiniLM-L6-v2'):
    # Load preprocessed resumes
    df = pd.read_csv(input_csv, encoding='utf-8')
    
    if 'preprocessed_resume' not in df.columns:
        raise ValueError("Input CSV must contain 'preprocessed_resume' column")
    
    texts = df['preprocessed_resume'].astype(str).tolist()

    print(f"Generating embeddings for {len(texts)} resumes using model: {model_name}")
    
    model = SentenceTransformer(model_name)

    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    
    np.save(embeddings_path, embeddings)
    
    print(f"Embeddings saved to {embeddings_path}")


if __name__ == "__main__":
    input_file = 'data/Resume_preprocessed.csv' 
    output_file = 'models/resume_embeddings.npy'     
    generate_embeddings(input_file, output_file)



# How to use

# Run this script after preprocessing resumes.

# It creates a NumPy file with all resume embeddings, which you can later load for similarity calculations.

# You can swap model_name for other SentenceTransformer models or adapt for Gemini API if needed.