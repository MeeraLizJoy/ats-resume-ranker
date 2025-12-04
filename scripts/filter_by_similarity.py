import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def filter_resumes_by_similarity(job_embedding_path, resume_embeddings_path, resumes_csv_path, threshold=0.6):
    job_embedding = np.load(job_embedding_path)  # Shape (embedding_dim,)
    resume_embeddings = np.load(resume_embeddings_path)  # Shape (num_resumes, embedding_dim)
    resumes_df = pd.read_csv(resumes_csv_path)

    sims = cosine_similarity([job_embedding], resume_embeddings)[0]
    filtered_indices = [i for i, score in enumerate(sims) if score >= threshold]

    filtered_resume_ids = resumes_df.iloc[filtered_indices]['ID'].tolist()
    print(f"Filtered {len(filtered_resume_ids)} resumes above threshold {threshold}")
    return filtered_resume_ids

# Usage
filtered_ids = filter_resumes_by_similarity(
    'models/it_software_engineer_embedding.npy',
    'models/resume_embeddings.npy',
    'data/Resume_preprocessed.csv',
    threshold=0.6
)
