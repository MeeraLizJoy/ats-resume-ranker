import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings(path):
    return np.load(path)

def rank_resumes(job_embedding, resume_embeddings, resume_ids):
    """
    Compute cosine similarity between job description embedding and resume embeddings.
    Return a sorted list [(resume_id, score), ...] in descending score order.
    """
    sims = cosine_similarity([job_embedding], resume_embeddings)[0]
    ranked_indices = sims.argsort()[::-1]
    ranked_resumes = [(resume_ids[i], sims[i]) for i in ranked_indices]
    return ranked_resumes

def main():
    # Load preprocessed resume metadata and embeddings
    resumes_df = pd.read_csv('data/Resume_preprocessed.csv')
    resume_embeddings = load_embeddings('models/resume_embeddings.npy')

    # Load job description embedding (generated separately)
    job_embedding = load_embeddings('models/it_software_engineer_embedding.npy')

    ranked = rank_resumes(job_embedding, resume_embeddings, resumes_df['ID'].tolist())

    print("Top 10 ranked resumes for the job description:")
    for rid, score in ranked[:10]:
        print(f"Resume ID: {rid}, Similarity Score: {score:.4f}")

if __name__ == "__main__":
    main()
