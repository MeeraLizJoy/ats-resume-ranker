import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingEngine:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the Transformer model.
        'all-MiniLM-L6-v2' is fast, balanced model mapping text to 38f dimentions.
        """
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, text: str):
        """
        Converts text into a numerical vector (Embedding).
        """
        return self.model.encode(text)

    def calculate_similarity(self, resume_text: str, jd_text: str) -> float:
        """
        The mathematical core: Calculates the Cosine Similarity between two vectors.
        """
        # 1. Generate Vectors
        resume_vector = self.get_embeddings(resume_text).reshape(1, -1)
        jd_vector = self.get_embeddings(jd_text).reshape(1, -1)

        # 2. Compute Cosine Similarity
        # Formula: cos(theta) = (A . B) / (||A|| * ||B||)
        similarity_score = cosine_similarity(resume_vector, jd_vector)[0][0]

        return float(similarity_score)
        
        
        