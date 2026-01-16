import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer

class StatisticalAnalyzer:
    """
    Module for mathematical keyword weighting and Information Entropy.
    """
    def __init__(self):
        # We use sublinear_tf to scale word counts logarithmatically.
        # This prevents a resume with 'Python' written 50 times from
        # unfairly dominating a resume with 'Python' written 5 times.
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            sublinear_tf=True
        )

    def extract_top_keywords(self, texts, top_n = 20):
        """
        Use TF-IDF to identify the most 'Information-Rich' terms in a corpus.
        """
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()

        # Calculate mean TF-IDF across all documents to find global importance
        importance = np.asarray(tfidf_matrix.mean(axis = 0)).ravel().tolist()
        keyword_importance = sorted(
            zip(feature_names, importance),
            key = lambda x: x[1],
            reverse = True
        )

    def calculate_weighted_overlap(self, resume_text, jd_text):
        """
        Calculates a similarity score weighted by word importance (IDF).
        This is much more accurate than simple keyword counting.
        """
        vectors = self.vectorizer.fit_transform([jd_text, resume_text])

        # The First vector is out 'Ideal' (the Job DEscription)
        # The second vector is our 'Subject' (the Resume)
        jd_vector = vectors.toarray()[0]
        resume_vector = vectors.toarray()[1]

        # Calculate overlap only for terms that actually matter in the JD
        important_indices = np.where(jd_vector > 0)[0]

        # Math: Sum of (Resume_Weight * JD_Weight) for important terms
        score = np.dot(resume_vector[important_indices], jd_vector[important_indices]) 

        return float(score)
    
    def detect_metrics(self, text):
        """
        NEW: Detects quantifiable impact (metrics) in text.
        Returns a list of found metrics and a normalized 'Impact Score'.
        """
        patterns = [
            r'\d+%',                                     # Percentages (e.g., 20%)
            r'\$\d+(?:,\d{3})*(?:\.\d+)?(?:[kKmMbB])?',  # Currency (e.g., $50k, $1,000)
            r'\b\d{1,3}(?:,\d{3})+\b',                   # Large numbers (e.g., 1,200)
            r'(?:increased|reduced|saved|improved)\s+\w+\s+by\s+\d+' # Impact phrases
        ]
        
        found_metrics = []
        for p in patterns:
            found_metrics.extend(re.findall(p, text, re.IGNORECASE))
        
        # Calculate impact score (0.0 to 1.0) based on how many metrics are found
        # 5+ metrics is considered a 'High Impact' resume
        impact_score = min(len(found_metrics) / 5, 1.0)
        
        return found_metrics, impact_score
            
        
    