import numpy as np
from src.core.embeddings import EmbeddingEngine
from src.core.stats import StatisticalAnalyzer

class CompositeRanker:
    def __init__(self, semantic_weight = 0.7, keyword_weight = 0.3):
        self.embed_engine = EmbeddingEngine()
        self.stats_engine = StatisticalAnalyzer()
        self.w1 = semantic_weight
        self.w2 = keyword_weight

    def get_composite_score(self, resume_text, jd_text):
        """
        Calculates a hybrid score that is more robust than a simple similarity. 
        """
        # 1. Semantic Score (Geometry-based)
        # Higher score = the 'vibe' and experience level match
        semantic_score = self.embed_engine.calculate_similarity(resume_text, jd_text)

        # 2. Keyword Score (Statistical-based)
        # Higher score = the specific required tools/ tech are present
        keyword_score = self.stats_engine.calculate_weighted_overlap(resume_text, jd_text)

        # 3. Final Hybrid Calculation
        # Normalized between 0 and 1
        final_score = (self.w1 * semantic_score) + (self.w2 * keyword_score)

        return {
            "total_score" : round(final_score, 4),
            "semantic_match" : round(semantic_score, 4),
            "keyword_match" : round(keyword_score, 4)
        }
        
    def rank_multiple_resumes(self, resume_list, jd_text):
        """
        Takes a list of preprocessed resumes and ranks them.
        """
        ranked_results = []
        for resume in resume_list:
            scores = self.get_composite_score(resume['text'], jd_text)
            ranked_results.append({
                "id": resume['id'],
                **scores
            })

        # Sort by total_score descending
        return sorted(ranked_results, key = lambda x: x['total_score'], reverse = True)
        