import numpy as np
from src.core.embeddings import EmbeddingEngine
from src.core.stats import StatisticalAnalyzer

class CompositeRanker:
    # Change default weights here for a more stable score
    def __init__(self, semantic_weight=0.5, keyword_weight=0.3, impact_weight=0.2):
        self.embed_engine = EmbeddingEngine()
        self.stats_engine = StatisticalAnalyzer()
        self.w1 = semantic_weight # 50%
        self.w2 = keyword_weight # 30% 
        self.w3 = impact_weight # 20%

    def get_semantic_match(self, resume_text, jd_text):
        """
        FIXED: Now correctly calls calculate_similarity to match 
        your EmbeddingEngine class.
        """
        return self.embed_engine.calculate_similarity(resume_text, jd_text)

    def get_keyword_match(self, resume_skills, jd_skills):
        """
        Calculates hard-skill overlap with a Boosted Ratio.
        """
        if not jd_skills:
            return 1.0
        
        # Clean skill sets
        res_set = set([s.lower() for s in resume_skills]) if resume_skills else set()
        jd_set = set([s.lower() for s in jd_skills]) if jd_skills else set()
        
        intersection = res_set & jd_set
        
        # Avoid division by zero
        if len(jd_set) == 0: return 1.0
        
        raw_ratio = len(intersection) / len(jd_set)
        
        # Boosted score (np.sqrt) pushes scores toward the 85-95% range
        boosted_score = np.sqrt(raw_ratio)
        return min(boosted_score, 1.0)

    def get_composite_score(self, resume_text, jd_text, resume_skills=None, jd_skills=None):
        """
        Final score calculation integrating context, keywords, and impact metrics.
        """
        # 1. Semantic Vibe
        semantic_score = self.get_semantic_match(resume_text, jd_text)
        
        # 2. Keyword Accuracy (using boosted NER results)
        keyword_score = self.get_keyword_match(resume_skills, jd_skills)
    
        # 3. Quantifiable Impact
        _, impact_score = self.stats_engine.detect_metrics(resume_text)

        # Weighting logic
        total_score = (semantic_score * self.w1) + (keyword_score * self.w2) + (impact_score * self.w3)
    
        return {
            "total_score": total_score,
            "semantic_match": semantic_score,
            "keyword_match": keyword_score,
            "impact_score": impact_score
        }