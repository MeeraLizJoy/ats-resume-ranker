import spacy
import json
from src.core.ranker import CompositeRanker

class ATSPipeline:
    def __init__(self, model_path="./output/model-last", skills_json="skills_list.json"):
        # 1. Load the "First Model" we refined
        self.nlp = spacy.load(model_path)
        
        # 2. Load the Verified Skills List
        with open(skills_json, "r") as f:
            self.verified_skills = set(json.load(f))
            
        # 3. Initialize your sophisticated Ranker
        self.ranker = CompositeRanker()

    def extract_verified_skills(self, text):
        """Uses the Hybrid approach: AI extracts, JSON verifies."""
        doc = self.nlp(text)
        # Only keep it if it's in our master list (Filter out 'I', '5', etc.)
        return [ent.text for ent in doc.ents if ent.text in self.verified_skills]

    def process_candidate(self, resume_text, jd_text):
        # Step A: Extract clean skills from both texts
        resume_skills = self.extract_verified_skills(resume_text)
        jd_skills = self.extract_verified_skills(jd_text)
        
        # Step B: Pass clean data into your CompositeRanker
        results = self.ranker.get_composite_score(
            resume_text=resume_text,
            jd_text=jd_text,
            resume_skills=resume_skills,
            jd_skills=jd_skills
        )
        
        # Add the extracted skills to the output for transparency
        results["extracted_resume_skills"] = resume_skills
        results["extracted_jd_skills"] = jd_skills
        
        return results

# --- QUICK TEST ---
if __name__ == "__main__":
    pipeline = ATSPipeline()
    
    sample_jd = "Looking for a Senior Developer with Python, AWS, and SQL experience."
    sample_resume = "I am a Senior Developer with 5 years experience in Python. I also know SQL."
    
    final_output = pipeline.process_candidate(sample_resume, sample_jd)
    
    print(f"🎯 Total Score: {final_output['total_score']:.2f}")
    print(f"✅ Verified Skills Found: {final_output['extracted_resume_skills']}")
    print(f"📊 Breakout: Semantic({final_output['semantic_match']:.2f}), Keyword({final_output['keyword_match']:.2f})")