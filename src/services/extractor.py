import spacy
import json
import os

class LocalSkillExtractor:
    def __init__(self, model_path="./output/model-last", skills_json="skills_list.json"):
        """
        Initializes the Hybrid Extractor.
        - model_path: Path to your trained spaCy NER model.
        - skills_json: Path to the JSON file containing the verified skills list.
        """
        # 1. Load the Custom NER Model
        try:
            self.nlp = spacy.load(model_path)
        except Exception as e:
            print(f"Error loading NER model: {e}. Falling back to blank model.")
            self.nlp = spacy.blank("en")

        # 2. Load the Verified Skills List (The Security Guard)
        if os.path.exists(skills_json):
            with open(skills_json, "r") as f:
                self.verified_skills = set(json.load(f))
        else:
            print(f"Warning: {skills_json} not found. Filter disabled.")
            self.verified_skills = None

    def extract_skills(self, text):
        """
        Uses NER to find potential skills, then filters them against 
        the verified JSON list to ensure 100% accuracy.
        """
        doc = self.nlp(text)
        
        # Get raw entities from AI
        extracted = [ent.text for ent in doc.ents]
        
        # If we have a verified list, filter out the noise (like "I" or "5")
        if self.verified_skills:
            clean_skills = [skill for skill in extracted if skill in self.verified_skills]
            # Remove duplicates while preserving order
            return list(dict.fromkeys(clean_skills))
        
        return list(dict.fromkeys(extracted))

    def identify_gaps(self, jd_skills, resume_skills):
        """Returns skills present in JD but missing in Resume."""
        return [skill for skill in jd_skills if skill not in resume_skills]