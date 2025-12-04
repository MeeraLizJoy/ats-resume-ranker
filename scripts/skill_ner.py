import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")

def build_skill_matcher(skills):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp(skill.strip()) for skill in skills]
    matcher.add("SKILLS", patterns)
    return matcher

def extract_skills_phrase_match(resume_text, skills_list):
    """
    Extract skills from resume text by matching phrases from skills_list (comma-separated).
    """
    skills = [s.strip().lower() for s in skills_list.split(",")]
    matcher = build_skill_matcher(skills)

    doc = nlp(resume_text.lower())
    matches = matcher(doc)
    matched_skills = set()
    for match_id, start, end in matches:
        span = doc[start:end]
        matched_skills.add(span.text.title())  # Title case for display

    return list(matched_skills)


def extract_missing_skills_ner(resume_text, jd_skills_text):
    """
    Using phrase matcher for extracting skills and finding missing ones from resume.
    """
    jd_skills = set([s.strip().title() for s in jd_skills_text.split(",")])
    resume_skills = set(extract_skills_phrase_match(resume_text, jd_skills_text))
    missing_skills = jd_skills - resume_skills
    return list(missing_skills)


# # For testing standalone
# if __name__ == "__main__":
#     jd_text = "JavaScript, Python, React, Machine Learning, NLP, AWS, GCP, Docker"
#     resume = "Experienced developer skilled in React and python scripting deployed on aws cloud."
#     print("Extracted Skills:", extract_skills_phrase_match(resume, jd_text))
#     print("Missing Skills:", extract_missing_skills_ner(resume, jd_text))
