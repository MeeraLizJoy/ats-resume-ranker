import os
import re
import pandas as pd
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

from scripts.skill_ner import extract_missing_skills_ner  # phrase matcher skill extraction

load_dotenv()
genai.configure(api_key=os.getenv("MY_API_KEY"))


def filter_resumes_by_similarity(job_embedding_path, resume_embeddings_path, resumes_csv_path, threshold=0.6):
    job_embedding = np.load(job_embedding_path)
    resume_embeddings = np.load(resume_embeddings_path)
    resumes_df = pd.read_csv(resumes_csv_path)

    sims = cosine_similarity([job_embedding], resume_embeddings)[0]
    filtered_indices = [i for i, score in enumerate(sims) if score >= threshold]

    filtered_resume_ids = resumes_df.iloc[filtered_indices]['ID'].tolist()
    print(f"Filtered {len(filtered_resume_ids)} resumes above threshold {threshold}")
    return filtered_resume_ids, sims


def extract_skills_from_jd(jd_text):
    try:
        model = genai.GenerativeModel('models/gemini-flash-latest')
        prompt = f"""
        You are an expert at extracting skills from job descriptions.
        Extract the key skills from the following job description and return them as a comma-separated list.
        Do not include any additional text or explanations.

        Job Description:
        {jd_text}

        Skills:
        """
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error extracting skills: {e}")
        return ""


def compute_keyword_density(resume_text, required_skills):
    words = re.findall(r'\w+', resume_text.lower())
    total_words = len(words)
    skill_count = 0
    for skill in required_skills:
        skill_count += resume_text.lower().count(skill.lower())
    if total_words == 0:
        return 0.0
    return skill_count / total_words


def compute_formatting_score(resume_text):
    score = 0.0
    bullets = len(re.findall(r'^\s*[-*+]', resume_text, re.MULTILINE))
    if bullets >= 3:
        score += 0.3
    headers = re.findall(r'^(professional summary|skills|experience|education):', resume_text.lower(), re.MULTILINE)
    if headers:
        score += 0.3
    newlines = resume_text.count('\n')
    if newlines > 10:
        score += 0.4
    return min(score, 1.0)


def compute_composite_score(cosine_sim, resume_text, required_skills):
    kd = compute_keyword_density(resume_text, required_skills)
    fs = compute_formatting_score(resume_text)
    composite = 0.6 * cosine_sim + 0.25 * kd + 0.15 * fs
    return composite


def generate_improvement_suggestion(original_text, missing_skills):
    try:
        model = genai.GenerativeModel('models/gemini-flash-latest')
        prompt = (
            f"The following resume text is missing these important skills: {', '.join(missing_skills)}. "
            "Provide comprehensive, context-aware improvement advice including:\n"
            "1. How to naturally incorporate these missing skills into the resume text.\n"
            "2. Recommendations on keyword placement to optimize ATS scanning.\n"
            "3. Formatting tips to improve visual hierarchy and readability.\n"
            "4. Suggestions to better highlight achievements and impact.\n"
            f"\nResume Text:\n{original_text}\n\nDetailed Suggestions:"
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating suggestions: {e}")
        return ""


def main():
    print("Loading job description text for skill extraction...")
    with open('data/job_descriptions/it_software_engineer.txt', 'r', encoding='utf-8') as f:
        jd_text = f.read()

    required_skills_text = extract_skills_from_jd(jd_text)
    print("Extracted required skills (text):")
    print(required_skills_text)

    required_skills = [s.strip() for s in required_skills_text.split(",")]

    filtered_ids, similarities = filter_resumes_by_similarity(
        'models/it_software_engineer_embedding.npy',
        'models/resume_embeddings.npy',
        'data/Resume_preprocessed.csv',
        threshold=0.6
    )

    df = pd.read_csv('data/Resume_preprocessed.csv')
    df_filtered = df[df['ID'].isin(filtered_ids)]

    id_to_sim = dict(zip(df_filtered['ID'], [similarities[i] for i in df_filtered.index]))

    for idx, row in df_filtered.iterrows():
        original_resume = row['preprocessed_resume']
        missing = extract_missing_skills_ner(original_resume, required_skills_text)
        sim_score = id_to_sim[row['ID']]
        composite_score = compute_composite_score(sim_score, original_resume, required_skills)

        print(f"Resume ID {row['ID']}")
        print(f"Similarity Score: {sim_score:.3f}")
        print(f"Keyword Density: {compute_keyword_density(original_resume, required_skills):.3f}")
        print(f"Formatting Score: {compute_formatting_score(original_resume):.3f}")
        print(f"Composite ATS Score: {composite_score:.3f}")
        print(f"Missing Skills: {missing}")

        if missing:
            suggestion = generate_improvement_suggestion(original_resume, missing)
            print("AI Suggested Improvement:\n", suggestion)
        else:
            print("Resume contains all required skills.\n")
        print("=" * 80)


if __name__ == "__main__":
    main()
