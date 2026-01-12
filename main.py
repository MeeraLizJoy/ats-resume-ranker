import os
from src.utils.parser import ResumeParser
from src.utils.preprocessing import TextPreprocessor
from src.core.ranker import CompositeRanker
from src.services.gemini_api import GeminiService

def main():
    # 1. initialize the Engines
    parser = ResumeParser()
    preprocessor = TextPreprocessor()
    ranker = CompositeRanker()
    gemini = GeminiService()

    print("--- ATS Ranker V2: Professional Model ---")

    # 2. Load Inputs
    resume_path = "ats_ranker_v2/data/sample_resume.pdf"
    jd_path = "ats_ranker_v2/data/job_description.txt"

    if not os.path.exists(resume_path) or not os.path.exists(jd_path):
        print("Error: Please ensure your data/ folder contains your test files.")
        return
    
    # 3. The Pipeline: Parse -> Clean -> Math -> Explain
    print("[1/4] Extracting and Cleaning Text...")
    raw_resume = parser.extract_text_from_pdf(resume_path)
    clean_resume = " ".join(preprocessor.preprocess(raw_resume))

    with open(jd_path, 'r') as f:
        raw_jd = f.read()
    clean_jd = " ".join(preprocessor.preprocess(raw_jd))

    print("[2/4] Claculating Mathematical Scores...")
    scores = ranker.get_composite_score(clean_resume, clean_jd)

    print("\n--- RESULTS ---")
    print(f"Overall Match: {scores['total_score']*100:.2f}%")
    print(f"Semantic Vibe: {scores['semantic_match']*100:.2f}%")
    print(f"Hard Skill Match: {scores['keyword_match']*100:.2f}%")
    print("-------------------\n")

    # 4. Gemini Feedback
    print("[3/4] Generating AI Insights...")
    feedback = gemini.generate_feedback(scores, raw_resume, raw_jd)
    print(f"\nAI ANALYSIS:\n{feedback}")

    # 5. The CVhat Section
    print("\n[4/4] Entering Chat Mode (Type 'exit' to quit)")
    history = []
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break

        response = gemini.chat_with_resume(user_input, raw_resume, history)
        print(f"Gemini: {response}")
        # Append to history for continuous context
        history.append({"role": "user", "parts": [user_input]})
        history.append({"role": "model", "parts": [response]})

if __name__ == "__main__":
    main()
