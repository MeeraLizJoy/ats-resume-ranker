# ATS Resume Ranker & AI Suggestion System

## Overview

An end-to-end AI system that evaluates how well a candidate’s resume matches a given job description and helps improve it for Applicant Tracking Systems (ATS). The app performs semantic matching, highlights skills, computes an ATS-style score, and provides interactive AI-powered suggestions and chat-based career guidance via a Streamlit web interface.

---

## Problem Statement

Modern hiring pipelines rely heavily on ATS tools that often focus on shallow keyword matching and struggle to understand context. Candidates and recruiters lack an easy way to:

- Assess how closely a resume aligns with a specific job description.
- Identify missing skills and weak sections.
- Get concrete, tailored suggestions to improve ATS compatibility.

This project addresses these gaps by building an **AI-powered ATS-friendly resume scoring and improvement system** that:

- Parses resumes in multiple formats (PDF, DOCX, TXT).
- Computes semantic similarity between resumes and job descriptions using embeddings.
- Detects present and missing skills and key sections.
- Generates targeted improvement suggestions using generative AI.
- Exposes everything through a clean, interactive Streamlit dashboard.

---

## Key Features

- Multi-format resume upload and text extraction (PDF, DOCX, TXT).
- Semantic similarity scoring between resume and job description using Sentence Transformers.
- ATS composite scoring that combines similarity, keyword density, and formatting quality.
- Skill/keyword highlighting to show strengths and gaps.
- AI-powered improvement suggestions using the Gemini API.
- Simple AI chat interface to ask questions about the resume and JD.
- Highlighted resume view with matched and missing skills.

---

## Project Structure

```
GENAI_GEMINI/
└── ATS/
    ├── app/
    │   ├── components/
    │   │   └── circular_progress.py      # Custom circular score widget
    │   └── streamlit_app.py              # Main Streamlit application
    ├── data/
    │   ├── job_descriptions/             # Sample JDs (optional / gitignored)
    │   ├── resumes/                      # Sample resumes (optional / gitignored)
    │   ├── Resume.csv
    │   ├── Resume_cleaned.csv
    │   └── Resume_preprocessed.csv
    ├── models/
    │   ├── it_software_engineer_embedding.npy
    │   └── resume_embeddings.npy
    ├── notebooks/
    │   └── 01_data_explore.ipynb         # Data exploration (optional)
    ├── scripts/
    │   ├── embed_jd.py                   # Embed JD text
    │   ├── embed.py                      # Embed resumes
    │   ├── filter_by_similarity.py       # Filter resumes by similarity threshold
    │   ├── preprocess.py                 # Clean / normalize raw text
    │   ├── rank.py                       # Ranking / scoring utilities
    │   ├── skill_ner.py                  # Skill extraction with spaCy
    │   ├── suggest.py                    # ATS scoring + AI suggestions
    │   └── utils.py                      # Shared helpers (PDF/DOCX parsing, embeddings, etc.)
    ├── .env                              # API keys & secrets (not committed)
    ├── README.md
    └── requirements.txt
```

---

## Tech Stack & Tools

- **Language & Framework**
  - Python 3.9+
  - Streamlit (web frontend)

- **NLP & Embeddings**
  - Sentence Transformers (`all-MiniLM-L6-v2`) for semantic embeddings
  - spaCy (`en_core_web_sm`) for skill/phrase extraction and basic NER

- **Similarity & Scoring**
  - Scikit-learn (cosine similarity)
  - Custom keyword density and formatting heuristics

- **Generative AI**
  - Google Gemini API for:
    - Extracting key skills from JDs
    - Generating resume improvement suggestions
    - Powering the chat assistant

- **Data & Utilities**
  - Pandas, NumPy
  - BeautifulSoup for text cleaning
  - python-dotenv for environment variable management

---

## Pipeline

1. **Data Input**
   - User uploads a resume (PDF/DOCX/TXT) and a job description via the Streamlit UI.
   - Text is extracted using helper functions (PDF/DOCX parsers or plain text read).

2. **Preprocessing**
   - Clean and normalize text (strip HTML, fix whitespace, lowercase, basic noise removal).
   - (Offline scripts) generate `Resume_cleaned.csv` and `Resume_preprocessed.csv` for batch experiments.

3. **Embedding**
   - Generate embeddings for:
     - The full resume text.
     - The job description text.
   - Use Sentence Transformers to obtain fixed-size semantic vectors.

4. **Similarity & ATS Scoring**
   - Compute cosine similarity between resume and JD embeddings.
   - Extract required skills from the JD and check their presence in the resume.
   - Compute:
     - Similarity score
     - Keyword/skill density
     - Simple formatting score
   - Combine into a composite ATS score.

5. **Skill Extraction & AI Suggestions**
   - Extract skills from the JD using Gemini and/or spaCy.
   - Identify missing skills in the resume.
   - Generate personalized improvement suggestions with Gemini (how to integrate missing skills, improve phrasing, etc.).

6. **Frontend & Interaction**
   - Show:
     - Highlighted resume text with matched and missing skills.
     - Circular progress indicators for similarity, keyword density, formatting, and composite ATS score.
   - Provide a chat tab where the user can ask questions about:
     - How to tailor the resume to the JD.
     - How to present specific experience or skills.

---

## Setup & Usage

1. **Clone the repository**

```
git clone <your-repo-url>
cd ATS
```

2. **Install dependencies**

```
pip install -r requirements.txt
```

3. **Set up environment variables**

Create `scripts/.env` (or `.env` at project root, matching your code):

```
MY_API_KEY=your_gemini_api_key_here
```

4. **Install spaCy model (for NER / skills)**

```
python -m spacy download en_core_web_sm
```

5. **Run the Streamlit app**

```
streamlit run app/streamlit_app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

---

## License

MIT License

---

## Author

- Meera Liz Joy
