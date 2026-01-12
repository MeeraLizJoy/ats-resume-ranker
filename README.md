# 🚀 AI-Powered ATS Resume Ranker (V2)

A professional-grade Applicant Tracking System (ATS) that leverages **Statistical NLP** and **Semantic AI** to rank resumes against job descriptions with high precision.

**[Live Demo Link: Paste your Render URL here]**

---

## 🌟 The V2 Evolution

While V1 relied on basic keyword matching, V2 introduces a **Hybrid Scoring Engine** and **Universal Document Parsing**, ensuring that context—not just keywords—is captured.

### 🧠 Key Features

* **Universal Parser:** Seamlessly extracts text from `.pdf`, `.docx`, and `.txt` formats using `pdfplumber` and `python-docx`.
* **Hybrid Ranking Engine:** Combines **TF-IDF (Statistical)** and **Cosine Similarity (Semantic)** via `sentence-transformers` for a balanced match score.
* **Contextual AI Feedback:** Integrated with **Google Gemini 1.5 Flash** to provide actionable insights on *why* a candidate is or isn't a fit.
* **Modular Architecture:** Built with a clean separation of concerns: `Parser` ➔ `Preprocessor` ➔ `Ranker` ➔ `GeminiService`.

---

## 🛠️ Technical Stack

* **Frontend:** Streamlit
* **Extraction:** `pdfplumber`, `python-docx`
* **NLP & Math:** `nltk`, `scikit-learn`, `numpy`, `PyTorch`
* **Embeddings:** `all-MiniLM-L6-v2` (SBERT)
* **LLM:** Google Gemini 1.5 Flash API
* **Deployment:** Render

---

## 🏗️ System Architecture

1. **Extraction Layer:** Detects file type and converts binary data to raw text.
2. **Preprocessing Layer:** Normalizes whitespace, filters non-ASCII noise, removes stopwords, and performs **Lemmatization**.
3. **Scoring Layer:**
* **Hard Skill Match:** Uses TF-IDF to find exact keyword overlaps.
* **Semantic Vibe:** Uses Vector Embeddings to understand conceptual alignment.


4. **AI Layer:** Injects scores and text into Gemini to generate a professional critique and "Chat with Resume" capability.

---

## 🚀 Getting Started

### 1. Installation

```bash
git clone https://github.com/MeeraLizJoy/ats-resume-ranker.git
cd ats-resume-ranker
pip install -r requirements.txt

```

### 2. Environment Setup

Create a `.env` file in the root directory:

```bash
GOOGLE_API_KEY=your_gemini_api_key_here

```

### 3. Run Locally

```bash
streamlit run app.py

```

---

## 📈 Future Roadmap

* [ ] **Multi-resume comparison:** Bulk upload with leaderboard visualization.
* [ ] **Domain Embeddings:** Fine-tuned models for specialized Healthcare or Law roles.
* [ ] **Career Path Prediction:** Suggesting roles based on historical candidate data.

---

## 📄 License & Contact

Distributed under the MIT License.

**Author:** Meera Liz Joy

**Email:** [meeraliz2003@gmail.com](mailto:meeraliz2003@gmail.com)

**LinkedIn:** [www.linkedin.com/in/meeralizjoy]

---

