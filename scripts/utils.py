# scripts/utils.py

import os
import re
import numpy as np
import pandas as pd
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import google.generativeai as genai
from dotenv import load_dotenv

# Load .env and configure Gemini API key
load_dotenv(os.path.join("scripts", ".env"))
genai.configure(api_key=os.getenv("MY_API_KEY"))


# --------- Text Extraction Helpers ---------
def pdf_to_text(pdf_file):
    import PyPDF2
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + '\n'
    return text


def docx_to_text(docx_file):
    doc = Document(docx_file)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)


def clean_html(raw_text):
    if pd.isna(raw_text):
        return ""
    return BeautifulSoup(str(raw_text), "html.parser").get_text(separator=" ")


def normalize_text(text):
    if pd.isna(text):
        return ""
    text = text.replace('\u2028', ' ').replace('\u2029', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


# --------- Embedding Helpers ---------
def get_embedding_model(model_name='all-MiniLM-L6-v2'):
    return SentenceTransformer(model_name)


def generate_embeddings(texts, model):
    return model.encode(texts, batch_size=32, show_progress_bar=True)


def compute_cosine_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]


def load_embeddings_from_file(file_path):
    return np.load(file_path)


# --------- Skills Utilities ---------
def highlight_skills(text, matched_skills, missing_skills):
    for skill in missing_skills:
        pattern = re.compile(rf'\b{re.escape(skill)}\b', flags=re.IGNORECASE)
        text = pattern.sub(f"<span style='background-color:#faa; font-weight:bold;'>{skill}</span>", text)
    for skill in matched_skills:
        pattern = re.compile(rf'\b{re.escape(skill)}\b', flags=re.IGNORECASE)
        text = pattern.sub(f"<span style='background-color:#afa; font-weight:bold;'>{skill}</span>", text)
    return text


# --------- Gemini AI Calls ---------
def generate_content_from_gemini(prompt, model_name='models/gemini-flash-latest'):
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text.strip() if hasattr(response, "text") else str(response)

# Additional utility wrappers for skill extraction, improvements, ranking can be added accordingly.
