import pandas as pd
import re
from bs4 import BeautifulSoup


def clean_html(raw_text):
    """Remove any HTML tags from text."""
    if pd.isna(raw_text):
        return ""
    return BeautifulSoup(str(raw_text), "html.parser").get_text(separator=" ")

def normalize_text(text):
    """Lowercase, strip extra whitespace, and fix basic unicode."""
    if pd.isna(text):
        return ""
    # Remove remaining unicode controls, excessive whitespace
    text = text.replace('\u2028', ' ').replace('\u2029', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def preprocess_resume_file(input_path, output_path):
    df = pd.read_csv(input_path, encoding='utf-8')
    print(f"Loaded {len(df)} resumes from {input_path}")
    # Combine additional cleaning as needed
    df['preprocessed_resume'] = df['Resume_str'].apply(clean_html).apply(normalize_text)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Preprocessing complete. Saved to: {output_path}")


if __name__ == "__main__":
    preprocess_resume_file('data/Resume_cleaned.csv', 'data/Resume_preprocessed.csv')
        
        