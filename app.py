import streamlit as st
import os
from src.utils.parser import ResumeParser
from src.utils.preprocessing import TextPreprocessor
from src.core.ranker import CompositeRanker
from src.services.gemini_api import GeminiService

st.set_page_config(page_title="V2 ATS Ranker", layout='wide')

# initialize Engines (Caching them so they don't reload every click)
@st.cache_resource
def init_engines():
    return ResumeParser(), TextPreprocessor(), CompositeRanker(), GeminiService()

parser, preprocessor, ranker, gemini = init_engines()

st.title("🚀 Professional ATS Resume Ranker")
st.markdown("---")

col1, col2 = st.columns([1,1])

with col1: 
    st.header("Job Description")
    jd_text = st.text_area("Paste Job Description Here", height=300)

with col2:
    st.header("Resume Upload")
    uploaded_file = st.file_uploader("Upload Resume", type=['pdf', 'docx', 'txt'])

if st.button("Analyze Resume"):
    if jd_text and uploaded_file:
        with st.spinner("Analyzing..."):
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # UPDATED: Use the universal dispatcher
            raw_resume = parser.extract_text(temp_path)
            
            # The preprocessor now handles your original regex cleaning!
            clean_resume = " ".join(preprocessor.preprocess(raw_resume))
            clean_jd = " ".join(preprocessor.preprocess(jd_text))
            
            scores = ranker.get_composite_score(clean_resume, clean_jd)

            # 3. Display Scores
            st.metric("Overall Match", f"{scores['total_score']*100:.2f}%")
            
            c1, c2 = st.columns(2)
            c1.progress(scores['semantic_match'], text="Semantic Vibe")
            c2.progress(scores['keyword_match'], text="Hard Skill Match")

            # 4. AI Feedback
            feedback = gemini.generate_feedback(scores, raw_resume, jd_text)
            st.subheader("AI Analysis")
            st.write(feedback)

            # Cleanup
            os.remove(temp_path)
    else:
        st.error("Please provide both a JD and a Resume.")
