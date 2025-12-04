import streamlit as st
from io import StringIO
import os
from dotenv import load_dotenv

from scripts.utils import (
    pdf_to_text,
    docx_to_text,
    highlight_skills,
    get_embedding_model,
    generate_content_from_gemini,
    compute_cosine_similarity,
)

from scripts.suggest import (
    extract_skills_from_jd,
    extract_missing_skills_ner,
    compute_composite_score,
    compute_keyword_density,
    compute_formatting_score,
    generate_improvement_suggestion,
)

from components.circular_progress import st_circular_progress

load_dotenv(os.path.join("scripts", ".env"))
model = get_embedding_model()

st.title("ATS Resume Ranker and Career Advisor")

# Resume upload with change detection and chat clearing
resume_file = st.file_uploader("Upload your Resume (TXT, PDF, DOCX)", type=['txt', 'pdf', 'docx'])
if resume_file:
    if resume_file.type == 'application/pdf':
        new_resume_text = pdf_to_text(resume_file)
    elif resume_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        new_resume_text = docx_to_text(resume_file)
    else:
        new_resume_text = StringIO(resume_file.getvalue().decode("utf-8")).read()

    if st.session_state.get('last_resume_text', None) != new_resume_text:
        st.session_state.chat_history = []
        st.session_state['last_resume_text'] = new_resume_text
        st.session_state['resume_text_area'] = new_resume_text
    else:
        new_resume_text = st.session_state.get('resume_text_area', new_resume_text)

    resume_text_area = st.text_area("Edit Resume Text", value=new_resume_text, height=300)
    st.session_state['resume_text_area'] = resume_text_area
else:
    resume_text_area = ""

# Job description upload with change detection and chat clearing
jd_file = st.file_uploader("Upload Job Description (TXT, PDF, DOCX)", type=['txt', 'pdf', 'docx'])
if jd_file:
    if jd_file.type == 'application/pdf':
        new_jd_text = pdf_to_text(jd_file)
    elif jd_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        new_jd_text = docx_to_text(jd_file)
    else:
        new_jd_text = StringIO(jd_file.getvalue().decode("utf-8")).read()

    if st.session_state.get('last_jd_text', None) != new_jd_text:
        st.session_state.chat_history = []
        st.session_state['last_jd_text'] = new_jd_text
    else:
        new_jd_text = st.session_state.get('jd_text', new_jd_text)

    st.subheader("Job Description Preview")
    st.text_area("Job Description Text", value=new_jd_text, height=200, disabled=True)
    st.session_state['jd_text'] = new_jd_text
else:
    new_jd_text = ""

st.markdown("<hr>", unsafe_allow_html=True)

with st.expander("About ATS Scores and Suggestions 📊"):
    st.markdown("""
    - **Similarity**: How semantically close your resume is to the job description.
    - **Keyword Density**: Coverage of key skills extracted from the job description.
    - **Formatting**: Resume formatting friendliness for ATS parsing.
    - **ATS Score**: Composite score combining all factors.
    """)

tab1, tab2 = st.tabs(["Scores & Suggestions", "Chat with AI"])

with tab1:
    if st.button("Analyze Resume"):
        if not resume_text_area or not new_jd_text:
            st.error("Please upload both resume and job description files.")
        else:
            with st.spinner("Extracting skills and computing scores..."):
                jd_skills_text = extract_skills_from_jd(new_jd_text)
                required_skills = [s.strip() for s in jd_skills_text.split(",") if s.strip()]
                missing_skills = extract_missing_skills_ner(resume_text_area, jd_skills_text)
                matched_skills = list(set(required_skills) - set(missing_skills))

                resume_embedding = model.encode([resume_text_area])[0]
                jd_embedding = model.encode([new_jd_text])[0]
                similarity_score = compute_cosine_similarity(resume_embedding, jd_embedding)

                keyword_density = compute_keyword_density(resume_text_area, required_skills)
                formatting_score = compute_formatting_score(resume_text_area)
                composite_score = compute_composite_score(similarity_score, resume_text_area, required_skills)

            highlighted_resume = highlight_skills(resume_text_area, matched_skills, missing_skills)
            st.markdown("### Resume Text with Highlighted Skills")
            st.markdown(f"<div style='white-space: pre-wrap; font-family: monospace;'>{highlighted_resume}</div>", unsafe_allow_html=True)

            st.markdown("<h2>ATS Score Breakdown</h2>", unsafe_allow_html=True)
            cols = st.columns(4)
            scores = [
                (similarity_score*100, "#00e0ff", "Similarity"),
                (keyword_density*100, "#33d17a", "Keyword Density"),
                (formatting_score*100, "#ffae00", "Formatting"),
                (composite_score*100, "#7757ec", "Composite Score"),
            ]
            for col, (score, color, label) in zip(cols, scores):
                with col:
                    st_circular_progress(score, color, label)

            st.markdown("<hr>", unsafe_allow_html=True)

            if missing_skills:
                st.markdown(f"### Missing Skills ({len(missing_skills)}):")
                st.write(", ".join(missing_skills))

                with st.spinner("Generating personalized AI suggestions..."):
                    suggestions = generate_improvement_suggestion(resume_text_area, missing_skills)

                st.markdown("### AI-Powered Improvement Suggestions")
                st.markdown(suggestions)

                st.download_button(
                    label="Download Improvement Suggestions",
                    data=suggestions,
                    file_name="resume_improvement_suggestions.txt",
                    mime="text/plain",
                )
            else:
                st.success("No missing skills detected! Resume looks good based on provided job description.")

with tab2:
    st.header("Ask AI about your Resume & Job Description")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Only allow chat if both resume & JD exist
    if resume_text_area and new_jd_text:
        user_msg = st.text_input("Enter your question", key="chat_input")
        if user_msg:
            st.session_state.chat_history.append({"role": "user", "content": user_msg})

            with st.spinner("AI is thinking... generating response."):
                prompt = (
                    f"Resume:\n{resume_text_area}\n\nJob Description:\n{new_jd_text}\n\nQuestion:\n{user_msg}\n"
                )
                ai_response = generate_content_from_gemini(prompt)
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
    else:
        st.info("Please upload both a resume and a job description to start chatting with AI.")

    for chat in st.session_state.chat_history:
        role = "You" if chat["role"] == "user" else "AI"
        st.markdown(f"**{role}:** {chat['content']}")
