import streamlit as st
import os
import pandas as pd
from src.utils.parser import ResumeParser
from src.core.ranker import CompositeRanker
from src.services.gemini_api import GeminiService
from src.services.extractor import LocalSkillExtractor
from src.services.coach_engine import ResumeCoach
from src.utils.report_gen import generate_pdf_report, generate_chat_txt

st.set_page_config(page_title="ATS AI Command Center", layout='wide')

# --- 1. INITIALIZE SESSION STATE ---
if 'candidate_data' not in st.session_state:
    st.session_state.candidate_data = {}

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'leaderboard' not in st.session_state:
    st.session_state.leaderboard = None

@st.cache_resource
def load_engines():
    """Initializes all AI engines once and caches them."""
    return (
        ResumeParser(), 
        CompositeRanker(), 
        GeminiService(), 
        LocalSkillExtractor(model_path="./output/model-last", skills_json="skills_list.json"),
        ResumeCoach() 
    )

# --- 2. LAZY LOADING LOGIC ---
# This prevents the app from crashing/timing out on first load by showing the UI 
# before the heavy 4GB models are fully initialized in RAM.
if 'engines_ready' not in st.session_state:
    with st.spinner("🚀 Initializing AI Engines... Hugging Face is warming up the models (this may take 60s)."):
        parser, ranker, gemini, extractor, coach = load_engines()
        st.session_state.parser = parser
        st.session_state.ranker = ranker
        st.session_state.gemini = gemini
        st.session_state.extractor = extractor
        st.session_state.coach = coach
        st.session_state.engines_ready = True
else:
    parser = st.session_state.parser
    ranker = st.session_state.ranker
    gemini = st.session_state.gemini
    extractor = st.session_state.extractor
    coach = st.session_state.coach

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ System Status")
    
    # Check status safely to handle Cloud vs Local environments
    try:
        status = coach.get_status()
        ollama_val = status.get('ollama', 'Not Found')
        device_val = status.get('device', 'CPU')
    except Exception:
        ollama_val = "Cloud Mode (Gemini)"
        device_val = "CPU"
    
    st.write(f"**Ollama:** {ollama_val}")
    st.write(f"**Hardware:** {device_val}")
    
    st.divider()
    
    if st.button("🗑️ Clear Local Database"):
        if os.path.exists("./chroma_db"):
            import shutil
            shutil.rmtree("./chroma_db")
            st.success("Database wiped! Refreshing...")
            st.rerun()

# --- MAIN UI ---
st.title("🏆 AI Recruitment Leaderboard")

jd_text = st.text_area("Paste Job Description", height=150, placeholder="Paste the target JD here...")
uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=['pdf'], accept_multiple_files=True)

if st.button("🚀 Rank All Resumes", type="primary", width="stretch"):
    if jd_text and uploaded_files:
        results = []
        with st.spinner("Analyzing Resumes..."):
            # Index the JD into local vector store for RAG
            coach.add_jd_to_index(jd_text)
            
            for file in uploaded_files:
                temp_path = f"temp_{file.name}"
                with open(temp_path, "wb") as f: 
                    f.write(file.getbuffer())
                
                # Extract and Rank
                text = parser.extract_text(temp_path)
                res_skills = extractor.extract_skills(text)
                jd_skills = extractor.extract_skills(jd_text)
                
                scores = ranker.get_composite_score(text, jd_text, res_skills, jd_skills)
                
                # Add to local ChromaDB for Chat retrieval
                coach.add_to_index(text, file.name)
                
                # Store data in session state for the chat interface
                st.session_state.candidate_data[file.name] = {
                    "text": text,
                    "scores": scores
                }
                
                results.append({
                    "Candidate": file.name,
                    "Score": round(scores['total_score']*100, 1),
                    "Skills Match": f"{len(set(res_skills) & set(jd_skills))}/{len(jd_skills)}",
                    "Impact": f"{scores['impact_score']*100:.0f}%",
                })
                os.remove(temp_path)
            
            st.session_state.leaderboard = pd.DataFrame(results).sort_values("Score", ascending=False)
            st.rerun()

# --- RESULTS & ANALYSIS ---
if st.session_state.leaderboard is not None:
    df = st.session_state.leaderboard
    st.subheader("📊 Candidate Rankings")
    st.dataframe(df, use_container_width=True)

    selected_name = st.selectbox("Select Candidate for Analysis", df["Candidate"])
    
    if selected_name in st.session_state.candidate_data:
        data = st.session_state.candidate_data[selected_name]
        
        pdf_bytes = generate_pdf_report(selected_name, data['scores'], jd_text)
        
        st.download_button(
            label=f"📥 Download {selected_name} Analysis (PDF)",
            data=pdf_bytes,
            file_name=f"ATS_Report_{selected_name}.pdf",
            mime="application/pdf"
        )

    # --- CHAT SECTION ---
    st.divider()
    st.header(f"💬 Local Coach: {selected_name}")
    
    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Ask: 'How can I improve this resume for the JD?'")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})
    
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Streaming response from the Coach Engine (Gemini in Cloud / Ollama in Local)
            for chunk in coach.query_stream(prompt, target_filename=selected_name):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})

    if st.session_state.chat_history:
        chat_txt = generate_chat_txt(st.session_state.chat_history)
        st.download_button(
            label="💾 Save Chat Conversation",
            data=chat_txt,
            file_name=f"Coach_Chat_{selected_name}.txt",
            mime="text/plain"
        )
else:
    st.info("Upload resumes and paste a Job Description to start the ranking process.")