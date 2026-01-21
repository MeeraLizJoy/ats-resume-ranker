import streamlit as st
import os
import pandas as pd
import json
from huggingface_hub import hf_hub_download, HfApi
from src.utils.parser import ResumeParser
from src.core.ranker import CompositeRanker
from src.services.gemini_api import GeminiService
from src.services.extractor import LocalSkillExtractor
from src.services.coach_engine import ResumeCoach
from src.utils.report_gen import generate_pdf_report, generate_chat_txt

# --- BRIDGE HF OAUTH TO STREAMLIT ---
# Using 'auth' because st.login() looks for st.secrets["auth"]
try:
    # Check if we need to inject the secrets manually
    if "OAUTH_CLIENT_ID" in os.environ:
        # Check if "auth" already exists safely
        auth_exists = False
        try:
            auth_exists = "auth" in st.secrets
        except:
            auth_exists = False

        if not auth_exists:
            st.secrets.update({
                "auth": {
                    "client_id": os.environ["OAUTH_CLIENT_ID"],
                    "client_secret": os.environ["OAUTH_CLIENT_SECRET"],
                    "server_metadata_url": "https://huggingface.co/.well-known/openid-configuration",
                    "cookie_secret": os.environ.get("OAUTH_COOKIE_SECRET", "at-least-32-characters-long-unique-string"),
                    "redirect_uri": f"https://{os.environ.get('SPACE_ID', '').replace('/', '-')}.hf.space/oauth2callback"
                }
            })
except Exception as e:
    # If secrets can't be initialized, we print to logs but don't crash the whole app
    print(f"Secret Bridge Info: {e}")

# --- 0. CONFIGURATION & DATABASE HELPERS ---
REPO_ID = "meeralizjoy/ats-usage-logs"  # 👈 UPDATE THIS to your repo name
FILE_NAME = "usage.json"

try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    HF_TOKEN = os.getenv("HF_TOKEN") # Fallback to environment variable

if not HF_TOKEN:
    st.warning("⚠️ HF_TOKEN not found in Secrets. Usage tracking will be disabled.")
    api = None
else:
    api = HfApi(token=HF_TOKEN)

def load_usage_data():
    try:
        # Download the usage file from your private dataset
        path = hf_hub_download(repo_id=REPO_ID, filename=FILE_NAME, repo_type="dataset", token=HF_TOKEN)
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}  # Return empty if file doesn't exist yet

def save_usage_data(data):
    with open("usage.json", "w") as f:
        json.dump(data, f)
    api.upload_file(
        path_or_fileobj="usage.json",
        path_in_repo=FILE_NAME,
        repo_id=REPO_ID,
        repo_type="dataset"
    )

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
    return (
        ResumeParser(), 
        CompositeRanker(), 
        GeminiService(), 
        LocalSkillExtractor(model_path="./output/model-last", skills_json="skills_list.json"),
        ResumeCoach() 
    )

if 'engines_ready' not in st.session_state:
    with st.spinner("🚀 Initializing AI Engines..."):
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

# --- 2. AUTHENTICATION LOGIC ---
is_logged_in = st.user.get("is_logged_in", False)
user_email = st.user.get("email") if is_logged_in else None

# Use try-except here because st.user is only available on Streamlit Community Cloud or HF Spaces
try:
    is_logged_in = st.user.get("is_logged_in", False)
    user_email = st.user.get("email") if is_logged_in else None
except Exception:
    is_logged_in = False
    user_email = None

# --- SIDEBAR ---
# --- SIDEBAR ---
with st.sidebar:
    st.header("👤 Account Status")
    if is_logged_in:
        st.success(f"Welcome!")
        usage_db = load_usage_data()
        user_count = usage_db.get(user_email, 0)
        st.write(f"**Lifetime Scans Used:** {user_count} / 2")
        if st.button("Log out"):
            st.logout()
    else:
        st.info("🔓 Log in with Hugging Face to enable ranking.")
        if st.button("Log in"):
            st.login()
    
    st.divider()
    st.header("⚙️ System Status")
    try:
        status = coach.get_status()
        ollama_val = status.get('ollama', 'Not Found')
        device_val = status.get('device', 'CPU')
    except Exception:
        ollama_val = "Cloud Mode (Gemini)"
        device_val = "CPU"
    
    st.write(f"**Ollama:** {ollama_val}")
    st.write(f"**Hardware:** {device_val}")
    
    if st.button("🗑️ Clear Local Database"):
        if os.path.exists("./chroma_db"):
            import shutil
            shutil.rmtree("./chroma_db")
            st.rerun()

# --- MAIN UI ---
st.title("🏆 AI Recruitment Leaderboard")

jd_text = st.text_area("Paste Job Description", height=150, placeholder="Paste the target JD here...")
uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=['pdf'], accept_multiple_files=True)

# Ranking Button Logic with Usage Limit
if st.button("🚀 Rank All Resumes", type="primary", width="stretch"):
    if not is_logged_in:
        st.warning("Please log in to rank your resumes.")
    elif user_count >= 2:
        st.error("🚫 You have reached your lifetime limit of 2 scans.")
    elif jd_text and uploaded_files:
        results = []
        with st.spinner("Analyzing Resumes..."):
            coach.add_jd_to_index(jd_text)
            for file in uploaded_files:
                temp_path = f"temp_{file.name}"
                with open(temp_path, "wb") as f: 
                    f.write(file.getbuffer())
                
                text = parser.extract_text(temp_path)
                res_skills = extractor.extract_skills(text)
                jd_skills = extractor.extract_skills(jd_text)
                scores = ranker.get_composite_score(text, jd_text, res_skills, jd_skills)
                coach.add_to_index(text, file.name)
                
                st.session_state.candidate_data[file.name] = {"text": text, "scores": scores}
                results.append({
                    "Candidate": file.name,
                    "Score": round(scores['total_score']*100, 1),
                    "Skills Match": f"{len(set(res_skills) & set(jd_skills))}/{len(jd_skills)}",
                    "Impact": f"{scores['impact_score']*100:.0f}%",
                })
                os.remove(temp_path)
            
            # Update usage count in Database
            usage_db[user_email] = user_count + 1
            save_usage_data(usage_db)
            
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
        st.download_button(label=f"📥 Download {selected_name} Analysis (PDF)", data=pdf_bytes, file_name=f"ATS_Report.pdf")

    # --- CHAT SECTION ---
    st.divider()
    st.header(f"💬 Local Coach: {selected_name}")
    
    # Career Coach also requires login
    if not is_logged_in:
        st.info("Log in to chat with the AI Career Coach.")
    else:
        for m in st.session_state.chat_history:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        prompt = st.chat_input("Ask: 'How can I improve this resume?'")
        if prompt:
            st.chat_message("user").markdown(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                for chunk in coach.query_stream(prompt, target_filename=selected_name):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
else:
    st.info("Upload resumes and paste a Job Description to start.")