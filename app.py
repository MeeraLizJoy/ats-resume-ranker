import streamlit as st
import os
from src.utils.parser import ResumeParser
from src.utils.preprocessing import TextPreprocessor
from src.core.ranker import CompositeRanker
from src.services.gemini_api import GeminiService

st.set_page_config(page_title="V2 ATS Ranker", layout='wide')

# --- INITIALIZE SESSION STATE ---
# We need these to persist data across reruns (like when chatting)
if "total_input_tokens" not in st.session_state:
    st.session_state.total_input_tokens = 0
if "total_output_tokens" not in st.session_state:
    st.session_state.total_output_tokens = 0
if "messages" not in st.session_state:
    st.session_state.messages = []
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# --- SIDEBAR SECTION ---
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("Manage your app session and monitor usage.")
    
    st.divider()
    st.header("📊 Usage Monitor")
    
    total_tokens = st.session_state.total_input_tokens + st.session_state.total_output_tokens
    st.metric("Total Tokens", f"{total_tokens:,}")
    
    col_in, col_out = st.columns(2)
    col_in.caption(f"📥 In: {st.session_state.total_input_tokens}")
    col_out.caption(f"📤 Out: {st.session_state.total_output_tokens}")

    if st.button("Reset Counter", use_container_width=True):
        st.session_state.total_input_tokens = 0
        st.session_state.total_output_tokens = 0
        st.rerun()

    st.divider()
    
    if st.button("Clear AI Cache", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache Cleared!")

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.success("Chat Reset!")
        st.rerun()

    st.divider()
    st.header("📄 Export")
    
    if st.session_state.analysis_results:
        # 1. Prepare the Report Content
        res = st.session_state.analysis_results
        report_text = f"""# ATS ANALYSIS REPORT
        
        ## SCORES
        - Overall Match: {res['scores']['total_score']*100:.2f}%
        - Semantic Match: {res['scores']['semantic_match']*100:.2f}%
        - Keyword Match: {res['scores']['keyword_match']*100:.2f}%

        ## AI SUGGESTIONS
        {res['feedback']}

        ---
        ## CHAT HISTORY
        """
        # Append Chat History
        if st.session_state.messages:
            for msg in st.session_state.messages:
                role = "User" if msg["role"] == "user" else "AI"
                report_text += f"\n**{role}**: {msg['content']}\n"
        else:
            report_text += "\n(No chat history available)"

        # 2. Create Download Button
        st.download_button(
            label="Download Full Report (.md)",
            data=report_text,
            file_name="ATS_Resume_Analysis.md",
            mime="text/markdown",
            use_container_width=True
        )
    else:
        st.info("Analyze a resume to enable download.")

    st.divider()
    st.info("Model: gemini-3-flash-preview")

# --- ENGINES ---
@st.cache_resource
def init_engines():
    return ResumeParser(), TextPreprocessor(), CompositeRanker(), GeminiService(model_name='gemini-3-flash-preview')

parser, preprocessor, ranker, gemini = init_engines()

# --- MAIN UI ---
st.title("🚀 Professional ATS Resume Ranker")
st.markdown("---")

col1, col2 = st.columns([1,1])

with col1: 
    st.header("Job Description")
    jd_text = st.text_area("Paste Job Description Here", height=300)

with col2:
    st.header("Resume Upload")
    uploaded_file = st.file_uploader("Upload Resume", type=['pdf', 'docx', 'txt'])

@st.cache_data(show_spinner="AI is analyzing your resume...")
def get_cached_feedback(scores, resume_text, jd_text):
    return gemini.generate_feedback(scores, resume_text, jd_text)

# --- ANALYSIS LOGIC ---
if st.button("Analyze Resume", type="primary"):
    if jd_text and uploaded_file:
        with st.spinner("Analyzing..."):
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            raw_resume = parser.extract_text(temp_path)
            clean_resume = " ".join(preprocessor.preprocess(raw_resume))
            clean_jd = " ".join(preprocessor.preprocess(jd_text))
            
            scores = ranker.get_composite_score(clean_resume, clean_jd)
            ai_res, usage = get_cached_feedback(scores, raw_resume, jd_text)

            # SAVE TO SESSION STATE (This fixes the NameError for 'raw_resume')
            st.session_state.analysis_results = {
                "scores": scores,
                "feedback": ai_res['feedback'],
                "missing_keywords": ai_res['keywords'],
                "raw_resume": raw_resume 
            }
            
            # Update token counters
            st.session_state.total_input_tokens += usage['input']
            st.session_state.total_output_tokens += usage['output']

            os.remove(temp_path)
    else:
        st.error("Please provide both a JD and a Resume.")

# --- DISPLAY ANALYSIS RESULTS ---
# This block runs every time the app reruns, ensuring the data stays visible
if st.session_state.analysis_results:
    res = st.session_state.analysis_results
    
    st.markdown("### Analysis Results")
    st.metric("Overall Match", f"{res['scores']['total_score']*100:.2f}%")
    
    c1, c2 = st.columns(2)
    c1.progress(res['scores']['semantic_match'], text="Semantic Vibe")
    c2.progress(res['scores']['keyword_match'], text="Hard Skill Match")

    st.subheader("AI Analysis")
    st.write(res['feedback'])

    # --- SKILL CLOUD ---
    st.markdown("---")
    st.subheader("🎯 Skill Gap Visualizer")
    missing = res.get('missing_keywords', [])
    if missing:
        tags_html = "".join([f'<span style="background-color: #1e293b; color: #60a5fa; padding: 4px 12px; margin: 4px; border-radius: 6px; font-weight: 600; font-size: 12px; border: 1px solid #334155; display: inline-block;">{word}</span>' for word in missing])
        st.markdown(f'<div style="line-height: 2.5;">{tags_html}</div>', unsafe_allow_html=True)

# --- CHAT SECTION ---
if st.session_state.analysis_results:
    st.markdown("---")
    st.header("💬 Chat with your Resume")

    # Display messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about the gaps..."):
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            # Use the raw_resume saved in session state
            resume_context = st.session_state.analysis_results['raw_resume']
            
            # Optimization: Last 5 messages for history
            recent_history = [
                {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
                for m in st.session_state.messages[-6:-1]
            ]
            
            response, usage = gemini.chat_with_resume(prompt, resume_context, recent_history)
            
            # Update tokens
            st.session_state.total_input_tokens += usage['input']
            st.session_state.total_output_tokens += usage['output']
            
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Trigger rerun to update the sidebar metrics immediately
        st.rerun()