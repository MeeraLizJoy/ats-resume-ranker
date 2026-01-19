from fpdf import FPDF

def generate_pdf_report(candidate_name, scores, jd_text):
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt=f"ATS Analysis Report: {candidate_name}", ln=True, align='C')
    pdf.ln(10)
    
    # Scores Section
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="--- Match Scores ---", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, txt=f"Overall Match: {scores['total_score']*100:.1f}%", ln=True)
    pdf.cell(200, 10, txt=f"Keyword Alignment: {scores['keyword_match']*100:.1f}%", ln=True)
    pdf.cell(200, 10, txt=f"Impact/Metric Score: {scores['impact_score']*100:.1f}%", ln=True)
    pdf.ln(5)

    # Methodology
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 10, txt="Analysis generated via Hybrid NER Extraction (spaCy) and Semantic Vector Similarity (all-MiniLM-L6-v2).")
    
    # Return as bytes
    return pdf.output(dest='S').encode('latin-1')

def generate_chat_txt(history):
    """Converts session history list to a clean string."""
    report = "--- RESUME COACH CHAT HISTORY ---\n\n"
    for msg in history:
        role = "Coach" if msg["role"] == "assistant" else "User"
        report += f"{role}: {msg['content']}\n\n"
    return report