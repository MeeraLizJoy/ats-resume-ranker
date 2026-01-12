import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key = os.getenv("MY_API_KEY"))

class GeminiService:
    def __init__(self, model_name = 'gemini-1.5-flash'):
        self.model = genai.GenerativeModel(model_name)

    def generate_feedback(self, scores, resume_text, jd_text):
        """
        Generated data-driven improvement suggestions.
        """
        # COntext Injection: We tell the AU exactly what math we found
        prompt = f"""
        You are a technical hiring manager. A candidate has been analyzed by our ATS system.
        
        SYSTEM DATA:
        - Semantic Match Score: {scores['semantic_match']*100:.f}%
        - Keyword/Hard Skill Match: {scores['keyword_match']*100:.f}%
        - Total Composite Score: {scores['total_score']*100:.f}%

        TASK:
        Based on these specific mathmatical scores, provide:
        1. A brief summary of why the score os what it is.
        2. Three actionable bullet points to improve the resume for this specific JD.

        JD Snippet: {jd_text[:500]}....
        Resume Snippet: {resume_text[:500]}...
        """

        response = self.model.generate_content(prompt)
        return response.text
    
    def chat_with_resume(self, user_query, resume_text, history):
        """
        Handles the Chat section where the user asks questions about their match.
        """
        chat = self.model.start_chat(history = history)
        context_prompt = f"The user is asking about their resume. Context: {resume_text}\n\nUser Question: {user_query}"

        response = chat.send_message(context_prompt)
        return response.text
        
        
        