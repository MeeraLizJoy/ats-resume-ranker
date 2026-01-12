import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key = os.getenv("MY_API_KEY"))

class GeminiService:
    def __init__(self, model_name='gemini-3-flash-preview'): # Default set to the new model
        self.model = genai.GenerativeModel(model_name)

    def generate_feedback(self, scores, resume_text, jd_text):
        """
        Generated data-driven improvement suggestions.
        """
        # COntext Injection: We tell the AU exactly what math we found
        prompt = f"""
        You are a Senior Technical Recruiter.
        Scores: Overall {round(scores['total_score']*100, 2)}%, Semantic {round(scores['semantic_match']*100, 2)}%, Keyword {round(scores['keyword_match']*100, 2)}%

        TASK:
        1. Provide a professional analysis and 3 actionable tips in Markdown.
        2. Identify exactly which Hard Skills/Tools are in the JD but missing from the resume.

        FORMAT:
        ANALYSIS: [Your Markdown Text Here]
        GAPS: [Comma separated list of missing skills]
        """

        try:
            response = self.model.generate_content(prompt)
            raw_text = response.text
            
            # Parsing the AI response
            analysis = raw_text.split("GAPS:")[0].replace("ANALYSIS:", "").strip()
            try:
                # Extract words after 'GAPS:', split by comma, clean whitespace
                keywords = [k.strip().upper() for k in raw_text.split("GAPS:")[1].split(",") if len(k.strip()) > 0]
            except:
                keywords = []

            usage = {
                "input": response.usage_metadata.prompt_token_count,
                "output": response.usage_metadata.candidates_token_count
            }
            # WE WANT TO RETURN A DICTIONARY
            return {"feedback": analysis, "keywords": keywords}, usage
            
        except Exception as e:
            return {"feedback": f"Error: {str(e)}", "keywords": []}, {"input": 0, "output": 0}
    
    def chat_with_resume(self, user_query, resume_text, history):
        """
        Handles the Chat section where the user asks questions about their match.
        """
        try:
            chat = self.model.start_chat(history=history)
            context_prompt = f"Context: {resume_text}\n\nUser Question: {user_query}"
            response = chat.send_message(context_prompt)
            
            usage = {
                "input": response.usage_metadata.prompt_token_count,
                "output": response.usage_metadata.candidates_token_count
            }
            return response.text, usage
        except Exception as e:
            error_msg = "⚠️ Rate limit reached in chat." if "429" in str(e) else f"⚠️ Chat Error: {str(e)}"
            return error_msg, {"input": 0, "output": 0}
        
        
        