import os
from google import genai # The new unified 2026 SDK
from dotenv import load_dotenv

load_dotenv()

class GeminiService:
    def __init__(self, model_name='gemini-2.0-flash'): 
        # The new SDK uses a centralized Client object
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_id = model_name

    def generate_feedback(self, scores, resume_text, jd_text):
        """
        Generates data-driven improvement suggestions using the new Client syntax.
        """
        prompt = f"""
        You are an expert Career Coach and ATS Specialist.
        Analyze these scores:
        - Overall Match: {scores['total_score']*100:.1f}%
        - Keyword Match: {scores['keyword_match']*100:.1f}%
        - Impact Score: {scores['impact_score']*100:.1f}%

        RESUME: {resume_text}
        JOB DESCRIPTION: {jd_text}
        
        Provide 3 actionable steps to improve these scores."""

        # Updated syntax for 2026: client.models.generate_content
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt
        )
        
        usage = {
            "input": response.usage_metadata.prompt_token_count,
            "output": response.usage_metadata.candidates_token_count
        }
        return response.text, usage

    def chat_with_resume(self, user_query, resume_text, history):
        """
        Handles chat using the new SDK's chat session management.
        """
        try:
            # Note: The new SDK handles history slightly differently within the generate_content call
            # or through a dedicated chat object if preferred.
            chat_response = self.client.models.generate_content(
                model=self.model_id,
                contents=f"Context: {resume_text}\n\nQuestion: {user_query}"
            )
            
            usage = {
                "input": chat_response.usage_metadata.prompt_token_count,
                "output": chat_response.usage_metadata.candidates_token_count
            }
            return chat_response.text, usage
        except Exception as e:
            return f"⚠️ Chat Error: {str(e)}", {"input": 0, "output": 0}