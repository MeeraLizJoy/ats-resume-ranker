from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("MY_API_KEY"))

print("Available models:")
for model in client.models.list():
    print(f"Model name: {model.name}, Supports: {model.supported_actions}")
