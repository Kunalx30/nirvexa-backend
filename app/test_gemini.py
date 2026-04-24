import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)

print("Available embedding models:")
for m in genai.list_models():
    if "embed" in m.name.lower():
        print(f"  {m.name} → {m.supported_generation_methods}")