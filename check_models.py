import os
import google.generativeai as genai
from dotenv import load_dotenv

# --- Initialization ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ Error: GEMINI_API_KEY not found in .env file.")
else:
    print("✅ GEMINI_API_KEY loaded successfully.")
    try:
        genai.configure(api_key=api_key)
        
        print("\n🔎 Attempting to list available models...")
        
        # List all models that support generateContent
        model_count = 0
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"  - {m.name}")
                model_count += 1
        
        if model_count > 0:
            print(f"\n✅ Found {model_count} usable models. Please use one of the names listed above in your app.py file.")
        else:
            print("\n❌ No models supporting 'generateContent' found. There might be an issue with your API key or permissions.")

    except Exception as e:
        print(f"\n❌ An error occurred while communicating with the Gemini API: {e}")