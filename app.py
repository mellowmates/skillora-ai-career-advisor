import os
import json
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
import google.generativeai as genai

# --- Initialization ---

# Load environment variables from .env file
load_dotenv()

# Configure the Flask application
app = Flask(__name__)

# Configure the Gemini API with the key from .env
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Warning: GEMINI_API_KEY not found. Please set it in your .env file.")
else:
    genai.configure(api_key=api_key)

# --- Data Loading ---

def load_data(filename):
    """A helper function to load JSON data from the data directory."""
    try:
        # Use absolute path to ensure we find the file regardless of working directory
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'data', filename)
        
        print(f"Looking for file at: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            print(f"Successfully loaded {filename}")
            return data
    except FileNotFoundError:
        print(f"Error: Could not find {filename} at path: {file_path}. Please make sure it exists in the 'data' directory.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode {filename}. Please check for valid JSON format.")
        return {}

# Load the market data at startup
market_data = load_data('job_market_data.json')

# Provide fallback market data if loading fails
if not market_data:
    print("Warning: Using fallback market data")
    market_data = {
        "tech_sector": {
            "growth_rate": "+18%",
            "average_salary": "₹8.5 LPA",
            "remote_opportunities": "65%",
            "skills_in_demand": ["Python", "JavaScript", "React", "Node.js", "AWS"]
        },
        "data_science": {
            "growth_rate": "+22%",
            "average_salary": "₹10.2 LPA", 
            "remote_opportunities": "55%",
            "skills_in_demand": ["Python", "R", "SQL", "Machine Learning", "Tableau"]
        }
    }

# --- Core Application Routes ---

@app.route('/')
def index():
    """Serves the main single-page application (index.html)."""
    return render_template('index.html')

# --- API Endpoints for Frontend ---

@app.route('/api/analyze', methods=['POST'])
def analyze_career():
    """
    API endpoint for the career advisor.
    Receives user assessment and returns personalized career recommendations using Gemini.
    """
    assessment = request.json
    print(f"Received assessment data: {assessment}")
    
    if not assessment:
        return jsonify({"error": "No assessment data received."}), 400
    
    if not api_key:
        return jsonify({"error": "API key for Gemini is not configured."}), 500
    
    if not market_data:
        return jsonify({"error": "Market data could not be loaded. Please check server configuration."}), 500

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Construct a detailed prompt for Gemini
        prompt = f"""
        You are 'Skillora', an expert AI career advisor for the Indian job market.
        Your task is to provide a personalized and insightful career analysis based on the user's assessment and current market data.

        **User Assessment:**
        - Experience: {assessment.get('experience', 'Not provided')}
        - Education: {assessment.get('education', 'Not provided')}
        - Field of Interest: {assessment.get('interestField', 'Not provided')}
        - Technical Skills: {assessment.get('techSkills', [])}
        - Career Goals: {assessment.get('goals', 'Not provided')}

        **Current Market Data:**
        {json.dumps(market_data, indent=2)}

        **CRITICAL: Your response MUST be a valid JSON object with this exact structure. Do not include any other text, explanation, or markdown formatting:**

        {{
          "top_matches": [
            {{
              "career": "Career Name",
              "match_percentage": "95%",
              "salary_range": "₹15L - ₹25L",
              "growth": "+12%"
            }},
            {{
              "career": "Another Career",
              "match_percentage": "85%",
              "salary_range": "₹12L - ₹20L",
              "growth": "+8%"
            }},
            {{
              "career": "Third Career",
              "match_percentage": "75%",
              "salary_range": "₹10L - ₹18L",
              "growth": "+15%"
            }}
          ],
          "skill_development": [
            {{
              "skill": "Skill Name",
              "recommendation": "Recommended",
              "proficiency_level": "75%"
            }},
            {{
              "skill": "Another Skill",
              "recommendation": "Good to have",
              "proficiency_level": "60%"
            }},
            {{
              "skill": "Third Skill",
              "recommendation": "Essential",
              "proficiency_level": "80%"
            }}
          ],
          "market_insights": {{
            "job_growth_percentage": "+15.2%",
            "remote_jobs_percentage": "45%",
            "avg_fresher_salary": "₹5.5 LPA",
            "skills_demand": "High"
          }}
        }}

        **Instructions:**
        1. Analyze the user's profile deeply considering their experience, education, and skills.
        2. Generate exactly 3 top_matches that are realistic and aligned with the user's profile.
        3. Suggest exactly 3 skill_development areas relevant to the recommended careers.
        4. Provide market_insights based on the market data, tailored to the recommended career paths.
        5. IMPORTANT: Return ONLY the JSON object. No additional text, explanations, or markdown formatting.
        """

        response = model.generate_content(prompt)
        
        # Clean the response to ensure it's valid JSON
        cleaned_response = response.text.strip()
        
        # Remove markdown formatting if present
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response.replace('```json', '', 1)
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response.rsplit('```', 1)[0]
        
        # Parse and validate the JSON response
        try:
            parsed_response = json.loads(cleaned_response)
            
            # Validate required fields exist
            if not all(key in parsed_response for key in ['top_matches', 'skill_development', 'market_insights']):
                raise ValueError("Missing required fields in AI response")
            
            return jsonify(parsed_response)
            
        except json.JSONDecodeError as je:
            print(f"JSON decode error: {je}")
            print(f"Raw response: {response.text}")
            # Return a fallback response with the actual AI content if possible
            return jsonify({
                "error": f"AI response formatting issue. Raw response: {response.text[:200]}..."
            }), 500

    except Exception as e:
        print(f"An error occurred during career analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to generate career analysis: {str(e)}"}), 500


@app.route('/api/chat', methods=['POST'])
def chat_with_gemini():
    """
    API endpoint for the AI Chatbot.
    Connects to the Gemini API for intelligent, conversational responses.
    """
    user_message = request.json.get('message', '')

    if not api_key:
        return jsonify({"response": "API key for Gemini is not configured. Please check the server setup."}), 500

    if not user_message:
        return jsonify({"response": "I didn't receive a message. Could you try again?"}), 400

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')

        prompt = f"""
        You are 'Skillora', a friendly and encouraging AI career advisor for the Indian job market. 
        Your goal is to provide helpful, concise, and supportive career advice.
        A user has asked the following question: "{user_message}"
        Please provide a helpful and brief response (2-3 sentences max).
        """

        response = model.generate_content(prompt)
        
        return jsonify({"response": response.text})

    except Exception as e:
        print(f"An error occurred with the Gemini API: {e}")
        error_message = "I'm having a little trouble connecting to my AI brain right now. Please try again in a moment."
        return jsonify({"response": error_message}), 503

# --- Main Entry Point ---

if __name__ == '__main__':
    app.run(debug=True)