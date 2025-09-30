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
career_profiles = load_data('career_profiles.json') # Load this at startup
career_qualities = load_data('career_qualities.json')

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

@app.route('/api/career-cards', methods=['POST'])
def get_career_cards():
    """
    API endpoint to provide a list of careers for the swiping interface based on user assessment.
    """
    assessment = request.json
    if not assessment:
        return jsonify({"error": "No assessment data received."}), 400
    
    if not api_key:
        return jsonify({"error": "API key for Gemini is not configured."}), 500
    
    try:
        model = genai.GenerativeModel('models/gemini-pro-latest')
        prompt = f"""
        You are 'Skillora', an expert AI career advisor. Based on the user's assessment below,
        generate a list of career suggestions for a swiping interface.

        **User Assessment:**
        - Experience: {assessment.get('experience', 'Not provided')}
        - Education: {assessment.get('education', 'Not provided')}
        - Field of Interest: {assessment.get('interestField', 'Not provided')}
        - Technical Skills: {assessment.get('techSkills', [])}
        - Career Goals: {assessment.get('goals', 'Not provided')}
        - Career Values: {assessment.get('careerValues', [])}

        **CRITICAL: Your response MUST be a valid JSON array of exactly 12 career objects.
        Each object must have this exact structure:**
        {{
          "career_id": "unique_id",
          "title": "Career Name",
          "description": "A brief, engaging description of the job (2-3 sentences).",
          "required_skills": ["Skill1", "Skill2", "Skill3"],
          "interests": ["Interest1", "Interest2"],
          "job_market_demand": "High/Medium/Low"
        }}

        **Instructions:**
        1. The first 4 careers should be TOP MATCHES for the user's profile.
        2. The next 8 careers should be EXPLORATORY suggestions related to their skills/interests.
        3. Make descriptions engaging and easy to understand.
        4. Return ONLY the JSON array. No additional text or markdown.
        """
        
        response = model.generate_content(prompt)
        
        # Clean the response to ensure it's valid JSON
        cleaned_response = response.text.strip()
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response.replace('```json', '', 1)
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response.rsplit('```', 1)[0]
            
        return jsonify(json.loads(cleaned_response))

    except Exception as e:
        print(f"An error occurred during career cards generation: {e}")
        # Fallback to static data if API fails
        return jsonify(career_profiles)

@app.route('/api/analyze', methods=['POST'])
def analyze_career():
    """
    API endpoint for the career advisor.
    Receives user assessment and returns personalized career recommendations using Gemini.
    """
    assessment = request.json
    career_values = assessment.get('careerValues', []) # e.g., ["High Earning Potential", "Remote Work"]
    print(f"Received assessment data: {assessment}")
    
    if not assessment:
        return jsonify({"error": "No assessment data received."}), 400
    
    if not api_key:
        return jsonify({"error": "API key for Gemini is not configured."}), 500
    
    if not market_data:
        return jsonify({"error": "Market data could not be loaded. Please check server configuration."}), 500

    try:
        model = genai.GenerativeModel('models/gemini-pro-latest')

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
        - **Most Important Career Values:** {', '.join(career_values) if career_values else 'Not provided'}

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
        2. **CRUCIAL:** Heavily weigh the user's "Most Important Career Values" when selecting career matches. For example, if they value "Remote Work", prioritize careers known for that flexibility.
        3. Generate exactly 3 top_matches that are realistic and aligned with BOTH the user's skills AND their core values.
        4. Suggest exactly 3 skill_development areas relevant to the recommended careers.
        5. Provide market_insights based on the market data, tailored to the recommended career paths.
        6. IMPORTANT: Return ONLY the JSON object. No additional text, explanations, or markdown formatting.
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
                print(f"Missing required fields. Available keys: {list(parsed_response.keys()) if isinstance(parsed_response, dict) else 'Not a dict'}")
                print(f"Raw AI response: {response.text}")
                # Create fallback response based on user input
                fallback_response = {
                    "top_matches": [
                        {
                            "career": f"{assessment.get('interestField', 'Technology')} Specialist",
                            "match_percentage": "85%",
                            "salary_range": "₹6L - ₹12L",
                            "growth": "+15%"
                        },
                        {
                            "career": f"Junior {assessment.get('interestField', 'Technology')} Engineer",
                            "match_percentage": "80%",
                            "salary_range": "₹5L - ₹10L",
                            "growth": "+12%"
                        },
                        {
                            "career": f"{assessment.get('interestField', 'Technology')} Analyst",
                            "match_percentage": "75%",
                            "salary_range": "₹4L - ₹8L",
                            "growth": "+10%"
                        }
                    ],
                    "skill_development": [
                        {
                            "skill": assessment.get('techSkills', 'Programming').split(',')[0].strip(),
                            "recommendation": "Essential",
                            "proficiency_level": "70%"
                        },
                        {
                            "skill": "Communication Skills",
                            "recommendation": "Recommended",
                            "proficiency_level": "60%"
                        },
                        {
                            "skill": "Problem Solving",
                            "recommendation": "Good to have",
                            "proficiency_level": "65%"
                        }
                    ],
                    "market_insights": {
                        "job_growth_percentage": "+12%",
                        "remote_jobs_percentage": "40%",
                        "avg_fresher_salary": "₹5.5 LPA",
                        "skills_demand": "High"
                    }
                }
                print("Using fallback response due to missing required fields")
                return jsonify(fallback_response)
            
            return jsonify(parsed_response)
            
        except json.JSONDecodeError as je:
            print(f"JSON decode error: {je}")
            print(f"Raw response: {response.text}")
            # Return a fallback response based on user input
            fallback_response = {
                "top_matches": [
                    {
                        "career": f"{assessment.get('interestField', 'Technology')} Specialist",
                        "match_percentage": "85%",
                        "salary_range": "₹6L - ₹12L",
                        "growth": "+15%"
                    },
                    {
                        "career": f"Junior {assessment.get('interestField', 'Technology')} Engineer",
                        "match_percentage": "80%",
                        "salary_range": "₹5L - ₹10L",
                        "growth": "+12%"
                    },
                    {
                        "career": f"{assessment.get('interestField', 'Technology')} Analyst",
                        "match_percentage": "75%",
                        "salary_range": "₹4L - ₹8L",
                        "growth": "+10%"
                    }
                ],
                "skill_development": [
                    {
                        "skill": assessment.get('techSkills', 'Programming').split(',')[0].strip(),
                        "recommendation": "Essential",
                        "proficiency_level": "70%"
                    },
                    {
                        "skill": "Communication Skills",
                        "recommendation": "Recommended",
                        "proficiency_level": "60%"
                    },
                    {
                        "skill": "Problem Solving",
                        "recommendation": "Good to have",
                        "proficiency_level": "65%"
                    }
                ],
                "market_insights": {
                    "job_growth_percentage": "+12%",
                    "remote_jobs_percentage": "40%",
                    "avg_fresher_salary": "₹5.5 LPA",
                    "skills_demand": "High"
                }
            }
            print("Using fallback response due to AI parsing error")
            return jsonify(fallback_response)

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
        model = genai.GenerativeModel('models/gemini-pro-latest')

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

@app.route('/api/generate-resume', methods=['POST'])
def generate_resume():
    """
    API endpoint for generating comprehensive resume content using Gemini.
    Receives user assessment and personal info, returns complete resume data.
    """
    data = request.json
    assessment = data.get('assessment', {})
    personal_info = data.get('personalInfo', {})
    
    print(f"Received data for resume generation: {data}")

    if not assessment and not personal_info:
        return jsonify({"error": "No assessment or personal data received."}), 400

    if not api_key:
        return jsonify({"error": "API key for Gemini is not configured."}), 500

    try:
        model = genai.GenerativeModel('models/gemini-pro-latest')

        # Construct a detailed prompt for Gemini
        prompt = f"""
        You are 'Skillora', an expert AI resume writer for the Indian job market.
        Your task is to generate professional and comprehensive resume content based on user data.

        **User Assessment:**
        - Experience: {assessment.get('experience', 'Not provided')}
        - Education: {assessment.get('education', 'Not provided')}
        - Field of Interest: {assessment.get('interestField', 'Not provided')}
        - Technical Skills: {assessment.get('techSkills', [])}
        - Career Goals: {assessment.get('goals', 'Not provided')}
        - Career Values: {assessment.get('careerValues', [])}

        **Personal Information:**
        - Name: {personal_info.get('name', '[Your Name]')}
        - Email: {personal_info.get('email', '[Your Email]')}
        - Phone: {personal_info.get('phone', '[Your Phone]')}
        - Location: {personal_info.get('location', 'India')}

        **CRITICAL: Your response MUST be a valid JSON object with this exact structure:**

        {{
          "personal_info": {{
            "name": "{personal_info.get('name', '[Your Name]')}",
            "email": "{personal_info.get('email', '[Your Email]')}",
            "phone": "{personal_info.get('phone', '[Your Phone]')}",
            "location": "{personal_info.get('location', 'India')}",
            "linkedin": "linkedin.com/in/[username]",
            "github": "github.com/[username]"
          }},
          "summary": "A compelling 3-4 sentence professional summary tailored to their career goals and experience level.",
          "skills": {{
            "programming_languages": ["Language1", "Language2", "Language3"],
            "frameworks": ["Framework1", "Framework2", "Framework3"],
            "developer_tools": ["Tool1", "Tool2", "Tool3"],
            "soft_skills": ["Skill1", "Skill2", "Skill3", "Skill4"]
          }},
          "experience": [
            {{
              "title": "Relevant Job Title",
              "company": "Company Name",
              "location": "City, India",
              "dates": "Month Year - Present/Month Year",
              "description": "Detailed description of responsibilities and achievements with quantifiable results where possible."
            }}
          ],
          "education": [
            {{
              "degree": "Degree Name",
              "institution": "Institution Name",
              "location": "City, India",
              "dates": "Year - Year",
              "details": "Relevant coursework, CGPA, or achievements."
            }}
          ],
          "projects": [
            {{
              "name": "Project Name 1",
              "description": "Comprehensive description highlighting technical implementation, challenges solved, and impact.",
              "technologies": ["Tech1", "Tech2", "Tech3"]
            }},
            {{
              "name": "Project Name 2",
              "description": "Another detailed project description with specific achievements.",
              "technologies": ["Tech4", "Tech5", "Tech6"]
            }}
          ],
          "certifications": [
            {{
              "name": "Relevant Certification",
              "issuer": "Issuing Organization",
              "date": "Month Year"
            }}
          ]
        }}

        **Instructions:**
        1. Generate realistic and professional content based on the user's profile
        2. Tailor experience and projects to match their field of interest and skill level
        3. Include quantifiable achievements where appropriate
        4. Make the summary compelling and specific to their career goals
        5. Ensure all content is relevant to the Indian job market
        6. Return ONLY the JSON object. No additional text or markdown.
        """

        response = model.generate_content(prompt)
        
        # Clean the response to ensure it's valid JSON
        cleaned_response = response.text.strip()
        
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response.replace('```json', '', 1)
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response.rsplit('```', 1)[0]
        
        try:
            parsed_response = json.loads(cleaned_response)
            required_fields = ['personal_info', 'summary', 'skills', 'experience', 'education', 'projects']
            if not all(key in parsed_response for key in required_fields):
                raise ValueError("Missing required fields in AI response")
            
            return jsonify(parsed_response)
            
        except json.JSONDecodeError as je:
            print(f"JSON decode error: {je}")
            print(f"Raw response: {response.text}")
            return jsonify({
                "error": f"AI response formatting issue. Raw response: {response.text[:200]}..."
            }), 500

    except Exception as e:
        print(f"An error occurred during resume generation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to generate resume content: {str(e)}"}), 500

@app.route('/api/analyze-qualities', methods=['POST'])
def analyze_qualities():
    """
    Analyzes user's choices from the comparative judgment quiz using Gemini.
    """
    choices = request.json.get('choices', [])
    if not choices:
        return jsonify({"error": "No choices received."}), 400

    try:
        model = genai.GenerativeModel('models/gemini-pro-latest')
        prompt = f"""
        You are an expert career psychologist. A user has completed a comparative judgment quiz
        by choosing between pairs of career qualities. Their winning choices were: {', '.join(choices)}.

        Based on these choices, analyze and infer the user's top 3 most important career values.
        
        CRITICAL: Your response MUST be a valid JSON object with this exact structure:
        {{
          "top_values": ["Value 1", "Value 2", "Value 3"]
        }}
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
            if not all(key in parsed_response for key in ['top_values']):
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
        print(f"An error occurred during quality analysis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate-pathway', methods=['POST'])
def generate_pathway():
    """
    Generates a personalized career pathway with visual elements.
    """
    data = request.json
    user_profile = data.get('userProfile')
    target_career = data.get('targetCareer')

    if not user_profile or not target_career:
        return jsonify({"error": "User profile and target career are required."}), 400

    try:
        model = genai.GenerativeModel('models/gemini-pro-latest')
        prompt = f"""
        You are an expert career pathway advisor for the Indian market.
        
        **User's Current Profile:**
        - Experience: {user_profile.get('experience', 'N/A')}
        - Education: {user_profile.get('education', 'N/A')}
        - Skills: {user_profile.get('techSkills', [])}
        
        **Target Career:** {target_career.get('career', 'N/A')}
        
        Your task is to generate a realistic, step-by-step pathway for this user.
        
        CRITICAL: Your response MUST be a valid JSON object. The root object must have a "pathway" key,
        which is an array of step objects. Each step object MUST have this exact structure:
        {{
          "step": 1,
          "icon": "A relevant Lucide icon name (e.g., 'book-open', 'code-2', 'briefcase', 'users')",
          "action": "A concise title for the step (e.g., 'Master Advanced Python').",
          "details": "More details on what to do.",
          "time": "Estimated time (e.g., '2-3 Months')",
          "cost": "Estimated cost in INR (e.g., '₹5,000 - ₹15,000')"
        }}
        
        Instructions:
        1. Create between 3 to 5 logical steps.
        2. Assign a relevant Lucide icon name for each step.
        3. Keep descriptions clear and actionable.
        4. Return ONLY the JSON object.
        """
        response = model.generate_content(prompt)
        # Add the same JSON cleaning logic as in your other routes
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
        return jsonify(json.loads(cleaned_response))

    except Exception as e:
        print(f"An error occurred during pathway generation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/career-suggestions', methods=['POST'])
def get_career_suggestions():
    """
    API endpoint to generate a list of ~13 career suggestions based on user assessment
    for the swiping interface.
    """
    assessment = request.json
    if not assessment:
        return jsonify({"error": "No assessment data received."}), 400

    try:
        model = genai.GenerativeModel('models/gemini-pro-latest')
        prompt = f"""
        You are 'Skillora', an expert AI career advisor. Based on the user's assessment below,
        generate a list of career suggestions.

        **User Assessment:**
        - Experience: {assessment.get('experience', 'Not provided')}
        - Education: {assessment.get('education', 'Not provided')}
        - Field of Interest: {assessment.get('interestField', 'Not provided')}
        - Technical Skills: {assessment.get('techSkills', [])}
        - Career Goals: {assessment.get('goals', 'Not provided')}

        **CRITICAL: Your response MUST be a valid JSON object with an array of exactly 13 careers.
        Each object in the array must have this exact structure:**
        {{
          "title": "Career Name",
          "description": "A brief, easy-to-understand description of the job."
        }}

        **Instructions:**
        1. The first 3 careers should be the TOP MATCHES for the user.
        2. The next 10 careers should be EXPLORATORY suggestions that are related to the user's skills and interests but might be outside their direct field of interest.
        3. Return ONLY the JSON object. No additional text or markdown.
        """
        response = model.generate_content(prompt)
        
        # Clean the response to ensure it's valid JSON
        cleaned_response = response.text.strip()
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response.replace('```json', '', 1)
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response.rsplit('```', 1)[0]
            
        return jsonify(json.loads(cleaned_response))

    except Exception as e:
        print(f"An error occurred during career suggestion generation: {e}")
        return jsonify({"error": f"Failed to generate career suggestions: {str(e)}"}), 500

@app.route('/api/swipe-results', methods=['POST'])
def analyze_swipe_results():
    """
    Analyzes user's swiping behavior to generate personalized insights.
    """
    data = request.json
    liked_careers = data.get('likedCareers', [])
    disliked_careers = data.get('dislikedCareers', [])
    assessment = data.get('assessment', {})
    
    if not liked_careers:
        return jsonify({"error": "No liked careers to analyze."}), 400

    try:
        model = genai.GenerativeModel('models/gemini-pro-latest')
        prompt = f"""
        You are 'Skillora', an expert AI career advisor. A user has completed a career swiping exercise.
        
        **User Assessment:**
        - Experience: {assessment.get('experience', 'Not provided')}
        - Education: {assessment.get('education', 'Not provided')}
        - Field of Interest: {assessment.get('interestField', 'Not provided')}
        - Technical Skills: {assessment.get('techSkills', [])}
        - Career Goals: {assessment.get('goals', 'Not provided')}
        
        **Liked Careers:** {', '.join([career['title'] for career in liked_careers])}
        **Disliked Careers:** {', '.join([career['title'] for career in disliked_careers])}
        
        Based on their preferences and assessment, provide insights and updated career recommendations.
        
        CRITICAL: Your response MUST be a valid JSON object with this exact structure:
        {{
          "insights": "A brief analysis of their career preferences based on their swipes and assessment.",
          "recommended_skills": ["Skill 1", "Skill 2", "Skill 3"],
          "career_path_suggestions": [
            {{
              "career": "Career Name",
              "match_reason": "Why this matches their preferences and profile"
            }}
          ]
        }}
        """
        response = model.generate_content(prompt)
        
        # Clean the response to ensure it's valid JSON
        cleaned_response = response.text.strip()
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response.replace('```json', '', 1)
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response.rsplit('```', 1)[0]
            
        return jsonify(json.loads(cleaned_response))

    except Exception as e:
        print(f"An error occurred during swipe analysis: {e}")
        return jsonify({"error": f"Failed to analyze swipe results: {str(e)}"}), 500

@app.route('/api/market-insights', methods=['POST'])
def get_market_insights():
    """
    Generates personalized market insights based on user assessment.
    """
    assessment = request.json
    if not assessment:
        return jsonify({"error": "No assessment data received."}), 400

    if not api_key:
        return jsonify({"error": "API key for Gemini is not configured."}), 500

    try:
        model = genai.GenerativeModel('models/gemini-pro-latest')
        prompt = f"""
        You are 'Skillora', an expert AI career market analyst for the Indian job market.
        
        **User Assessment:**
        - Experience: {assessment.get('experience', 'Not provided')}
        - Education: {assessment.get('education', 'Not provided')}
        - Field of Interest: {assessment.get('interestField', 'Not provided')}
        - Technical Skills: {assessment.get('techSkills', [])}
        - Career Goals: {assessment.get('goals', 'Not provided')}
        - Career Values: {assessment.get('careerValues', [])}

        Generate personalized market insights and trends relevant to this user's profile.
        
        CRITICAL: Your response MUST be a valid JSON object with this exact structure:
        {{
          "market_overview": {{
            "top_in_demand_skill": "Most relevant skill for user",
            "fastest_growing_sector": "Sector most relevant to user",
            "avg_salary_range": "₹X.X LPA - ₹Y.Y LPA"
          }},
          "personalized_trends": [
            {{
              "trend": "Trend Name",
              "relevance": "High/Medium/Low",
              "description": "How this trend affects the user's career path"
            }}
          ],
          "salary_insights": {{
            "entry_level": "₹X.X LPA",
            "mid_level": "₹Y.Y LPA",
            "senior_level": "₹Z.Z LPA",
            "growth_potential": "X% annually"
          }},
          "skill_recommendations": [
            {{
              "skill": "Skill Name",
              "priority": "High/Medium/Low",
              "reason": "Why this skill is important for the user"
            }}
          ]
        }}
        
        Instructions:
        1. Tailor all insights to the user's specific profile and interests
        2. Use realistic Indian market data and salary ranges
        3. Focus on actionable insights
        4. Return ONLY the JSON object.
        """
        
        response = model.generate_content(prompt)
        
        # Clean the response to ensure it's valid JSON
        cleaned_response = response.text.strip()
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response.replace('```json', '', 1)
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response.rsplit('```', 1)[0]
            
        return jsonify(json.loads(cleaned_response))

    except Exception as e:
        print(f"An error occurred during market insights generation: {e}")
        return jsonify({"error": f"Failed to generate market insights: {str(e)}"}), 500

# --- Main Entry Point ---



if __name__ == '__main__':
    app.run(debug=True)
