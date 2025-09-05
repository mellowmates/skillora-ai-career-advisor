"""
Skillora AI Career Advisor - Main Flask Application
A comprehensive career guidance platform powered by AI
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# Import our custom modules
from modules.personality_analyzer import PersonalityAnalyzer
from modules.skills_assessor import SkillsAssessor
from modules.job_market_scraper import JobMarketScraper
from modules.career_matcher import CareerMatcher

# Load environment variables (optional)
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'skillora-career-advisor-secret-key-2024')
CORS(app)

# Initialize modules
personality_analyzer = PersonalityAnalyzer()
skills_assessor = SkillsAssessor()
job_market_scraper = JobMarketScraper()
career_matcher = CareerMatcher()

# Load career data
career_matcher.load_data()

@app.route('/')
def index():
    """Main landing page"""
    return render_template('index.html')

@app.route('/api/personality/questions', methods=['GET'])
def get_personality_questions():
    """Get personality assessment questions"""
    try:
        questions = personality_analyzer.get_personality_questions()
        return jsonify({
            'success': True,
            'questions': questions
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/personality/analyze', methods=['POST'])
def analyze_personality():
    """Analyze user personality based on responses"""
    try:
        data = request.get_json()
        responses = data.get('responses', {})
        
        if not responses:
            return jsonify({
                'success': False,
                'error': 'No responses provided'
            }), 400
        
        # Analyze personality
        result = personality_analyzer.analyze_responses(responses)
        
        # Store in session for later use
        session['personality_profile'] = result.get('personality_profile')
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/skills/questions', methods=['GET'])
def get_skills_questions():
    """Get skills assessment questions"""
    try:
        questions = skills_assessor.get_skill_assessment_questions()
        return jsonify({
            'success': True,
            'questions': questions
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/skills/assess', methods=['POST'])
def assess_skills():
    """Assess user skills based on input"""
    try:
        data = request.get_json()
        user_input = data.get('user_input', {})
        
        if not user_input:
            return jsonify({
                'success': False,
                'error': 'No user input provided'
            }), 400
        
        # Assess skills
        result = skills_assessor.assess_skills(user_input)
        
        # Store in session for later use
        session['skills_profile'] = result.get('skills_profile')
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/career/match', methods=['POST'])
def match_career():
    """Match user with suitable career paths"""
    try:
        data = request.get_json()
        preferences = data.get('preferences', {})
        
        # Get stored profiles from session
        personality_profile = session.get('personality_profile')
        skills_profile = session.get('skills_profile')
        
        if not personality_profile or not skills_profile:
            return jsonify({
                'success': False,
                'error': 'Personality and skills profiles required. Please complete assessments first.'
            }), 400
        
        # Match career
        result = career_matcher.match_career(
            personality_profile, 
            skills_profile, 
            preferences
        )
        
        # Store results in session
        session['career_matches'] = result.get('top_careers', [])
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/career/roadmap/<career_id>', methods=['GET'])
def get_career_roadmap(career_id):
    """Get detailed career roadmap for a specific career"""
    try:
        roadmap = career_matcher.get_career_path_roadmap(career_id)
        return jsonify(roadmap)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/jobs/recommendations', methods=['POST'])
def get_job_recommendations():
    """Get job recommendations based on user profile"""
    try:
        data = request.get_json()
        location = data.get('location', '')
        
        # Get stored skills profile
        skills_profile = session.get('skills_profile')
        
        if not skills_profile:
            return jsonify({
                'success': False,
                'error': 'Skills profile required. Please complete skills assessment first.'
            }), 400
        
        # Get job recommendations
        user_skills = skills_profile.get('skills', {})
        result = job_market_scraper.get_job_recommendations(user_skills, location)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/jobs/market-data', methods=['GET'])
def get_market_data():
    """Get current job market data"""
    try:
        # Load market data from file
        with open('data/job_market_data.json', 'r') as f:
            market_data = json.load(f)
        
        return jsonify({
            'success': True,
            'market_data': market_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/careers/list', methods=['GET'])
def get_careers_list():
    """Get list of available careers"""
    try:
        # Load career profiles
        with open('data/career_profiles.json', 'r') as f:
            career_profiles = json.load(f)
        
        # Extract basic career information
        careers_list = []
        for career_id, career_data in career_profiles.items():
            careers_list.append({
                'id': career_id,
                'title': career_data.get('title', ''),
                'description': career_data.get('description', ''),
                'growth_outlook': career_data.get('growth_outlook', ''),
                'salary_range': career_data.get('salary_range', {})
            })
        
        return jsonify({
            'success': True,
            'careers': careers_list
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/skills/mapping', methods=['GET'])
def get_skills_mapping():
    """Get skills mapping data"""
    try:
        # Load skills mapping
        with open('data/skills_mapping.json', 'r') as f:
            skills_mapping = json.load(f)
        
        return jsonify({
            'success': True,
            'skills_mapping': skills_mapping
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/assessment/complete', methods=['POST'])
def complete_assessment():
    """Complete the full assessment and get comprehensive results"""
    try:
        # Get all stored data
        personality_profile = session.get('personality_profile')
        skills_profile = session.get('skills_profile')
        career_matches = session.get('career_matches', [])
        
        if not personality_profile or not skills_profile:
            return jsonify({
                'success': False,
                'error': 'Complete personality and skills assessments first'
            }), 400
        
        # Generate comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'personality_profile': personality_profile,
            'skills_profile': skills_profile,
            'career_matches': career_matches,
            'recommendations': {
                'next_steps': [
                    'Review your top career matches',
                    'Explore career roadmaps for your top choices',
                    'Identify skill gaps and development opportunities',
                    'Research job market trends in your areas of interest'
                ],
                'skill_development': skills_profile.get('recommendations', []),
                'personality_insights': personality_profile.get('recommendations', [])
            }
        }
        
        # Store complete report in session
        session['complete_report'] = report
        
        return jsonify({
            'success': True,
            'report': report
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/report/download', methods=['GET'])
def download_report():
    """Download the complete assessment report"""
    try:
        report = session.get('complete_report')
        
        if not report:
            return jsonify({
                'success': False,
                'error': 'No report available. Complete assessment first.'
            }), 400
        
        return jsonify({
            'success': True,
            'report': report
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/session/clear', methods=['POST'])
def clear_session():
    """Clear all session data"""
    try:
        session.clear()
        return jsonify({
            'success': True,
            'message': 'Session cleared successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)
