from flask import Flask, render_template, request, jsonify, session
import os
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Import modules with error handling
try:
    from modules.data_loader import DataLoader
    from modules.user_profiler import UserProfiler
    from modules.personality_assessor import PersonalityAssessor
    from modules.skills_assessor import SkillsAssessor
    from modules.career_matcher import CareerMatcher
    from modules.learning_roadmap import LearningRoadmap
    from modules.market_analyzer import MarketAnalyzer
    from chatbot.chatbot_api import ChatbotAPI
    MODULES_AVAILABLE = True
    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"⚠️ Warning: Some modules not found: {e}")
    print("The app will run with limited functionality")
    MODULES_AVAILABLE = False

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure random key

# Initialize modules with error handling
if MODULES_AVAILABLE:
    try:
        data_loader = DataLoader()
        user_profiler = UserProfiler(data_loader)
        personality_assessor = PersonalityAssessor(data_loader)
        skills_assessor = SkillsAssessor(data_loader)
        career_matcher = CareerMatcher(data_loader)
        learning_roadmap = LearningRoadmap(data_loader)
        market_analyzer = MarketAnalyzer(data_loader)
        chatbot_api = ChatbotAPI(data_loader)
        print("✅ All modules initialized successfully")
    except Exception as e:
        print(f"⚠️ Warning: Module initialization failed: {e}")
        MODULES_AVAILABLE = False

# Create fallback objects if modules are not available
if not MODULES_AVAILABLE:
    data_loader = None
    user_profiler = None
    personality_assessor = None
    skills_assessor = None
    career_matcher = None
    learning_roadmap = None
    market_analyzer = None
    chatbot_api = None


# Web Routes
@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/profile')
def profile():
    """User profile creation page"""
    return render_template('profile.html')

@app.route('/assessment')
def assessment():
    """Assessment page"""
    return render_template('assessment.html')

@app.route('/dashboard')
def dashboard():
    """Main dashboard page"""
    try:
        # Get user data from session
        user_data = {
            'personality': session.get('personality', {}),
            'skills': session.get('skills', {}),
            'profile': session.get('profile', {}),
            'profile_completeness': 75  # Default value
        }
        
        # Get career recommendations if user has assessment data
        recommendations = []
        if user_data['personality'] or user_data['skills']:
            try:
                if career_matcher:
                    recommendations = career_matcher.get_recommendations(user_data)
                else:
                    # Fallback recommendations
                    recommendations = [
                        {
                            'career_title': 'Software Engineer',
                            'career_category': 'Technology',
                            'career_description': 'Develop software applications and systems',
                            'final_score': 0.85,
                            'growth_outlook': 'High',
                            'salary_range': {'formatted': '₹6-12 LPA'},
                            'skill_analysis': {'skill_match_percentage': 78},
                            'career_id': 'software_engineer'
                        },
                        {
                            'career_title': 'Data Scientist',
                            'career_category': 'Technology',
                            'career_description': 'Analyze data to drive business decisions',
                            'final_score': 0.78,
                            'growth_outlook': 'Very High',
                            'salary_range': {'formatted': '₹8-15 LPA'},
                            'skill_analysis': {'skill_match_percentage': 72},
                            'career_id': 'data_scientist'
                        }
                    ]
            except Exception as e:
                print(f"Error getting recommendations: {e}")
                recommendations = []
        
        # Get market overview
        market_overview = {}
        try:
            if market_analyzer:
                market_overview = market_analyzer.get_comprehensive_market_overview()
            else:
                # Fallback market data
                market_overview = {
                    'total_jobs_available': 125000,
                    'overall_growth_rate': 12.5,
                    'market_health_score': 0.78,
                    'top_sectors': [
                        {'sector': 'Technology', 'growth_rate': 15.2},
                        {'sector': 'Healthcare', 'growth_rate': 11.8},
                        {'sector': 'Finance', 'growth_rate': 9.4}
                    ]
                }
        except Exception as e:
            print(f"Error getting market overview: {e}")
            market_overview = {
                'total_jobs_available': 125000,
                'overall_growth_rate': 12.5,
                'market_health_score': 0.78,
                'top_sectors': []
            }
        
        return render_template('dashboard.html', 
                             user_data=user_data,
                             recommendations=recommendations,
                             market_overview=market_overview)
                             
    except Exception as e:
        print(f"Dashboard error: {e}")
        # Return dashboard with minimal data
        return render_template('dashboard.html',
                             user_data={'profile_completeness': 0},
                             recommendations=[],
                             market_overview={'total_jobs_available': 0, 'overall_growth_rate': 0})

@app.route('/career/<career_id>')
def career_detail(career_id):
    """Career details page"""
    return render_template('career_detail.html', career_id=career_id)

@app.route('/roadmap')
def roadmap():
    """Learning roadmap page"""
    try:
        # Get user data from session
        user_data = {
            'personality': session.get('personality', {}),
            'skills': session.get('skills', {}),
            'profile': session.get('profile', {})
        }
        
        # Generate a default roadmap if no specific career is selected
        roadmap_data = {}
        if learning_roadmap and user_data:
            try:
                roadmap_data = learning_roadmap.generate_roadmap('software_engineer', user_data)
            except Exception as e:
                print(f"Error generating roadmap: {e}")
                roadmap_data = {'error': 'Unable to generate roadmap. Please complete your assessment first.'}
        else:
            roadmap_data = {'error': 'Please complete your assessment to generate a personalized roadmap.'}
        
        return render_template('roadmap.html', roadmap=roadmap_data)
        
    except Exception as e:
        print(f"Roadmap error: {e}")
        return render_template('roadmap.html', roadmap={'error': 'Unable to load roadmap.'})

@app.route('/roadmap/<career_id>')
def roadmap_career(career_id):
    """Learning roadmap for specific career"""
    try:
        # Get user data from session
        user_data = {
            'personality': session.get('personality', {}),
            'skills': session.get('skills', {}),
            'profile': session.get('profile', {})
        }
        
        # Generate roadmap for specific career
        roadmap_data = {}
        if learning_roadmap:
            try:
                roadmap_data = learning_roadmap.generate_roadmap(career_id, user_data)
            except Exception as e:
                print(f"Error generating roadmap for {career_id}: {e}")
                roadmap_data = {'error': f'Unable to generate roadmap for {career_id}.'}
        else:
            roadmap_data = {'error': 'Roadmap service is currently unavailable.'}
        
        return render_template('roadmap.html', roadmap=roadmap_data, career_id=career_id)
        
    except Exception as e:
        print(f"Career roadmap error: {e}")
        return render_template('roadmap.html', roadmap={'error': 'Unable to load career roadmap.'}, career_id=career_id)

@app.route('/chatbot')
def chatbot():
    """AI chatbot interface"""
    return render_template('chatbot.html')

@app.route('/chat')
def chat():
    """Alternative chatbot route"""
    return render_template('chatbot.html')

@app.route('/chatbot_page')
def chatbot_page():
    """Alternative chatbot page route"""
    return render_template('chatbot.html')

# API Endpoints - Personality Assessment
@app.route('/api/personality/questions', methods=['GET'])
def get_personality_questions():
    """Get personality assessment questions"""
    try:
        if personality_assessor:
            questions = personality_assessor.get_assessment_questions()
            return jsonify({"status": "success", "data": questions})
        else:
            # Fallback questions
            fallback_questions = [
                {"id": 1, "text": "I enjoy meeting new people", "trait": "extraversion", "scale": "1-5"},
                {"id": 2, "text": "I am organized and methodical", "trait": "conscientiousness", "scale": "1-5"},
                {"id": 3, "text": "I enjoy creative activities", "trait": "openness", "scale": "1-5"},
                {"id": 4, "text": "I am sympathetic to others' feelings", "trait": "agreeableness", "scale": "1-5"},
                {"id": 5, "text": "I remain calm under pressure", "trait": "neuroticism", "scale": "1-5"}
            ]
            return jsonify({"status": "success", "data": fallback_questions})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/personality/analyze', methods=['POST'])
def analyze_personality():
    """Analyze personality from responses"""
    try:
        data = request.json
        responses = data.get('responses', {})
        
        if personality_assessor:
            personality_result = personality_assessor.analyze_personality(responses)
        else:
            # Fallback personality result
            personality_result = {
                'personality_profile': {
                    'openness': {'score': 5, 'level': 'Neutral'},
                    'conscientiousness': {'score': 5, 'level': 'Neutral'},
                    'extraversion': {'score': 5, 'level': 'Neutral'},
                    'agreeableness': {'score': 5, 'level': 'Neutral'},
                    'neuroticism': {'score': 5, 'level': 'Neutral'}
                },
                'summary': 'Basic personality assessment completed',
                'recommendations': ['Consider careers that match your balanced personality profile']
            }
        
        session['personality'] = personality_result
        return jsonify({"status": "success", "data": personality_result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# API Endpoints - Skills Assessment
@app.route('/api/skills/questions', methods=['GET'])
def get_skills_questions():
    """Get skills assessment questions"""
    try:
        if skills_assessor:
            questions = skills_assessor.get_assessment_questions()
            return jsonify({"status": "success", "data": questions})
        else:
            # Fallback skills questions
            fallback_questions = [
                {"category": "Technical Skills", "skills": ["Python", "JavaScript", "SQL", "Machine Learning"]},
                {"category": "Soft Skills", "skills": ["Communication", "Leadership", "Problem Solving", "Teamwork"]},
                {"category": "Domain Knowledge", "skills": ["Data Analysis", "Web Development", "Mobile Development", "AI/ML"]}
            ]
            return jsonify({"status": "success", "data": fallback_questions})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/skills/assess', methods=['POST'])
def assess_skills():
    """Assess user skills"""
    try:
        data = request.json
        skills_data = data.get('skills', {})
        experience = data.get('experience', '')
        education = data.get('education', '')
        
        if skills_assessor:
            skills_result = skills_assessor.assess_skills(skills_data, experience, education)
        else:
            # Fallback skills assessment
            skills_result = {
                'skill_profile': {
                    'technical_skills': skills_data.get('technical', []),
                    'soft_skills': skills_data.get('soft', []),
                    'overall_score': 75
                },
                'skill_gaps': ['Advanced Python', 'Machine Learning', 'Cloud Computing'],
                'recommendations': ['Focus on developing technical skills', 'Consider online courses for skill gaps']
            }
        
        session['skills'] = skills_result
        return jsonify({"status": "success", "data": skills_result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# API Endpoints - Career Matching
@app.route('/api/career/match', methods=['POST'])
def match_careers():
    """Match user with suitable careers"""
    try:
        data = request.json
        user_profile = {
            'personality': session.get('personality', {}),
            'skills': session.get('skills', {}),
            'preferences': data.get('preferences', {}),
            'user_data': data.get('user_data', {})
        }
        
        if career_matcher:
            career_matches = career_matcher.get_recommendations(user_profile)
        else:
            # Fallback career recommendations
            career_matches = [
                {
                    'career_id': 'software_engineer',
                    'title': 'Software Engineer',
                    'compatibility_score': 85,
                    'description': 'Develop software applications and systems',
                    'salary_range': '₹6-12 LPA',
                    'growth_prospects': 'High'
                },
                {
                    'career_id': 'data_scientist',
                    'title': 'Data Scientist',
                    'compatibility_score': 78,
                    'description': 'Analyze data to drive business decisions',
                    'salary_range': '₹8-15 LPA',
                    'growth_prospects': 'Very High'
                }
            ]
        
        session['career_matches'] = career_matches
        return jsonify({"status": "success", "data": career_matches})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/career/details/<career_id>', methods=['GET'])
def get_career_details(career_id):
    """Get detailed information about a specific career"""
    try:
        if data_loader and market_analyzer:
            career_details = data_loader.get_career_by_id(career_id)
            market_data = market_analyzer.get_location_market_insights()
        else:
            # Fallback career details
            career_details = {
                'career_id': career_id,
                'title': career_id.replace('_', ' ').title(),
                'description': f'Detailed information about {career_id.replace("_", " ")} career',
                'required_skills': ['Skill 1', 'Skill 2', 'Skill 3'],
                'education_requirements': ['Bachelor\'s degree in relevant field'],
                'growth_prospects': 'Good'
            }
            market_data = {
                'demand': 'High',
                'salary_trend': 'Increasing',
                'job_openings': 1000
            }
        
        result = {
            'career_info': career_details,
            'market_data': market_data
        }
        
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# API Endpoints - Learning Roadmap
@app.route('/api/roadmap/generate', methods=['POST'])
def generate_roadmap():
    """Generate learning roadmap for a career"""
    try:
        data = request.json
        career_id = data.get('career_id')
        current_skills = session.get('skills', {})
        
        roadmap = learning_roadmap.generate_roadmap(career_id, current_skills)
        
        return jsonify({"status": "success", "data": roadmap})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# API Endpoints - Market Analysis
@app.route('/api/market/trends', methods=['GET'])
def get_market_trends():
    """Get job market trends"""
    try:
        location = request.args.get('location', 'india')
        trends = market_analyzer.get_market_trends(location)
        
        return jsonify({"status": "success", "data": trends})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/market/salary/<career_id>', methods=['GET'])
def get_salary_data(career_id):
    """Get salary information for a career"""
    try:
        location = request.args.get('location', 'india')
        experience = request.args.get('experience', '0-2')
        
        salary_data = market_analyzer.get_salary_data(career_id, location, experience)
        
        return jsonify({"status": "success", "data": salary_data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# API Endpoints - Chatbot
@app.route('/api/chatbot/message', methods=['POST'])
def chatbot_message():
    """Process chatbot message"""
    try:
        data = request.json
        message = data.get('message', '')
        conversation_id = data.get('conversation_id', 'default')
        
        if chatbot_api:
            response_data = chatbot_api.process_message(message, session.get('profile', {}), conversation_id)
        else:
            # Fallback chatbot response
            response_data = {
                'response': f'Thank you for your message: "{message}". The AI chatbot is currently unavailable, but I can help with basic career guidance questions.',
                'suggestions': ['Tell me about software engineering', 'What skills do I need for data science?', 'How to prepare for interviews?']
            }
        
        return jsonify({"status": "success", "data": response_data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Profile Management
@app.route('/api/profile/save', methods=['POST'])
def save_profile():
    """Save user profile data"""
    try:
        data = request.json
        profile_data = {
            'demographics': data.get('demographics', {}),
            'education': data.get('education', {}),
            'experience': data.get('experience', {}),
            'preferences': data.get('preferences', {})
        }
        
        if user_profiler:
            profile_result = user_profiler.create_profile(profile_data)
        else:
            # Fallback profile creation
            profile_result = {
                'profile_id': 'fallback_profile',
                'status': 'created',
                'data': profile_data
            }
        
        session['profile'] = profile_result
        return jsonify({"status": "success", "data": profile_result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/profile', methods=['GET'])
def get_profile():
    """Get user profile data"""
    try:
        profile = session.get('profile', {})
        return jsonify({"status": "success", "data": profile})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Utility Endpoints
@app.route('/api/session/clear', methods=['POST'])
def clear_session():
    """Clear user session data"""
    try:
        session.clear()
        return jsonify({"status": "success", "message": "Session cleared"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Skillora AI Career Advisor",
        "version": "1.0.0"
    })

# Error Handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    import sys
    port = 5001
    if len(sys.argv) > 1 and sys.argv[1].startswith('--port'):
        try:
            port = int(sys.argv[1].split('=')[1]) if '=' in sys.argv[1] else int(sys.argv[2])
        except:
            port = 5001
    
    print(f"🚀 Starting Skillora AI Career Advisor on port {port}")
    app.run(debug=True, host='0.0.0.0', port=port)
