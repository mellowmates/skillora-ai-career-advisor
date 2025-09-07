from typing import Dict, List, Any, Optional, Tuple
import re
import random
from datetime import datetime

class DialogFlow:
    """
    Manages conversation flow and context-aware responses
    Handles intent recognition and response generation
    """
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.knowledge_base = data_loader.get_chatbot_knowledge()
        
        # Intent patterns and responses
        self.intent_patterns = self._load_intent_patterns()
        self.response_templates = self._load_response_templates()
        self.context_handlers = self._setup_context_handlers()
        
    def _load_intent_patterns(self) -> Dict:
        """Load intent recognition patterns"""
        return {
            'greeting': {
                'patterns': [
                    r'\b(hello|hi|hey|good morning|good afternoon|good evening|greetings)\b',
                    r'^\s*(hi|hello|hey)\s*$',
                    r'\bstart\b'
                ],
                'confidence': 0.9
            },
            'career_inquiry': {
                'patterns': [
                    r'\b(career|job|profession|occupation|work)\b.*\b(options|choices|paths|suggestions|recommendations)\b',
                    r'\b(what|which).*\b(career|job|profession)\b',
                    r'\b(best|good|suitable).*\b(career|job|field)\b',
                    r'\b(career|job).*\b(for me|suitable|match|fit)\b'
                ],
                'confidence': 0.8
            },
            'skills_inquiry': {
                'patterns': [
                    r'\b(skills?|abilities|competencies)\b.*\b(need|require|develop|learn|improve)\b',
                    r'\b(what|which).*\bskills?\b',
                    r'\bskill.*\b(gap|development|improvement)\b',
                    r'\b(technical|soft).*\bskills?\b'
                ],
                'confidence': 0.8
            },
            'education_inquiry': {
                'patterns': [
                    r'\b(study|education|course|degree|certification)\b',
                    r'\b(college|university|institute)\b.*\b(recommend|suggest|best)\b',
                    r'\b(what|which).*\b(study|course|program)\b',
                    r'\b(should i|need to).*\bstudy\b'
                ],
                'confidence': 0.7
            },
            'salary_inquiry': {
                'patterns': [
                    r'\b(salary|pay|compensation|earnings|income)\b',
                    r'\b(how much|what.*pay|salary range)\b',
                    r'\bmoney.*\b(make|earn)\b',
                    r'\b(salary|pay).*\b(expect|expectation)\b'
                ],
                'confidence': 0.8
            },
            'market_trends': {
                'patterns': [
                    r'\b(market|industry|trends|demand|opportunities)\b',
                    r'\b(job market|employment|hiring)\b',
                    r'\b(future|emerging|growing|hot).*\b(fields|jobs|careers)\b',
                    r'\b(in demand|popular|trending)\b.*\b(skills|jobs)\b'
                ],
                'confidence': 0.7
            },
            'location_inquiry': {
                'patterns': [
                    r'\b(where|location|city|place)\b.*\b(work|jobs|opportunities)\b',
                    r'\b(bangalore|mumbai|delhi|hyderabad|pune|chennai)\b',
                    r'\b(best.*city|top.*location)\b.*\b(jobs|career)\b'
                ],
                'confidence': 0.7
            },
            'interview_prep': {
                'patterns': [
                    r'\b(interview|preparation|prepare)\b',
                    r'\b(how to|tips).*\b(interview|job|position)\b',
                    r'\b(interview|job).*\b(tips|advice|guidance)\b'
                ],
                'confidence': 0.7
            },
            'help': {
                'patterns': [
                    r'\b(help|assist|support|guide|guidance)\b',
                    r'\b(how.*work|what.*do)\b',
                    r'\b(confused|lost|stuck|dont know)\b'
                ],
                'confidence': 0.6
            },
            'thanks': {
                'patterns': [
                    r'\b(thank|thanks|appreciate)\b',
                    r'\b(good|great|excellent|awesome).*\b(help|advice|suggestion)\b'
                ],
                'confidence': 0.9
            },
            'goodbye': {
                'patterns': [
                    r'\b(bye|goodbye|see you|farewell)\b',
                    r'\b(that\'?s all|nothing else|no more questions)\b'
                ],
                'confidence': 0.9
            }
        }
    
    def _load_response_templates(self) -> Dict:
        """Load response templates for different intents"""
        return {
            'greeting': [
                "Hello! I'm here to help guide your career journey. What would you like to know?",
                "Hi there! Welcome to Skillora. How can I assist you with your career planning today?",
                "Greetings! I'm your AI career advisor. What career questions do you have?",
                "Hello! Ready to explore your career options? What interests you most?"
            ],
            'career_inquiry': [
                "I'd love to help you explore career options! To give you the most relevant suggestions, could you tell me about your educational background and interests?",
                "Great question about career paths! What field or industry interests you the most?",
                "Career exploration is exciting! What are your strongest skills and what type of work environment do you prefer?",
                "Let me help you find the perfect career match. What subjects or activities do you enjoy most?"
            ],
            'skills_inquiry': [
                "Skills development is crucial for career growth! What specific area would you like to focus on - technical skills, soft skills, or domain expertise?",
                "Excellent focus on skill building! Which career field are you targeting so I can suggest the most relevant skills?",
                "Smart thinking about skills! Are you looking to develop new skills or strengthen existing ones?",
                "Great question about skills! What's your current experience level and which direction interests you?"
            ],
            'education_inquiry': [
                "Education planning is important! What's your current educational level and what career field interests you?",
                "Let me help with educational guidance! Are you looking for degree programs, certifications, or skill-specific courses?",
                "Education choices matter for your career! What's your preferred learning style and timeline?",
                "Good question about education! Which subjects have you enjoyed most in your studies so far?"
            ],
            'salary_inquiry': [
                "Salary expectations are important for career planning! Which role or field are you curious about?",
                "Let me help with salary insights! What's your experience level and which location interests you?",
                "Great question about compensation! Are you looking at entry-level, mid-level, or senior positions?",
                "Salary planning is smart! Which industry or job role would you like salary information for?"
            ],
            'market_trends': [
                "The job market is constantly evolving! Which sector or skill area would you like trend information about?",
                "Market trends are fascinating! Are you interested in emerging technologies, traditional fields, or specific industries?",
                "Great interest in market intelligence! What type of opportunities are you most curious about?",
                "The future job market has exciting opportunities! Which skills or fields would you like to explore?"
            ],
            'location_inquiry': [
                "Location choice affects career opportunities! Which cities or regions are you considering?",
                "Great question about work locations! Are you interested in tech hubs, traditional business centers, or emerging cities?",
                "Location planning is smart! What type of work environment and lifestyle do you prefer?",
                "Different cities offer unique opportunities! What factors matter most in your location choice?"
            ],
            'interview_prep': [
                "Interview preparation is crucial for success! What type of role or interview format are you preparing for?",
                "Great focus on interview skills! Are you looking for general tips or specific to a particular field?",
                "Interview prep can make all the difference! What aspect of interviews would you like help with?",
                "Smart preparation approach! What's your biggest concern about the interview process?"
            ],
            'help': [
                "I'm here to help with all your career questions! You can ask about career paths, skills, education, salary, market trends, and more.",
                "Happy to guide you! Try asking about career options for your field, skill development plans, or job market insights.",
                "Let me assist you! I can help with career matching, educational guidance, skill gap analysis, and market intelligence.",
                "I'm your career guidance companion! Ask me about anything related to career planning, skills, or professional development."
            ],
            'thanks': [
                "You're very welcome! Feel free to ask more questions anytime.",
                "Happy to help! Is there anything else about your career journey you'd like to explore?",
                "Glad I could assist! Do you have any other career-related questions?",
                "You're welcome! I'm here whenever you need career guidance."
            ],
            'goodbye': [
                "Goodbye! Best of luck with your career journey. Feel free to return anytime for more guidance!",
                "Take care! Remember, I'm here whenever you need career advice. Wishing you success!",
                "Farewell! Keep exploring and growing in your career. Come back anytime for more insights!",
                "See you later! Your career journey is unique - embrace the opportunities ahead!"
            ],
            'fallback': [
                "That's an interesting question! Could you provide more details so I can give you the most helpful career guidance?",
                "I want to make sure I understand correctly. Could you rephrase your question or provide more context?",
                "Let me help you better! What specific aspect of career planning or development are you most interested in?",
                "I'm here to provide the best career guidance possible. Could you be more specific about what you'd like to know?"
            ]
        }
    
    def _setup_context_handlers(self) -> Dict:
        """Setup context-aware response handlers"""
        return {
            'follow_up_career': self._handle_career_followup,
            'follow_up_skills': self._handle_skills_followup,
            'follow_up_education': self._handle_education_followup,
            'follow_up_salary': self._handle_salary_followup,
            'follow_up_market': self._handle_market_followup
        }
    
    def process_message(self, message: str, conversation_context: Dict) -> Dict:
        """Process message and return appropriate response"""
        # Detect intent
        intent, confidence = self._detect_intent(message)
        
        # Get user data from context
        user_data = conversation_context.get('user_data', {})
        
        # Generate response based on intent and context
        response = self._generate_response(intent, message, user_data, conversation_context)
        
        return {
            'message': response['text'],
            'suggestions': response['suggestions'],
            'actions': response['actions'],
            'confidence': confidence,
            'intent': intent,
            'context': response['context'],
            'update_context': response['update_context']
        }
    
    def _detect_intent(self, message: str) -> Tuple[str, float]:
        """Detect user intent from message"""
        best_intent = 'fallback'
        best_confidence = 0.0
        
        for intent, data in self.intent_patterns.items():
            for pattern in data['patterns']:
                if re.search(pattern, message, re.IGNORECASE):
                    confidence = data['confidence']
                    if confidence > best_confidence:
                        best_intent = intent
                        best_confidence = confidence
        
        return best_intent, best_confidence
    
    def _generate_response(self, intent: str, message: str, user_data: Dict, context: Dict) -> Dict:
        """Generate contextual response"""
        # Get base response template
        templates = self.response_templates.get(intent, self.response_templates['fallback'])
        base_response = random.choice(templates)
        
        # Personalize response based on user data
        personalized_response = self._personalize_response(base_response, user_data, intent)
        
        # Generate suggestions and actions
        suggestions = self._generate_suggestions(intent, user_data)
        actions = self._generate_actions(intent, user_data)
        
        # Update context
        update_context = {}
        if intent in ['career_inquiry', 'skills_inquiry', 'education_inquiry', 'salary_inquiry', 'market_trends']:
            update_context['last_intent'] = intent
            update_context['awaiting_followup'] = True
        
        return {
            'text': personalized_response,
            'suggestions': suggestions,
            'actions': actions,
            'context': {'intent': intent, 'personalized': bool(user_data)},
            'update_context': update_context
        }
    
    def _personalize_response(self, base_response: str, user_data: Dict, intent: str) -> str:
        """Personalize response based on user profile"""
        if not user_data:
            return base_response
        
        # Add personalization based on education level
        education = user_data.get('education', {})
        current_level = education.get('current_level', '')
        
        if current_level and intent == 'career_inquiry':
            if 'student' in current_level.lower():
                base_response += " As a student, you have great flexibility to explore different paths!"
            elif any(degree in current_level.lower() for degree in ['b.tech', 'b.e.', 'engineering']):
                base_response += " With your engineering background, you have excellent analytical skills!"
        
        # Add personalization based on experience
        experience = user_data.get('experience', {})
        total_exp = experience.get('total_experience', 0)
        
        if total_exp > 0 and intent == 'skills_inquiry':
            base_response += f" With {total_exp} years of experience, you're in a great position to advance your skills!"
        
        return base_response
    
    def _generate_suggestions(self, intent: str, user_data: Dict) -> List[str]:
        """Generate contextual suggestions"""
        suggestions_map = {
            'greeting': [
                "Explore career options",
                "Assess my skills",
                "Job market trends",
                "Salary insights"
            ],
            'career_inquiry': [
                "Tell me about specific roles",
                "What skills do I need?",
                "Growth opportunities",
                "Salary expectations"
            ],
            'skills_inquiry': [
                "Technical skills roadmap",
                "Soft skills development",
                "Certification recommendations",
                "Learning resources"
            ],
            'education_inquiry': [
                "Degree recommendations",
                "Online courses",
                "Certification programs",
                "Skill-specific training"
            ],
            'salary_inquiry': [
                "Entry-level salaries",
                "Experience-based pay",
                "Location comparison",
                "Negotiation tips"
            ],
            'market_trends': [
                "Emerging technologies",
                "High-demand skills",
                "Industry growth",
                "Future opportunities"
            ]
        }
        
        return suggestions_map.get(intent, [
            "Career guidance",
            "Skills development",
            "Market insights",
            "Educational paths"
        ])
    
    def _generate_actions(self, intent: str, user_data: Dict) -> List[Dict]:
        """Generate actionable items"""
        actions = []
        
        if intent == 'career_inquiry':
            actions.append({
                'type': 'assessment',
                'label': 'Take Career Assessment',
                'action': 'start_assessment'
            })
        
        if intent == 'skills_inquiry':
            actions.append({
                'type': 'analysis',
                'label': 'Analyze Skill Gaps',
                'action': 'skill_analysis'
            })
        
        if intent in ['education_inquiry', 'skills_inquiry']:
            actions.append({
                'type': 'roadmap',
                'label': 'Create Learning Plan',
                'action': 'create_roadmap'
            })
        
        return actions
    
    # Context-aware follow-up handlers
    def _handle_career_followup(self, message: str, context: Dict) -> Dict:
        """Handle career-related follow-up questions"""
        return {'response': 'Let me provide more specific career guidance...'}
    
    def _handle_skills_followup(self, message: str, context: Dict) -> Dict:
        """Handle skills-related follow-up questions"""
        return {'response': 'Here are detailed skill recommendations...'}
    
    def _handle_education_followup(self, message: str, context: Dict) -> Dict:
        """Handle education-related follow-up questions"""
        return {'response': 'Let me suggest educational pathways...'}
    
    def _handle_salary_followup(self, message: str, context: Dict) -> Dict:
        """Handle salary-related follow-up questions"""
        return {'response': 'Here are salary insights for your situation...'}
    
    def _handle_market_followup(self, message: str, context: Dict) -> Dict:
        """Handle market trends follow-up questions"""
        return {'response': 'Current market analysis shows...'}
