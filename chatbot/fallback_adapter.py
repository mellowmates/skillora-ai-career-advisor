from typing import Dict, List, Any, Optional
import re
import random

class FallbackAdapter:
    """
    Handles edge cases, unusual educational backgrounds, and fallback responses
    Ensures the chatbot always provides helpful guidance
    """
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.education_paths = data_loader.get_education_paths()
        self.career_profiles = data_loader.get_career_profiles()
        self.skills_mapping = data_loader.get_skills_mapping()
        
        # Specialized handlers
        self.education_patterns = self._setup_education_patterns()
        self.fallback_responses = self._setup_fallback_responses()
        self.clarification_questions = self._setup_clarification_questions()
    
    def _setup_education_patterns(self) -> Dict:
        """Setup patterns for unusual educational backgrounds"""
        return {
            'professional_courses': {
                'patterns': [
                    r'\b(ca|chartered accountant|cs|company secretary|cma|icwa)\b',
                    r'\b(law|llb|legal|advocate|lawyer)\b',
                    r'\b(medical|mbbs|doctor|medicine|nursing)\b',
                    r'\b(pharmacy|b.pharma|pharma)\b'
                ],
                'handler': self._handle_professional_courses
            },
            'vocational_training': {
                'patterns': [
                    r'\b(iti|polytechnic|diploma)\b',
                    r'\b(vocational|trade|technical training)\b',
                    r'\b(skill development|apprenticeship)\b'
                ],
                'handler': self._handle_vocational_training
            },
            'alternative_education': {
                'patterns': [
                    r'\b(self taught|autodidact|online learning)\b',
                    r'\b(bootcamp|coding bootcamp|intensive course)\b',
                    r'\b(certification|certified|online certification)\b'
                ],
                'handler': self._handle_alternative_education
            },
            'international_education': {
                'patterns': [
                    r'\b(foreign|international|abroad|overseas)\b.*\b(degree|education|study)\b',
                    r'\b(uk|us|usa|canada|australia|germany)\b.*\b(degree|university)\b'
                ],
                'handler': self._handle_international_education
            },
            'career_transition': {
                'patterns': [
                    r'\b(career change|transition|switch|pivot)\b',
                    r'\b(different field|new career|change direction)\b',
                    r'\b(mid career|career shift)\b'
                ],
                'handler': self._handle_career_transition
            }
        }
    
    def _setup_fallback_responses(self) -> List[str]:
        """Setup general fallback responses"""
        return [
            "I understand you're looking for career guidance. Let me help you explore your options step by step.",
            "That's an interesting situation! Every career journey is unique. Let's work together to find the best path for you.",
            "Career decisions can be complex, but I'm here to help simplify them. What's your main priority right now?",
            "I appreciate you sharing that with me. Let's focus on what matters most for your career growth.",
            "Your career path doesn't have to follow a traditional route. Let's explore what works best for your situation.",
            "Sometimes the best careers come from unconventional paths. Tell me more about your interests and goals.",
            "I'm here to provide personalized guidance regardless of your background. What would you like to focus on first?"
        ]
    
    def _setup_clarification_questions(self) -> Dict:
        """Setup clarification questions for different scenarios"""
        return {
            'vague_career': [
                "What type of work environment appeals to you most?",
                "Are you more interested in creative, analytical, or people-focused roles?",
                "What subjects or activities have you enjoyed most in the past?",
                "Do you prefer working independently or as part of a team?"
            ],
            'skill_confusion': [
                "What tasks or projects have you found most engaging?",
                "Which of your abilities do others often compliment?",
                "What type of problems do you enjoy solving?",
                "Are there any tools or technologies you're curious about?"
            ],
            'education_uncertainty': [
                "What's your current educational level?",
                "Are you interested in formal degrees, certifications, or hands-on training?",
                "How much time can you dedicate to learning?",
                "Do you prefer structured courses or self-paced learning?"
            ],
            'general_confusion': [
                "What brings you the most satisfaction in your work or studies?",
                "If you could solve any problem in the world, what would it be?",
                "What does a successful career look like to you?",
                "Are you looking for immediate opportunities or long-term planning?"
            ]
        }
    
    def handle_message(self, message: str, conversation_context: Dict) -> Dict:
        """Main fallback handler for difficult or unclear messages"""
        user_data = conversation_context.get('user_data', {})
        
        # Try specialized handlers first
        for category, data in self.education_patterns.items():
            for pattern in data['patterns']:
                if re.search(pattern, message, re.IGNORECASE):
                    return data['handler'](message, user_data, conversation_context)
        
        # Check for specific career/education terms
        if self._contains_career_terms(message):
            return self._handle_career_related(message, user_data)
        
        if self._contains_education_terms(message):
            return self._handle_education_related(message, user_data)
        
        if self._contains_skill_terms(message):
            return self._handle_skill_related(message, user_data)
        
        # General fallback with smart clarification
        return self._generate_smart_fallback(message, user_data, conversation_context)
    
    def _handle_professional_courses(self, message: str, user_data: Dict, context: Dict) -> Dict:
        """Handle professional courses like CA, CS, Law, Medical"""
        course_type = self._extract_professional_course(message)
        
        responses = {
            'ca': "Chartered Accountancy is an excellent professional path! CAs are in high demand across finance, consulting, and corporate roles. You can work in public practice, industry, or even start your own firm.",
            'cs': "Company Secretary is a specialized and valuable profession! CS professionals are essential for corporate governance and compliance. Great opportunities in corporates and consultancy.",
            'law': "Legal profession offers diverse career paths! You can specialize in corporate law, litigation, legal consulting, or even combine law with technology for legal-tech roles.",
            'medical': "Medical profession is noble and rewarding! Beyond clinical practice, you can explore medical research, healthcare management, medical technology, or public health."
        }
        
        base_response = responses.get(course_type, "Professional courses provide specialized expertise that's highly valued in the market!")
        
        suggestions = [
            f"Explore {course_type.upper()} career specializations",
            "Industry opportunities and growth",
            "Skill development recommendations",
            "Salary and career progression"
        ]
        
        return {
            'message': base_response + " Would you like to explore specific opportunities in your field?",
            'suggestions': suggestions,
            'actions': [
                {'type': 'specialization', 'label': 'Explore Specializations', 'action': 'show_specializations'},
                {'type': 'roadmap', 'label': 'Career Roadmap', 'action': 'create_professional_roadmap'}
            ],
            'confidence': 0.8,
            'context': {'professional_course': course_type}
        }
    
    def _handle_vocational_training(self, message: str, user_data: Dict, context: Dict) -> Dict:
        """Handle vocational and technical training backgrounds"""
        response = """Vocational and technical training provides practical, job-ready skills! These programs often lead directly to employment and offer excellent career growth opportunities. Many vocational graduates become entrepreneurs or advance to supervisory and management roles."""
        
        suggestions = [
            "Technical career advancement",
            "Skill certification programs",
            "Entrepreneurship opportunities",
            "Higher education pathways"
        ]
        
        return {
            'message': response + " What specific trade or technical area interests you most?",
            'suggestions': suggestions,
            'actions': [
                {'type': 'advancement', 'label': 'Career Advancement', 'action': 'vocational_advancement'},
                {'type': 'certification', 'label': 'Skill Certifications', 'action': 'suggest_certifications'}
            ],
            'confidence': 0.8,
            'context': {'education_type': 'vocational'}
        }
    
    def _handle_alternative_education(self, message: str, user_data: Dict, context: Dict) -> Dict:
        """Handle self-taught, bootcamp, and alternative education paths"""
        response = """Alternative education paths are increasingly valued by employers! Self-taught skills, bootcamps, and online certifications demonstrate initiative, dedication, and practical abilities. Many successful professionals have non-traditional educational backgrounds."""
        
        suggestions = [
            "Portfolio building strategies",
            "Industry recognition tips",
            "Networking and community building",
            "Continuous learning pathways"
        ]
        
        return {
            'message': response + " What skills have you developed through your alternative learning path?",
            'suggestions': suggestions,
            'actions': [
                {'type': 'portfolio', 'label': 'Build Portfolio', 'action': 'portfolio_guidance'},
                {'type': 'networking', 'label': 'Networking Tips', 'action': 'networking_advice'}
            ],
            'confidence': 0.8,
            'context': {'education_type': 'alternative'}
        }
    
    def _handle_international_education(self, message: str, user_data: Dict, context: Dict) -> Dict:
        """Handle international education backgrounds"""
        response = """International education brings valuable global perspective and cross-cultural skills! These experiences are highly valued by multinational companies and global organizations. Your international background can be a significant career advantage."""
        
        suggestions = [
            "Leverage global experience",
            "MNC opportunities in India",
            "International career paths",
            "Cross-cultural skills value"
        ]
        
        return {
            'message': response + " How would you like to leverage your international education experience?",
            'suggestions': suggestions,
            'actions': [
                {'type': 'global', 'label': 'Global Opportunities', 'action': 'international_careers'},
                {'type': 'mnc', 'label': 'MNC Roles', 'action': 'multinational_jobs'}
            ],
            'confidence': 0.8,
            'context': {'education_type': 'international'}
        }
    
    def _handle_career_transition(self, message: str, user_data: Dict, context: Dict) -> Dict:
        """Handle career change and transition scenarios"""
        response = """Career transitions are common and can be very rewarding! Many successful professionals have changed careers to find better alignment with their interests, values, or life goals. The key is leveraging transferable skills and strategic planning."""
        
        suggestions = [
            "Transferable skills analysis",
            "Transition planning strategies",
            "Industry switching tips",
            "Networking for career change"
        ]
        
        return {
            'message': response + " What's motivating your interest in career transition?",
            'suggestions': suggestions,
            'actions': [
                {'type': 'transition', 'label': 'Transition Plan', 'action': 'career_transition_plan'},
                {'type': 'skills', 'label': 'Transferable Skills', 'action': 'identify_transferable_skills'}
            ],
            'confidence': 0.8,
            'context': {'scenario': 'career_transition'}
        }
    
    def _handle_career_related(self, message: str, user_data: Dict) -> Dict:
        """Handle career-related messages that don't fit standard patterns"""
        response = random.choice([
            "Career planning is important! Let me help you explore opportunities that match your interests and goals.",
            "Every career journey is unique. What aspects of career development are most important to you right now?",
            "I'd love to help with your career questions! Could you tell me more about what you're looking for?"
        ])
        
        clarifications = random.sample(self.clarification_questions['vague_career'], 2)
        
        return {
            'message': response,
            'suggestions': clarifications + ["Tell me about your background", "Explore career options"],
            'actions': [
                {'type': 'assessment', 'label': 'Career Assessment', 'action': 'start_assessment'}
            ],
            'confidence': 0.6,
            'context': {'needs_clarification': True}
        }
    
    def _handle_education_related(self, message: str, user_data: Dict) -> Dict:
        """Handle education-related messages"""
        response = random.choice([
            "Education choices shape career opportunities! What type of educational path interests you most?",
            "There are many educational routes to achieve your career goals. Let me help you find the best fit.",
            "Educational planning is crucial for career success. What are you most curious about?"
        ])
        
        clarifications = random.sample(self.clarification_questions['education_uncertainty'], 2)
        
        return {
            'message': response,
            'suggestions': clarifications + ["Degree recommendations", "Certification options"],
            'actions': [
                {'type': 'education', 'label': 'Educational Pathways', 'action': 'education_guidance'}
            ],
            'confidence': 0.6,
            'context': {'topic': 'education'}
        }
    
    def _handle_skill_related(self, message: str, user_data: Dict) -> Dict:
        """Handle skill-related messages"""
        response = random.choice([
            "Skills are the foundation of career success! What skills would you like to develop or strengthen?",
            "Great focus on skill development! Let me help you identify the most valuable skills for your goals.",
            "Skill building is always a smart investment. What areas interest you most?"
        ])
        
        clarifications = random.sample(self.clarification_questions['skill_confusion'], 2)
        
        return {
            'message': response,
            'suggestions': clarifications + ["Technical skills", "Soft skills"],
            'actions': [
                {'type': 'skills', 'label': 'Skill Analysis', 'action': 'skill_assessment'}
            ],
            'confidence': 0.6,
            'context': {'topic': 'skills'}
        }
    
    def _generate_smart_fallback(self, message: str, user_data: Dict, context: Dict) -> Dict:
        """Generate intelligent fallback response"""
        # Try to extract any meaningful information from the message
        extracted_info = self._extract_information(message)
        
        base_response = random.choice(self.fallback_responses)
        
        # Add context-specific guidance if user data available
        if user_data:
            education = user_data.get('education', {})
            if education.get('current_level'):
                base_response += f" I see you're at the {education['current_level']} level - that's a great starting point!"
        
        # Generate helpful clarification questions
        clarifications = random.sample(self.clarification_questions['general_confusion'], 2)
        
        return {
            'message': base_response,
            'suggestions': clarifications + [
                "Help me understand your goals",
                "Start with career exploration",
                "Tell me about your interests"
            ],
            'actions': [
                {'type': 'help', 'label': 'Getting Started Guide', 'action': 'show_help'},
                {'type': 'assessment', 'label': 'Quick Assessment', 'action': 'quick_assessment'}
            ],
            'confidence': 0.5,
            'context': {'fallback_used': True, 'extracted_info': extracted_info}
        }
    
    # Utility methods
    def _extract_professional_course(self, message: str) -> str:
        """Extract professional course type from message"""
        if re.search(r'\b(ca|chartered accountant)\b', message, re.IGNORECASE):
            return 'ca'
        elif re.search(r'\b(cs|company secretary)\b', message, re.IGNORECASE):
            return 'cs'
        elif re.search(r'\b(law|llb|legal)\b', message, re.IGNORECASE):
            return 'law'
        elif re.search(r'\b(medical|mbbs|doctor)\b', message, re.IGNORECASE):
            return 'medical'
        return 'professional'
    
    def _contains_career_terms(self, message: str) -> bool:
        """Check if message contains career-related terms"""
        career_terms = ['job', 'career', 'profession', 'work', 'employment', 'role', 'position', 'occupation']
        return any(term in message.lower() for term in career_terms)
    
    def _contains_education_terms(self, message: str) -> bool:
        """Check if message contains education-related terms"""
        education_terms = ['study', 'course', 'degree', 'college', 'university', 'education', 'learning', 'certification']
        return any(term in message.lower() for term in education_terms)
    
    def _contains_skill_terms(self, message: str) -> bool:
        """Check if message contains skill-related terms"""
        skill_terms = ['skill', 'ability', 'competency', 'expertise', 'talent', 'knowledge', 'capability']
        return any(term in message.lower() for term in skill_terms)
    
    def _extract_information(self, message: str) -> Dict:
        """Extract any useful information from unclear messages"""
        info = {}
        
        # Extract numbers (could be experience, age, etc.)
        numbers = re.findall(r'\b\d+\b', message)
        if numbers:
            info['numbers'] = numbers
        
        # Extract location mentions
        indian_cities = ['bangalore', 'mumbai', 'delhi', 'hyderabad', 'pune', 'chennai', 'kolkata', 'ahmedabad']
        for city in indian_cities:
            if city in message.lower():
                info['location'] = city
                break
        
        # Extract technology mentions
        tech_terms = ['python', 'java', 'javascript', 'ai', 'ml', 'data', 'web', 'mobile', 'cloud']
        mentioned_tech = [tech for tech in tech_terms if tech in message.lower()]
        if mentioned_tech:
            info['technologies'] = mentioned_tech
        
        return info
