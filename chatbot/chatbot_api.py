from typing import Dict, List, Any, Optional
import re
import random
from datetime import datetime
from .dialog_flow import DialogFlow
from .fallback_adapter import FallbackAdapter

class ChatbotAPI:
    """
    Enhanced chatbot API with ML model integration
    Handles career guidance conversations with context awareness and ML-powered insights
    """
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.dialog_flow = DialogFlow(data_loader)
        self.fallback_adapter = FallbackAdapter(data_loader)
        
        # Load chatbot knowledge base
        self.knowledge_base = data_loader.get_chatbot_knowledge()
        self.career_profiles = data_loader.get_career_profiles()
        self.skills_mapping = data_loader.get_skills_mapping()
        self.market_data = data_loader.get_job_market_data()
        
        # Load ML models for enhanced responses
        self.career_matcher = None
        self.skills_assessor = None
        self.market_analyzer = None
        
        # Conversation state
        self.conversation_context = {}
        
    def set_ml_models(self, career_matcher=None, skills_assessor=None, market_analyzer=None):
        """Set ML models for enhanced responses"""
        self.career_matcher = career_matcher
        self.skills_assessor = skills_assessor
        self.market_analyzer = market_analyzer
        
    def process_message(self, message: str, user_data: Dict = None, session_id: str = "default") -> Dict:
        """
        Process user message and return appropriate response
        """
        try:
            # Initialize session if not exists
            if session_id not in self.conversation_context:
                self.conversation_context[session_id] = {
                    'history': [],
                    'context': {},
                    'user_data': user_data or {},
                    'current_topic': None,
                    'conversation_flow': 'greeting'
                }
            
            # Update user data if provided
            if user_data:
                self.conversation_context[session_id]['user_data'].update(user_data)
            
            # Clean and normalize message
            cleaned_message = self._clean_message(message)
            
            # Add to conversation history
            self.conversation_context[session_id]['history'].append({
                'timestamp': datetime.now().isoformat(),
                'user_message': message,
                'cleaned_message': cleaned_message
            })
            
            # Process message through dialog flow
            response = self.dialog_flow.process_message(
                cleaned_message, 
                self.conversation_context[session_id]
            )
            
            # Enhance response with ML-powered insights if available
            if self._should_use_ml_enhancement(response, cleaned_message):
                enhanced_response = self._enhance_with_ml_insights(
                    response, cleaned_message, self.conversation_context[session_id]
                )
                if enhanced_response:
                    response = enhanced_response
            
            # If dialog flow doesn't handle it, use fallback
            if not response or response.get('confidence', 0) < 0.5:
                response = self.fallback_adapter.handle_message(
                    cleaned_message,
                    self.conversation_context[session_id]
                )
            
            # Add response to history
            self.conversation_context[session_id]['history'][-1]['bot_response'] = response
            
            # Update conversation context
            if response.get('update_context'):
                self.conversation_context[session_id]['context'].update(response['update_context'])
            
            return {
                'response': response.get('message', 'I\'m here to help with your career guidance!'),
                'suggestions': response.get('suggestions', []),
                'actions': response.get('actions', []),
                'confidence': response.get('confidence', 1.0),
                'context': response.get('context', {}),
                'session_id': session_id
            }
            
        except Exception as e:
            return {
                'response': 'I apologize, but I encountered an issue. How can I help you with your career questions?',
                'suggestions': ['Tell me about career options', 'What skills do I need?', 'Job market trends'],
                'actions': [],
                'confidence': 0.8,
                'context': {},
                'session_id': session_id,
                'error': str(e)
            }
    
    def _clean_message(self, message: str) -> str:
        """Clean and normalize user message"""
        if not message:
            return ""
        
        # Convert to lowercase
        cleaned = message.lower().strip()
        
        # Remove extra spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove special characters but keep basic punctuation
        cleaned = re.sub(r'[^\w\s\?\!\.\,\-]', '', cleaned)
        
        return cleaned
    
    def get_conversation_history(self, session_id: str = "default") -> List[Dict]:
        """Get conversation history for a session"""
        if session_id in self.conversation_context:
            return self.conversation_context[session_id]['history']
        return []
    
    def clear_session(self, session_id: str = "default") -> bool:
        """Clear conversation session"""
        if session_id in self.conversation_context:
            del self.conversation_context[session_id]
            return True
        return False
    
    def get_quick_responses(self, user_data: Dict = None) -> List[str]:
        """Get contextual quick response suggestions"""
        suggestions = [
            "What career paths match my profile?",
            "What skills should I develop?",
            "Tell me about job market trends",
            "How to prepare for interviews?",
            "Salary expectations for my field"
        ]
        
        # Personalize based on user data
        if user_data:
            education = user_data.get('education', {})
            if education.get('current_level') == 'student':
                suggestions.insert(0, "What should I study next?")
                suggestions.insert(1, "Best internship opportunities?")
            
            experience = user_data.get('experience', {})
            if experience.get('total_experience', 0) > 0:
                suggestions.insert(0, "How to advance my career?")
                suggestions.insert(1, "Should I change my career path?")
        
        return suggestions[:6]  # Return top 6 suggestions
    
    def _should_use_ml_enhancement(self, response: Dict, message: str) -> bool:
        """Determine if ML enhancement should be used for this response"""
        if not self.career_matcher or not self.skills_assessor:
            return False
        
        # Use ML for career and skills related queries
        ml_keywords = [
            'career', 'job', 'profession', 'skills', 'salary', 'recommendation',
            'suitable', 'match', 'path', 'development', 'learning'
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in ml_keywords)
    
    def _enhance_with_ml_insights(self, response: Dict, message: str, context: Dict) -> Optional[Dict]:
        """Enhance response with ML-powered insights"""
        try:
            user_data = context.get('user_data', {})
            
            # Extract user profile from context
            user_profile = {
                'personality': user_data.get('personality', {}),
                'skills': user_data.get('skills', {}),
                'preferences': user_data.get('preferences', {}),
                'user_data': user_data
            }
            
            # Check if we have enough data for ML analysis
            if not self._has_sufficient_data(user_profile):
                return None
            
            # Generate ML-powered insights based on message intent
            if 'career' in message.lower() or 'job' in message.lower():
                return self._enhance_career_response(response, user_profile)
            elif 'skill' in message.lower():
                return self._enhance_skills_response(response, user_profile)
            elif 'salary' in message.lower():
                return self._enhance_salary_response(response, user_profile)
            
        except Exception as e:
            print(f"ML enhancement failed: {e}")
            return None
        
        return None
    
    def _has_sufficient_data(self, user_profile: Dict) -> bool:
        """Check if user profile has sufficient data for ML analysis"""
        personality = user_profile.get('personality', {})
        skills = user_profile.get('skills', {})
        
        # Need at least personality or skills data
        has_personality = bool(personality.get('personality_profile'))
        has_skills = bool(skills.get('skill_profile'))
        
        return has_personality or has_skills
    
    def _enhance_career_response(self, response: Dict, user_profile: Dict) -> Dict:
        """Enhance career-related responses with ML recommendations"""
        try:
            # Get ML-powered career recommendations
            recommendations = self.career_matcher.get_recommendations(user_profile)
            
            if recommendations:
                # Add top 3 recommendations to response
                top_careers = recommendations[:3]
                career_info = []
                
                for rec in top_careers:
                    career_info.append({
                        'title': rec.get('career_title', 'Unknown'),
                        'score': rec.get('compatibility_score', 0),
                        'description': rec.get('career_description', '')[:100] + '...',
                        'salary': rec.get('salary_range', {}).get('formatted', 'Competitive')
                    })
                
                # Enhance the response
                enhanced_text = response.get('message', '')
                enhanced_text += "\n\n🤖 **AI-Powered Career Recommendations:**\n"
                
                for i, career in enumerate(career_info, 1):
                    enhanced_text += f"{i}. **{career['title']}** ({career['score']:.1f}% match)\n"
                    enhanced_text += f"   💰 Salary: {career['salary']}\n"
                    enhanced_text += f"   📝 {career['description']}\n\n"
                
                response['message'] = enhanced_text
                response['ml_enhanced'] = True
                response['career_recommendations'] = career_info
                
        except Exception as e:
            print(f"Career enhancement failed: {e}")
        
        return response
    
    def _enhance_skills_response(self, response: Dict, user_profile: Dict) -> Dict:
        """Enhance skills-related responses with ML insights"""
        try:
            # Get skills assessment
            skills_data = user_profile.get('skills', {}).get('skill_profile', {})
            if skills_data:
                # Use skills assessor to get skill gaps and recommendations
                skill_gaps = self.skills_assessor._identify_skill_gaps(skills_data, '', '')
                
                if skill_gaps:
                    enhanced_text = response.get('message', '')
                    enhanced_text += "\n\n🎯 **Personalized Skill Recommendations:**\n"
                    
                    for i, gap in enumerate(skill_gaps[:5], 1):
                        skill_name = gap.get('skill', 'Unknown Skill')
                        priority = gap.get('priority', 'Medium')
                        enhanced_text += f"{i}. **{skill_name}** (Priority: {priority})\n"
                    
                    response['message'] = enhanced_text
                    response['ml_enhanced'] = True
                    response['skill_gaps'] = skill_gaps[:5]
                    
        except Exception as e:
            print(f"Skills enhancement failed: {e}")
        
        return response
    
    def _enhance_salary_response(self, response: Dict, user_profile: Dict) -> Dict:
        """Enhance salary-related responses with ML predictions"""
        try:
            # Get career recommendations first
            recommendations = self.career_matcher.get_recommendations(user_profile)
            
            if recommendations:
                # Extract salary information from top recommendations
                salary_info = []
                for rec in recommendations[:3]:
                    salary_range = rec.get('salary_range', {})
                    if salary_range:
                        salary_info.append({
                            'career': rec.get('career_title', 'Unknown'),
                            'salary': salary_range.get('formatted', 'Competitive'),
                            'match': rec.get('compatibility_score', 0)
                        })
                
                if salary_info:
                    enhanced_text = response.get('message', '')
                    enhanced_text += "\n\n💰 **Personalized Salary Insights:**\n"
                    
                    for info in salary_info:
                        enhanced_text += f"• **{info['career']}**: {info['salary']} ({info['match']:.1f}% match)\n"
                    
                    response['message'] = enhanced_text
                    response['ml_enhanced'] = True
                    response['salary_insights'] = salary_info
                    
        except Exception as e:
            print(f"Salary enhancement failed: {e}")
        
        return response
