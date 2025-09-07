from typing import Dict, List, Any, Optional
import re
import random
from datetime import datetime
from .dialog_flow import DialogFlow
from .fallback_adapter import FallbackAdapter

class ChatbotAPI:
    """
    Main chatbot API with zero external dependencies
    Handles career guidance conversations with context awareness
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
        
        # Conversation state
        self.conversation_context = {}
        
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
