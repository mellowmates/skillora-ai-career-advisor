"""
Skillora AI Career Advisor - Rule-Based Chatbot Module
Zero-dependency offline conversational system for career guidance
"""

__version__ = "1.0.0"
__author__ = "Skillora Team"

from .chatbot_api import ChatbotAPI
from .dialog_flow import DialogFlow
from .fallback_adapter import FallbackAdapter

__all__ = ['ChatbotAPI', 'DialogFlow', 'FallbackAdapter']
