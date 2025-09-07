"""
Configuration settings for Skillora AI Career Advisor
"""

import os
from datetime import timedelta

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'skillora_hackathon_2025_secret_key'
    
    # Data paths
    DATA_DIR = 'data'
    MODELS_DIR = 'models'
    STATIC_DIR = 'static'
    TEMPLATES_DIR = 'templates'
    
    # Session configuration
    PERMANENT_SESSION_LIFETIME = timedelta(hours=2)
    
    # ML Model settings
    ML_ENABLED = True
    ML_FALLBACK_ENABLED = True
    
    # Career matching weights
    PERSONALITY_WEIGHT = 0.40
    SKILLS_WEIGHT = 0.35
    MARKET_WEIGHT = 0.25
    
    # Hybrid scoring weights
    ML_WEIGHT = 0.70
    RULE_WEIGHT = 0.30
    
    # Assessment settings
    PERSONALITY_QUESTIONS_COUNT = 15
    SKILLS_ASSESSMENT_CATEGORIES = 6
    
    # Recommendation settings
    TOP_RECOMMENDATIONS = 5
    MAX_SKILL_GAPS = 10
    
    # Chatbot settings
    CHATBOT_CONTEXT_LENGTH = 10
    CHATBOT_FALLBACK_RESPONSES = True
    
    # Market analysis settings
    MARKET_FORECAST_YEARS = 5
    SALARY_CONFIDENCE_INTERVAL = 0.8
    
    # Learning roadmap settings
    DEFAULT_HOURS_PER_WEEK = 10
    MIN_PHASE_DURATION_WEEKS = 2
    MAX_ROADMAP_DURATION_WEEKS = 52

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    
class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
