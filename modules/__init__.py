# Skillora AI Career Advisor Modules
# This package contains all the core modules for the career advisor application

__version__ = "1.0.0"
__author__ = "Skillora Team"

from .personality_analyzer import PersonalityAnalyzer
from .skills_assessor import SkillsAssessor
from .job_market_scraper import JobMarketScraper
from .career_matcher import CareerMatcher

__all__ = [
    'PersonalityAnalyzer',
    'SkillsAssessor', 
    'JobMarketScraper',
    'CareerMatcher'
]
