"""
Career Matcher Module
Matches users with suitable career paths based on personality and skills
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pandas as pd

class CareerMatcher:
    def __init__(self):
        """Initialize the career matcher"""
        self.career_profiles = {}
        self.skills_mapping = {}
        self.personality_weights = {
            'openness': 0.2,
            'conscientiousness': 0.2,
            'extraversion': 0.2,
            'agreeableness': 0.2,
            'neuroticism': 0.2
        }
        self.skill_weights = {
            'technical': 0.4,
            'soft': 0.3,
            'domain': 0.2,
            'tools': 0.1
        }
    
    def load_data(self, career_profiles_path: str = "data/career_profiles.json",
                  skills_mapping_path: str = "data/skills_mapping.json") -> bool:
        """
        Load career profiles and skills mapping data
        
        Args:
            career_profiles_path: Path to career profiles JSON file
            skills_mapping_path: Path to skills mapping JSON file
            
        Returns:
            Boolean indicating success
        """
        try:
            # Load career profiles
            with open(career_profiles_path, 'r') as f:
                self.career_profiles = json.load(f)
            
            # Load skills mapping
            with open(skills_mapping_path, 'r') as f:
                self.skills_mapping = json.load(f)
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def match_career(self, personality_profile: Dict[str, Any], 
                    skills_profile: Dict[str, Any],
                    preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Match user with suitable career paths
        
        Args:
            personality_profile: User's personality analysis results
            skills_profile: User's skills assessment results
            preferences: User's career preferences
            
        Returns:
            Dictionary with career matching results
        """
        try:
            if not self.career_profiles or not self.skills_mapping:
                self.load_data()
            
            # Calculate compatibility scores for each career
            career_scores = self._calculate_career_scores(
                personality_profile, skills_profile, preferences
            )
            
            # Rank careers by compatibility
            ranked_careers = self._rank_careers(career_scores)
            
            # Generate detailed recommendations
            recommendations = self._generate_recommendations(
                ranked_careers, personality_profile, skills_profile
            )
            
            return {
                'success': True,
                'top_careers': ranked_careers[:10],
                'recommendations': recommendations,
                'match_analysis': self._analyze_matches(ranked_careers)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'top_careers': [],
                'recommendations': []
            }
    
    def _calculate_career_scores(self, personality_profile: Dict[str, Any],
                               skills_profile: Dict[str, Any],
                               preferences: Dict[str, Any] = None) -> Dict[str, float]:
        """Calculate compatibility scores for each career"""
        career_scores = {}
        
        for career_id, career_data in self.career_profiles.items():
            # Personality compatibility score
            personality_score = self._calculate_personality_compatibility(
                personality_profile, career_data
            )
            
            # Skills compatibility score
            skills_score = self._calculate_skills_compatibility(
                skills_profile, career_data
            )
            
            # Preferences compatibility score
            preferences_score = self._calculate_preferences_compatibility(
                preferences, career_data
            ) if preferences else 0.5
            
            # Weighted total score
            total_score = (
                personality_score * 0.4 +
                skills_score * 0.4 +
                preferences_score * 0.2
            )
            
            career_scores[career_id] = {
                'total_score': total_score,
                'personality_score': personality_score,
                'skills_score': skills_score,
                'preferences_score': preferences_score,
                'career_data': career_data
            }
        
        return career_scores
    
    def _calculate_personality_compatibility(self, personality_profile: Dict[str, Any],
                                          career_data: Dict[str, Any]) -> float:
        """Calculate personality compatibility score"""
        try:
            user_traits = personality_profile.get('big_five_scores', {})
            career_traits = career_data.get('ideal_personality', {})
            
            if not user_traits or not career_traits:
                return 0.5  # Neutral score if data is missing
            
            compatibility_scores = []
            
            for trait in self.personality_weights.keys():
                user_score = user_traits.get(trait, 50) / 100  # Normalize to 0-1
                career_ideal = career_traits.get(trait, 50) / 100
                
                # Calculate compatibility (closer values = higher compatibility)
                compatibility = 1 - abs(user_score - career_ideal)
                compatibility_scores.append(compatibility)
            
            return np.mean(compatibility_scores)
            
        except Exception as e:
            print(f"Error calculating personality compatibility: {str(e)}")
            return 0.5
    
    def _calculate_skills_compatibility(self, skills_profile: Dict[str, Any],
                                     career_data: Dict[str, Any]) -> float:
        """Calculate skills compatibility score"""
        try:
            user_skills = skills_profile.get('skills', {})
            required_skills = career_data.get('required_skills', [])
            preferred_skills = career_data.get('preferred_skills', [])
            
            if not user_skills or not required_skills:
                return 0.5
            
            # Calculate required skills match
            required_matches = 0
            for skill in required_skills:
                if skill in user_skills:
                    level = user_skills[skill]
                    level_scores = {'beginner': 0.25, 'intermediate': 0.5, 
                                  'advanced': 0.75, 'expert': 1.0}
                    required_matches += level_scores.get(level, 0.25)
            
            required_score = required_matches / len(required_skills) if required_skills else 0
            
            # Calculate preferred skills bonus
            preferred_matches = 0
            for skill in preferred_skills:
                if skill in user_skills:
                    level = user_skills[skill]
                    level_scores = {'beginner': 0.1, 'intermediate': 0.2, 
                                  'advanced': 0.3, 'expert': 0.4}
                    preferred_matches += level_scores.get(level, 0.1)
            
            preferred_bonus = preferred_matches / len(preferred_skills) if preferred_skills else 0
            
            # Combine scores (required skills are more important)
            total_score = (required_score * 0.8) + (preferred_bonus * 0.2)
            
            return min(total_score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            print(f"Error calculating skills compatibility: {str(e)}")
            return 0.5
    
    def _calculate_preferences_compatibility(self, preferences: Dict[str, Any],
                                          career_data: Dict[str, Any]) -> float:
        """Calculate preferences compatibility score"""
        try:
            if not preferences:
                return 0.5
            
            compatibility_factors = []
            
            # Work environment preference
            preferred_env = preferences.get('work_environment', '')
            career_env = career_data.get('work_environment', '')
            if preferred_env and career_env:
                env_match = 1.0 if preferred_env.lower() in career_env.lower() else 0.5
                compatibility_factors.append(env_match)
            
            # Salary expectation
            expected_salary = preferences.get('salary_expectation', 0)
            career_salary_range = career_data.get('salary_range', {})
            if expected_salary and career_salary_range:
                min_salary = career_salary_range.get('min', 0)
                max_salary = career_salary_range.get('max', 0)
                if min_salary <= expected_salary <= max_salary:
                    salary_match = 1.0
                elif expected_salary < min_salary:
                    salary_match = 0.3  # Below range
                else:
                    salary_match = 0.7  # Above range
                compatibility_factors.append(salary_match)
            
            # Work-life balance
            preferred_balance = preferences.get('work_life_balance', '')
            career_balance = career_data.get('work_life_balance', '')
            if preferred_balance and career_balance:
                balance_match = 1.0 if preferred_balance.lower() == career_balance.lower() else 0.5
                compatibility_factors.append(balance_match)
            
            return np.mean(compatibility_factors) if compatibility_factors else 0.5
            
        except Exception as e:
            print(f"Error calculating preferences compatibility: {str(e)}")
            return 0.5
    
    def _rank_careers(self, career_scores: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank careers by compatibility score"""
        ranked_careers = []
        
        for career_id, scores in career_scores.items():
            career_info = {
                'career_id': career_id,
                'title': scores['career_data'].get('title', 'Unknown'),
                'total_score': scores['total_score'],
                'personality_score': scores['personality_score'],
                'skills_score': scores['skills_score'],
                'preferences_score': scores['preferences_score'],
                'description': scores['career_data'].get('description', ''),
                'growth_outlook': scores['career_data'].get('growth_outlook', ''),
                'salary_range': scores['career_data'].get('salary_range', {}),
                'required_skills': scores['career_data'].get('required_skills', []),
                'education_requirements': scores['career_data'].get('education_requirements', ''),
                'work_environment': scores['career_data'].get('work_environment', '')
            }
            ranked_careers.append(career_info)
        
        # Sort by total score
        ranked_careers.sort(key=lambda x: x['total_score'], reverse=True)
        
        return ranked_careers
    
    def _generate_recommendations(self, ranked_careers: List[Dict[str, Any]],
                                personality_profile: Dict[str, Any],
                                skills_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed recommendations based on top career matches"""
        if not ranked_careers:
            return {'general': [], 'skill_development': [], 'personality_insights': []}
        
        top_career = ranked_careers[0]
        
        # General recommendations
        general_recommendations = [
            f"Based on your profile, {top_career['title']} appears to be an excellent match",
            f"Your compatibility score is {top_career['total_score']:.2f} out of 1.0",
            f"This career offers {top_career['growth_outlook']} growth prospects"
        ]
        
        # Skill development recommendations
        skill_recommendations = self._generate_skill_recommendations(
            top_career, skills_profile
        )
        
        # Personality insights
        personality_insights = self._generate_personality_insights(
            top_career, personality_profile
        )
        
        return {
            'general': general_recommendations,
            'skill_development': skill_recommendations,
            'personality_insights': personality_insights,
            'alternative_careers': ranked_careers[1:5]  # Next 4 best matches
        }
    
    def _generate_skill_recommendations(self, career: Dict[str, Any],
                                      skills_profile: Dict[str, Any]) -> List[str]:
        """Generate skill development recommendations"""
        recommendations = []
        user_skills = skills_profile.get('skills', {})
        required_skills = career.get('required_skills', [])
        
        missing_skills = []
        weak_skills = []
        
        for skill in required_skills:
            if skill not in user_skills:
                missing_skills.append(skill)
            elif user_skills[skill] in ['beginner', 'intermediate']:
                weak_skills.append(skill)
        
        if missing_skills:
            recommendations.append(f"Consider developing these essential skills: {', '.join(missing_skills[:3])}")
        
        if weak_skills:
            recommendations.append(f"Strengthen your skills in: {', '.join(weak_skills[:3])}")
        
        if not missing_skills and not weak_skills:
            recommendations.append("Your current skills align well with this career path!")
        
        return recommendations
    
    def _generate_personality_insights(self, career: Dict[str, Any],
                                     personality_profile: Dict[str, Any]) -> List[str]:
        """Generate personality-based insights"""
        insights = []
        
        personality_score = career.get('personality_score', 0)
        
        if personality_score > 0.8:
            insights.append("Your personality traits are highly compatible with this career")
        elif personality_score > 0.6:
            insights.append("Your personality is well-suited for this career path")
        else:
            insights.append("Consider how your personality traits align with this career's demands")
        
        # Add specific trait insights
        user_traits = personality_profile.get('big_five_scores', {})
        if user_traits:
            high_traits = [trait for trait, score in user_traits.items() if score > 70]
            if high_traits:
                insights.append(f"Your strong {', '.join(high_traits)} traits will be valuable in this role")
        
        return insights
    
    def _analyze_matches(self, ranked_careers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the overall matching results"""
        if not ranked_careers:
            return {'confidence': 'low', 'diversity': 'low', 'insights': []}
        
        scores = [career['total_score'] for career in ranked_careers]
        
        # Calculate confidence based on score distribution
        top_score = scores[0]
        score_variance = np.var(scores[:5])  # Variance of top 5 scores
        
        if top_score > 0.8 and score_variance < 0.1:
            confidence = 'high'
        elif top_score > 0.6:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        # Calculate diversity of top matches
        top_titles = [career['title'] for career in ranked_careers[:5]]
        unique_industries = len(set(top_titles))
        diversity = 'high' if unique_industries >= 4 else 'medium' if unique_industries >= 2 else 'low'
        
        insights = [
            f"Your profile shows {confidence} compatibility with career options",
            f"Top matches span {unique_industries} different career areas",
            f"Consider exploring the top {min(3, len(ranked_careers))} career paths"
        ]
        
        return {
            'confidence': confidence,
            'diversity': diversity,
            'insights': insights,
            'score_distribution': {
                'top_score': top_score,
                'average_top_5': np.mean(scores[:5]),
                'score_variance': score_variance
            }
        }
    
    def get_career_path_roadmap(self, career_id: str) -> Dict[str, Any]:
        """Get a detailed career path roadmap for a specific career"""
        try:
            if career_id not in self.career_profiles:
                return {'success': False, 'error': 'Career not found'}
            
            career_data = self.career_profiles[career_id]
            
            roadmap = {
                'success': True,
                'career_title': career_data.get('title', ''),
                'roadmap': {
                    'entry_level': career_data.get('entry_level_roles', []),
                    'mid_level': career_data.get('mid_level_roles', []),
                    'senior_level': career_data.get('senior_level_roles', []),
                    'leadership': career_data.get('leadership_roles', [])
                },
                'skill_progression': career_data.get('skill_progression', {}),
                'timeline': career_data.get('typical_timeline', {}),
                'certifications': career_data.get('recommended_certifications', []),
                'networking': career_data.get('networking_opportunities', [])
            }
            
            return roadmap
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
