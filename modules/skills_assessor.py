"""
Skills Assessment Module for Skillora AI Career Advisor

This module provides comprehensive skills assessment functionality including:
- Technical skills evaluation
- Soft skills assessment
- Skills gap analysis
- Integration with trained ML models for predictions
"""

import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import joblib


class SkillsAssessor:
    """
    Comprehensive skills assessment system that evaluates user skills
    and provides gap analysis using trained ML models
    """
    
    def __init__(self, data_loader):
        """
        Initialize the Skills Assessor
        
        Args:
            data_loader: DataLoader instance for accessing data
        """
        self.data_loader = data_loader
        self.skills_model = None
        self.skills_categories = self._load_skills_categories()
        self.assessment_questions = self._load_assessment_questions()
        
        # Try to load trained skills model
        self._load_trained_model()
    
    def _load_trained_model(self):
        """Load the trained skills prediction model if available"""
        try:
            model_path = 'models/trained_models/skills_model.pkl'
            if os.path.exists(model_path):
                self.skills_model = joblib.load(model_path)
                print("✅ Skills prediction model loaded successfully")
            else:
                print("⚠️ Skills model not found, using fallback assessment")
        except Exception as e:
            print(f"⚠️ Failed to load skills model: {e}")
            self.skills_model = None
    
    def _load_skills_categories(self) -> Dict[str, List[str]]:
        """Load skills categories and mappings"""
        try:
            if self.data_loader:
                return self.data_loader.get_skills_mapping()
        except:
            pass
        
        # Fallback skills categories
        return {
            "technical_skills": [
                "Python", "JavaScript", "Java", "C++", "SQL", "HTML/CSS",
                "React", "Node.js", "Django", "Flask", "Machine Learning",
                "Data Analysis", "Cloud Computing", "DevOps", "Git",
                "Docker", "Kubernetes", "AWS", "Azure", "MongoDB"
            ],
            "soft_skills": [
                "Communication", "Leadership", "Problem Solving", "Teamwork",
                "Time Management", "Critical Thinking", "Adaptability",
                "Creativity", "Project Management", "Public Speaking",
                "Negotiation", "Conflict Resolution", "Emotional Intelligence"
            ],
            "domain_skills": [
                "Web Development", "Mobile Development", "Data Science",
                "AI/ML", "Cybersecurity", "UI/UX Design", "Digital Marketing",
                "Business Analysis", "Quality Assurance", "System Administration"
            ]
        }
    
    def _load_assessment_questions(self) -> List[Dict]:
        """Load skills assessment questions"""
        return [
            {
                "category": "Technical Skills",
                "description": "Rate your proficiency in these technical skills",
                "skills": self.skills_categories.get("technical_skills", [])[:15],
                "scale": "1-5 (1=Beginner, 5=Expert)"
            },
            {
                "category": "Soft Skills", 
                "description": "Rate your proficiency in these soft skills",
                "skills": self.skills_categories.get("soft_skills", [])[:10],
                "scale": "1-5 (1=Poor, 5=Excellent)"
            },
            {
                "category": "Domain Knowledge",
                "description": "Rate your knowledge in these domain areas",
                "skills": self.skills_categories.get("domain_skills", [])[:10],
                "scale": "1-5 (1=No Knowledge, 5=Expert)"
            }
        ]
    
    def get_assessment_questions(self) -> List[Dict]:
        """
        Get skills assessment questions
        
        Returns:
            List of assessment question categories
        """
        return self.assessment_questions
    
    def assess_skills(self, skills_data: Dict, experience: str = "", education: str = "") -> Dict[str, Any]:
        """
        Assess user skills and provide comprehensive analysis
        
        Args:
            skills_data: Dictionary containing user's skill ratings
            experience: User's work experience description
            education: User's educational background
            
        Returns:
            Comprehensive skills assessment results
        """
        try:
            # Process skill ratings
            skill_profile = self._process_skill_ratings(skills_data)
            
            # Calculate overall scores
            overall_scores = self._calculate_overall_scores(skill_profile)
            
            # Identify skill gaps using ML model if available
            skill_gaps = self._identify_skill_gaps(skill_profile, experience, education)
            
            # Generate recommendations
            recommendations = self._generate_skill_recommendations(skill_profile, skill_gaps)
            
            # Create learning priorities
            learning_priorities = self._create_learning_priorities(skill_gaps)
            
            return {
                'skill_profile': skill_profile,
                'overall_scores': overall_scores,
                'skill_gaps': skill_gaps,
                'recommendations': recommendations,
                'learning_priorities': learning_priorities,
                'assessment_date': datetime.now().isoformat(),
                'experience_level': self._determine_experience_level(overall_scores, experience)
            }
            
        except Exception as e:
            print(f"Error in skills assessment: {e}")
            return self._get_fallback_assessment(skills_data)
    
    def _process_skill_ratings(self, skills_data: Dict) -> Dict[str, Any]:
        """Process and categorize skill ratings"""
        skill_profile = {
            'technical_skills': {},
            'soft_skills': {},
            'domain_skills': {},
            'all_skills': {}
        }
        
        for category, skills in skills_data.items():
            if isinstance(skills, dict):
                for skill, rating in skills.items():
                    try:
                        rating_value = float(rating)
                        skill_profile['all_skills'][skill] = rating_value
                        
                        # Categorize skills
                        if skill in self.skills_categories.get('technical_skills', []):
                            skill_profile['technical_skills'][skill] = rating_value
                        elif skill in self.skills_categories.get('soft_skills', []):
                            skill_profile['soft_skills'][skill] = rating_value
                        elif skill in self.skills_categories.get('domain_skills', []):
                            skill_profile['domain_skills'][skill] = rating_value
                    except (ValueError, TypeError):
                        continue
        
        return skill_profile
    
    def _calculate_overall_scores(self, skill_profile: Dict) -> Dict[str, float]:
        """Calculate overall scores for each skill category"""
        scores = {}
        
        for category, skills in skill_profile.items():
            if skills and category != 'all_skills':
                scores[category] = sum(skills.values()) / len(skills)
            elif category == 'all_skills' and skills:
                scores['overall'] = sum(skills.values()) / len(skills)
        
        return scores
    
    def _identify_skill_gaps(self, skill_profile: Dict, experience: str, education: str) -> List[Dict]:
        """Identify skill gaps using ML model or rule-based approach"""
        if self.skills_model:
            return self._ml_based_gap_analysis(skill_profile, experience, education)
        else:
            return self._rule_based_gap_analysis(skill_profile)
    
    def _ml_based_gap_analysis(self, skill_profile: Dict, experience: str, education: str) -> List[Dict]:
        """Use trained ML model for skill gap analysis"""
        try:
            # Prepare features for the model
            features = self._prepare_features_for_model(skill_profile, experience, education)
            
            # Get predictions from the model
            if hasattr(self.skills_model, 'predict_proba'):
                predictions = self.skills_model.predict_proba(features)
            else:
                # Fallback to simple prediction
                predictions = self.skills_model.predict(features)
                # Convert to probability-like format
                predictions = np.array([[pred] for pred in predictions])
            
            # Convert predictions to skill gaps
            return self._convert_predictions_to_gaps(predictions)
            
        except Exception as e:
            print(f"ML gap analysis failed: {e}")
            return self._rule_based_gap_analysis(skill_profile)
    
    def _rule_based_gap_analysis(self, skill_profile: Dict) -> List[Dict]:
        """Rule-based skill gap analysis as fallback"""
        gaps = []
        
        # Identify low-rated skills as gaps
        for category, skills in skill_profile.items():
            if category != 'all_skills' and skills:
                for skill, rating in skills.items():
                    if rating < 3.0:  # Skills rated below 3 are considered gaps
                        gaps.append({
                            'skill': skill,
                            'category': category,
                            'current_level': rating,
                            'target_level': 4.0,
                            'gap_size': 4.0 - rating,
                            'priority': 'High' if rating < 2.0 else 'Medium'
                        })
        
        # Add trending skills that user doesn't have
        trending_skills = self._get_trending_skills()
        for skill in trending_skills[:5]:  # Top 5 trending skills
            if skill not in skill_profile.get('all_skills', {}):
                gaps.append({
                    'skill': skill,
                    'category': 'technical_skills',
                    'current_level': 0,
                    'target_level': 3.0,
                    'gap_size': 3.0,
                    'priority': 'High',
                    'reason': 'Trending skill in the market'
                })
        
        return sorted(gaps, key=lambda x: x['gap_size'], reverse=True)
    
    def _get_trending_skills(self) -> List[str]:
        """Get trending skills from data loader or use defaults"""
        try:
            if self.data_loader:
                return self.data_loader.get_trending_skills()
        except:
            pass
        
        return [
            "Machine Learning", "Cloud Computing", "React", "Python",
            "Data Analysis", "DevOps", "Kubernetes", "AI/ML"
        ]
    
    def _prepare_features_for_model(self, skill_profile: Dict, experience: str, education: str) -> pd.DataFrame:
        """Prepare features for ML model input"""
        # This would create a feature vector based on the trained model's requirements
        features = {}
        
        # Add skill ratings as features
        all_skills = skill_profile.get('all_skills', {})
        if isinstance(all_skills, dict):
            for skill, rating in all_skills.items():
                if isinstance(skill, str):
                    features[f'skill_{skill.lower().replace(" ", "_")}'] = rating
        elif isinstance(all_skills, list):
            # If it's a list, assume all skills have default rating
            for skill in all_skills:
                if isinstance(skill, str):
                    features[f'skill_{skill.lower().replace(" ", "_")}'] = 3.0
        
        # Add experience and education features
        features['experience_years'] = self._extract_experience_years(experience)
        features['education_level'] = self._encode_education_level(education)
        
        return pd.DataFrame([features])
    
    def _extract_experience_years(self, experience: str) -> float:
        """Extract years of experience from text"""
        # Simple extraction logic
        import re
        matches = re.findall(r'(\d+)\s*(?:year|yr)', experience.lower())
        return float(matches[0]) if matches else 0.0
    
    def _encode_education_level(self, education: str) -> int:
        """Encode education level as numeric value"""
        education_lower = education.lower()
        if 'phd' in education_lower or 'doctorate' in education_lower:
            return 5
        elif 'master' in education_lower or 'msc' in education_lower or 'mba' in education_lower:
            return 4
        elif 'bachelor' in education_lower or 'bsc' in education_lower or 'btech' in education_lower:
            return 3
        elif 'diploma' in education_lower:
            return 2
        else:
            return 1
    
    def _convert_predictions_to_gaps(self, predictions) -> List[Dict]:
        """Convert ML model predictions to skill gaps"""
        # This would depend on the specific model output format
        # For now, return a placeholder
        return []
    
    def _generate_skill_recommendations(self, skill_profile: Dict, skill_gaps: List[Dict]) -> List[str]:
        """Generate skill development recommendations"""
        recommendations = []
        
        # Recommendations based on skill gaps
        high_priority_gaps = [gap for gap in skill_gaps if gap.get('priority') == 'High']
        
        if high_priority_gaps:
            recommendations.append(f"Focus on developing {len(high_priority_gaps)} high-priority skills")
            
            for gap in high_priority_gaps[:3]:  # Top 3 gaps
                skill = gap['skill']
                recommendations.append(f"Improve {skill} from level {gap['current_level']:.1f} to {gap['target_level']:.1f}")
        
        # Recommendations based on overall scores
        overall_scores = self._calculate_overall_scores(skill_profile)
        
        if overall_scores.get('technical_skills', 0) < 3.0:
            recommendations.append("Consider taking technical courses to strengthen your programming skills")
        
        if overall_scores.get('soft_skills', 0) < 3.5:
            recommendations.append("Work on developing soft skills through workshops or practice")
        
        # Add learning resource recommendations
        recommendations.extend([
            "Consider online courses on platforms like Coursera, Udemy, or edX",
            "Practice coding on platforms like LeetCode, HackerRank, or GitHub",
            "Join professional communities and attend networking events"
        ])
        
        return recommendations
    
    def _create_learning_priorities(self, skill_gaps: List[Dict]) -> List[Dict]:
        """Create prioritized learning plan"""
        priorities = []
        
        # Group gaps by priority
        high_priority = [gap for gap in skill_gaps if gap.get('priority') == 'High']
        medium_priority = [gap for gap in skill_gaps if gap.get('priority') == 'Medium']
        
        # Create learning phases
        if high_priority:
            priorities.append({
                'phase': 'Immediate (0-3 months)',
                'skills': [gap['skill'] for gap in high_priority[:3]],
                'focus': 'Critical skill gaps that need immediate attention'
            })
        
        if medium_priority:
            priorities.append({
                'phase': 'Short-term (3-6 months)',
                'skills': [gap['skill'] for gap in medium_priority[:3]],
                'focus': 'Important skills for career advancement'
            })
        
        # Add long-term goals
        priorities.append({
            'phase': 'Long-term (6-12 months)',
            'skills': ['Advanced specialization', 'Leadership skills', 'Industry certifications'],
            'focus': 'Advanced skills and career specialization'
        })
        
        return priorities
    
    def _determine_experience_level(self, overall_scores: Dict, experience: str) -> str:
        """Determine user's experience level"""
        overall_score = overall_scores.get('overall', 0)
        experience_years = self._extract_experience_years(experience)
        
        if overall_score >= 4.0 and experience_years >= 5:
            return 'Senior'
        elif overall_score >= 3.0 and experience_years >= 2:
            return 'Intermediate'
        elif overall_score >= 2.0 or experience_years >= 1:
            return 'Junior'
        else:
            return 'Entry Level'
    
    def _get_fallback_assessment(self, skills_data: Dict) -> Dict[str, Any]:
        """Provide fallback assessment when main assessment fails"""
        return {
            'skill_profile': {
                'technical_skills': skills_data.get('technical', {}),
                'soft_skills': skills_data.get('soft', {}),
                'overall_score': 3.0
            },
            'skill_gaps': [
                {'skill': 'Python', 'priority': 'High', 'gap_size': 2.0},
                {'skill': 'Machine Learning', 'priority': 'High', 'gap_size': 3.0},
                {'skill': 'Communication', 'priority': 'Medium', 'gap_size': 1.0}
            ],
            'recommendations': [
                'Focus on developing technical skills',
                'Consider online courses for skill gaps',
                'Practice coding regularly'
            ],
            'learning_priorities': [
                {
                    'phase': 'Immediate (0-3 months)',
                    'skills': ['Python', 'Machine Learning'],
                    'focus': 'Core technical skills'
                }
            ],
            'assessment_date': datetime.now().isoformat(),
            'experience_level': 'Intermediate'
        }
    
    def get_skills_for_career(self, career_id: str) -> List[str]:
        """Get required skills for a specific career"""
        try:
            if self.data_loader:
                career_data = self.data_loader.get_career_by_id(career_id)
                return career_data.get('required_skills', [])
        except:
            pass
        
        # Fallback career skills mapping
        career_skills = {
            'software_engineer': ['Python', 'JavaScript', 'Git', 'Problem Solving', 'Teamwork'],
            'data_scientist': ['Python', 'Machine Learning', 'Statistics', 'SQL', 'Data Analysis'],
            'web_developer': ['HTML/CSS', 'JavaScript', 'React', 'Node.js', 'Problem Solving'],
            'mobile_developer': ['Java', 'Swift', 'React Native', 'UI/UX Design', 'Problem Solving']
        }
        
        return career_skills.get(career_id, ['Problem Solving', 'Communication', 'Teamwork'])
    
    def compare_skills_with_career(self, user_skills: Dict, career_id: str) -> Dict[str, Any]:
        """Compare user skills with career requirements"""
        required_skills = self.get_skills_for_career(career_id)
        user_skill_ratings = user_skills.get('all_skills', {})
        
        matches = []
        gaps = []
        
        for skill in required_skills:
            if skill in user_skill_ratings:
                rating = user_skill_ratings[skill]
                matches.append({
                    'skill': skill,
                    'user_rating': rating,
                    'match_strength': 'Strong' if rating >= 4 else 'Moderate' if rating >= 3 else 'Weak'
                })
            else:
                gaps.append({
                    'skill': skill,
                    'required_level': 3.0,
                    'priority': 'High'
                })
        
        compatibility_score = len(matches) / len(required_skills) * 100 if required_skills else 0
        
        return {
            'career_id': career_id,
            'compatibility_score': compatibility_score,
            'skill_matches': matches,
            'skill_gaps': gaps,
            'recommendation': 'Good fit' if compatibility_score >= 70 else 'Needs development' if compatibility_score >= 40 else 'Significant gaps'
        }