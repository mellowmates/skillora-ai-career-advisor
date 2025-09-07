"""
Career Matching Module for Skillora AI Career Advisor

This module provides intelligent career matching using trained ML models:
- Career recommendation based on personality, skills, and preferences
- Compatibility scoring using trained models
- Market-aware recommendations
- Personalized career path suggestions
"""

import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import joblib


class CareerMatcher:
    """
    Intelligent career matching system that uses trained ML models
    to provide personalized career recommendations
    """
    
    def __init__(self, data_loader):
        """
        Initialize the Career Matcher
        
        Args:
            data_loader: DataLoader instance for accessing career data
        """
        self.data_loader = data_loader
        self.career_model = None
        self.salary_model = None
        self.career_profiles = self._load_career_profiles()
        
        # Load trained models
        self._load_trained_models()
    
    def _load_trained_models(self):
        """Load trained ML models for career recommendation and salary prediction"""
        try:
            # Load career recommendation model
            career_model_path = 'models/trained_models/career_model.pkl'
            if os.path.exists(career_model_path):
                model_data = joblib.load(career_model_path)
                if isinstance(model_data, dict) and 'model' in model_data:
                    self.career_model = model_data['model']
                    self.career_model_data = model_data  # Store full data for reference
                    print(f"✅ Career recommendation model loaded successfully")
                    print(f"🔍 Model type: {type(self.career_model)}")
                    if hasattr(self.career_model, 'classes_'):
                        print(f"📋 Model classes: {self.career_model.classes_}")
                else:
                    self.career_model = model_data
                    print(f"✅ Career recommendation model loaded successfully")
            
            # Load salary prediction model
            salary_model_path = 'models/trained_models/salary_model.pkl'
            if os.path.exists(salary_model_path):
                model_data = joblib.load(salary_model_path)
                if isinstance(model_data, dict) and 'model' in model_data:
                    self.salary_model = model_data['model']
                    self.salary_model_data = model_data
                    print(f"✅ Salary prediction model loaded successfully")
                else:
                    self.salary_model = model_data
                    print(f"✅ Salary prediction model loaded successfully")
                
        except Exception as e:
            print(f"⚠️ Failed to load ML models: {e}")
            print("Using fallback career matching system")
    
    def _load_career_profiles(self) -> List[Dict]:
        """Load career profiles from data loader"""
        try:
            if self.data_loader:
                return self.data_loader.get_career_profiles()
        except:
            pass
        
        # Fallback career profiles
        return [
            {
                'career_id': 'software_engineer',
                'title': 'Software Engineer',
                'description': 'Design, develop, and maintain software applications',
                'required_skills': ['Python', 'JavaScript', 'Problem Solving', 'Git', 'Teamwork'],
                'personality_traits': {
                    'openness': 'high',
                    'conscientiousness': 'high',
                    'extraversion': 'medium',
                    'agreeableness': 'medium',
                    'neuroticism': 'low'
                },
                'education_requirements': ['Bachelor\'s in Computer Science or related field'],
                'experience_level': 'entry_to_senior',
                'growth_prospects': 'High',
                'work_environment': 'office_remote',
                'salary_range': {'min': 600000, 'max': 1200000, 'currency': 'INR'},
                'job_market_demand': 'Very High'
            },
            {
                'career_id': 'data_scientist',
                'title': 'Data Scientist',
                'description': 'Analyze complex data to drive business decisions',
                'required_skills': ['Python', 'Machine Learning', 'Statistics', 'SQL', 'Data Analysis'],
                'personality_traits': {
                    'openness': 'very_high',
                    'conscientiousness': 'high',
                    'extraversion': 'medium',
                    'agreeableness': 'medium',
                    'neuroticism': 'low'
                },
                'education_requirements': ['Bachelor\'s in Statistics, Math, CS or related field'],
                'experience_level': 'junior_to_senior',
                'growth_prospects': 'Very High',
                'work_environment': 'office_remote',
                'salary_range': {'min': 800000, 'max': 1500000, 'currency': 'INR'},
                'job_market_demand': 'Very High'
            },
            {
                'career_id': 'web_developer',
                'title': 'Web Developer',
                'description': 'Create and maintain websites and web applications',
                'required_skills': ['HTML/CSS', 'JavaScript', 'React', 'Node.js', 'Problem Solving'],
                'personality_traits': {
                    'openness': 'high',
                    'conscientiousness': 'high',
                    'extraversion': 'medium',
                    'agreeableness': 'medium',
                    'neuroticism': 'low'
                },
                'education_requirements': ['Bachelor\'s in Computer Science or equivalent experience'],
                'experience_level': 'entry_to_senior',
                'growth_prospects': 'High',
                'work_environment': 'office_remote',
                'salary_range': {'min': 500000, 'max': 1000000, 'currency': 'INR'},
                'job_market_demand': 'High'
            },
            {
                'career_id': 'mobile_developer',
                'title': 'Mobile App Developer',
                'description': 'Develop mobile applications for iOS and Android',
                'required_skills': ['Java', 'Swift', 'React Native', 'UI/UX Design', 'Problem Solving'],
                'personality_traits': {
                    'openness': 'high',
                    'conscientiousness': 'high',
                    'extraversion': 'medium',
                    'agreeableness': 'medium',
                    'neuroticism': 'low'
                },
                'education_requirements': ['Bachelor\'s in Computer Science or related field'],
                'experience_level': 'entry_to_senior',
                'growth_prospects': 'High',
                'work_environment': 'office_remote',
                'salary_range': {'min': 550000, 'max': 1100000, 'currency': 'INR'},
                'job_market_demand': 'High'
            },
            {
                'career_id': 'product_manager',
                'title': 'Product Manager',
                'description': 'Lead product development and strategy',
                'required_skills': ['Product Strategy', 'Communication', 'Leadership', 'Analytics', 'Project Management'],
                'personality_traits': {
                    'openness': 'high',
                    'conscientiousness': 'very_high',
                    'extraversion': 'high',
                    'agreeableness': 'high',
                    'neuroticism': 'low'
                },
                'education_requirements': ['Bachelor\'s degree, MBA preferred'],
                'experience_level': 'junior_to_senior',
                'growth_prospects': 'Very High',
                'work_environment': 'office',
                'salary_range': {'min': 1000000, 'max': 2000000, 'currency': 'INR'},
                'job_market_demand': 'High'
            }
        ]
    
    def get_recommendations(self, user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get personalized career recommendations using trained ML models
        
        Args:
            user_profile: Dictionary containing user's personality, skills, and preferences
            
        Returns:
            List of career recommendations with compatibility scores
        """
        try:
            if self.career_model:
                print(f"🔍 Using ML model for career recommendations")
                print(f"📊 User profile keys: {list(user_profile.keys())}")
                return self._ml_based_recommendations(user_profile)
            else:
                print(f"⚠️ No ML model available, using rule-based recommendations")
                return self._rule_based_recommendations(user_profile)
                
        except Exception as e:
            print(f"Error in career matching: {e}")
            return self._fallback_recommendations(user_profile)
    
    def _ml_based_recommendations(self, user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations using trained ML models"""
        try:
            # Prepare features for the model
            features = self._prepare_features_for_model(user_profile)
            print(f"🔧 Prepared features: {features.columns.tolist()}")
            print(f"📈 Feature values: {features.iloc[0].to_dict()}")
            
            # Get predictions from the career model
            if hasattr(self.career_model, 'predict_proba'):
                career_probabilities = self.career_model.predict_proba(features)[0]
                career_classes = self.career_model.classes_
                print(f"🎯 ML predictions: {dict(zip(career_classes, career_probabilities))}")
            else:
                # Fallback to simple prediction
                predictions = self.career_model.predict(features)
                career_classes = np.unique(predictions)
                career_probabilities = np.zeros(len(career_classes))
                for i, cls in enumerate(career_classes):
                    career_probabilities[i] = np.mean(predictions == cls)
                print(f"🎯 Simple predictions: {dict(zip(career_classes, career_probabilities))}")
            
            # Create recommendations with ML-based scores
            recommendations = []
            
            for i, career_class in enumerate(career_classes):
                probability = career_probabilities[i]
                
                # Find career profile
                career_profile = self._find_career_profile(career_class)
                if career_profile:
                    # Get salary prediction if model is available
                    predicted_salary = self._predict_salary(features, career_profile)
                    
                    # Calculate comprehensive compatibility score
                    compatibility_score = self._calculate_ml_compatibility_score(
                        user_profile, career_profile, probability
                    )
                    
                    recommendation = {
                        'career_id': career_profile['career_id'],
                        'career_title': career_profile['title'],
                        'career_category': career_profile.get('category', 'Technology'),
                        'career_description': career_profile['description'],
                        'final_score': compatibility_score / 100.0,  # Convert to 0-1 range for template
                        'compatibility_score': round(compatibility_score, 1),
                        'ml_probability': round(probability * 100, 1),
                        'required_skills': career_profile['required_skills'],
                        'salary_range': self._format_salary_range(career_profile['salary_range']),
                        'predicted_salary': predicted_salary,
                        'growth_outlook': career_profile['growth_prospects'],
                        'job_market_demand': career_profile['job_market_demand'],
                        'education_requirements': career_profile['education_requirements'],
                        'work_environment': career_profile['work_environment'],
                        'match_reasons': self._generate_match_reasons(user_profile, career_profile),
                        'skill_gaps': self._identify_skill_gaps(user_profile, career_profile),
                        'skill_analysis': {
                            'skill_match_percentage': round(compatibility_score, 1)
                        },
                        'recommendation_confidence': 'High' if probability > 0.7 else 'Medium' if probability > 0.4 else 'Low'
                    }
                    
                    recommendations.append(recommendation)
            
            # Sort by compatibility score and return top recommendations
            recommendations.sort(key=lambda x: x['compatibility_score'], reverse=True)
            return recommendations[:10]  # Top 10 recommendations
            
        except Exception as e:
            print(f"ML-based recommendation failed: {e}")
            return self._rule_based_recommendations(user_profile)
    
    def _prepare_features_for_model(self, user_profile: Dict[str, Any]) -> pd.DataFrame:
        """Prepare features for ML model input"""
        features = {}
        
        # Personality features - match exact model expectations
        personality = user_profile.get('personality', {}).get('personality_profile', {})
        for trait in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
            trait_data = personality.get(trait, {})
            if isinstance(trait_data, dict) and 'score' in trait_data:
                features[f'personality_{trait}'] = trait_data['score']
            else:
                features[f'personality_{trait}'] = 3.0  # Default neutral score
        
        # Skills features
        skills = user_profile.get('skills', {}).get('skill_profile', {})
        skills_count = 0
        if isinstance(skills, dict):
            # Count all skills
            for skill_category in ['technical_skills', 'soft_skills', 'domain_skills']:
                category_skills = skills.get(skill_category, {})
                if isinstance(category_skills, dict):
                    skills_count += len(category_skills)
        
        features['skills_count'] = skills_count
        
        # Experience features
        user_data = user_profile.get('user_data', {})
        experience_text = user_data.get('experience', '')
        features['experience_years'] = self._extract_experience_years(experience_text)
        
        # Education features
        education_text = user_data.get('education', '')
        features['education_level'] = self._encode_education_level(education_text)
        
        # Location features
        location = user_data.get('location', 'india')
        features['location_tier'] = self._encode_location_tier(location)
        
        # Convert to DataFrame with proper column order
        feature_df = pd.DataFrame([features])
        
        # Ensure all required features are present
        required_features = [
            'education_level', 'experience_years', 'skills_count',
            'personality_openness', 'personality_conscientiousness', 
            'personality_extraversion', 'personality_agreeableness', 
            'personality_neuroticism', 'location_tier'
        ]
        
        for feature in required_features:
            if feature not in feature_df.columns:
                feature_df[feature] = 0  # Default value
        
        return feature_df[required_features]  # Return in correct order
    
    def _encode_work_environment(self, work_env: str) -> int:
        """Encode work environment preference as numeric value"""
        env_mapping = {
            'remote': 3,
            'hybrid': 2,
            'office': 1,
            'office_remote': 2
        }
        return env_mapping.get(work_env.lower(), 1)
    
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
    
    def _extract_experience_years(self, experience: str) -> float:
        """Extract years of experience from text"""
        import re
        matches = re.findall(r'(\d+)\s*(?:year|yr)', experience.lower())
        return float(matches[0]) if matches else 0.0
    
    def _encode_location_tier(self, location: str) -> int:
        """Encode location tier based on Indian city classification"""
        tier1_cities = ['bangalore', 'mumbai', 'delhi', 'hyderabad', 'pune', 'chennai', 'kolkata']
        tier2_cities = ['ahmedabad', 'jaipur', 'surat', 'lucknow', 'kanpur', 'nagpur', 'indore', 'thane']
        
        location_lower = location.lower()
        if location_lower in tier1_cities:
            return 3  # Tier 1
        elif location_lower in tier2_cities:
            return 2  # Tier 2
        else:
            return 1  # Tier 3 or other
    
    def _find_career_profile(self, career_id: str) -> Optional[Dict]:
        """Find career profile by ID"""
        for profile in self.career_profiles:
            if profile['career_id'] == career_id:
                return profile
        return None
    
    def _predict_salary(self, features: pd.DataFrame, career_profile: Dict) -> Optional[Dict]:
        """Predict salary using trained salary model"""
        if not self.salary_model:
            return None
        
        try:
            # Add career-specific features
            career_features = features.copy()
            career_features['career_id_encoded'] = hash(career_profile['career_id']) % 1000
            
            # Predict salary
            predicted_salary = self.salary_model.predict(career_features)[0]
            
            return {
                'predicted_amount': round(predicted_salary),
                'currency': 'INR',
                'confidence': 'Medium',
                'range': {
                    'min': round(predicted_salary * 0.8),
                    'max': round(predicted_salary * 1.2)
                }
            }
            
        except Exception as e:
            print(f"Salary prediction failed: {e}")
            return None
    
    def _calculate_ml_compatibility_score(self, user_profile: Dict, career_profile: Dict, ml_probability: float) -> float:
        """Calculate comprehensive compatibility score combining ML and rule-based factors"""
        # Base score from ML model (70% weight)
        ml_score = ml_probability * 70
        
        # Rule-based adjustments (30% weight)
        rule_score = self._calculate_rule_based_score(user_profile, career_profile) * 30
        
        # Combine scores
        total_score = ml_score + rule_score
        
        # Ensure score is between 0 and 100
        return min(100, max(0, total_score))
    
    def _calculate_rule_based_score(self, user_profile: Dict, career_profile: Dict) -> float:
        """Calculate rule-based compatibility score"""
        score = 0.0
        factors = 0
        
        # Personality compatibility
        personality_score = self._calculate_personality_compatibility(user_profile, career_profile)
        if personality_score is not None:
            score += personality_score
            factors += 1
        
        # Skills compatibility
        skills_score = self._calculate_skills_compatibility(user_profile, career_profile)
        if skills_score is not None:
            score += skills_score
            factors += 1
        
        # Preferences compatibility
        preferences_score = self._calculate_preferences_compatibility(user_profile, career_profile)
        if preferences_score is not None:
            score += preferences_score
            factors += 1
        
        return score / factors if factors > 0 else 0.5
    
    def _calculate_personality_compatibility(self, user_profile: Dict, career_profile: Dict) -> Optional[float]:
        """Calculate personality compatibility score"""
        try:
            user_personality = user_profile.get('personality', {}).get('personality_profile', {})
            required_traits = career_profile.get('personality_traits', {})
            
            if not user_personality or not required_traits:
                return None
            
            compatibility_score = 0.0
            trait_count = 0
            
            trait_level_mapping = {
                'very_low': 1, 'low': 2, 'medium': 3, 'high': 4, 'very_high': 5
            }
            
            for trait, required_level in required_traits.items():
                if trait in user_personality:
                    user_score = user_personality[trait].get('score', 3)
                    required_score = trait_level_mapping.get(required_level, 3)
                    
                    # Calculate compatibility (closer scores = higher compatibility)
                    diff = abs(user_score - required_score)
                    trait_compatibility = max(0, 1 - (diff / 4))  # Normalize to 0-1
                    
                    compatibility_score += trait_compatibility
                    trait_count += 1
            
            return compatibility_score / trait_count if trait_count > 0 else None
            
        except Exception:
            return None
    
    def _calculate_skills_compatibility(self, user_profile: Dict, career_profile: Dict) -> Optional[float]:
        """Calculate skills compatibility score"""
        try:
            user_skills = user_profile.get('skills', {}).get('skill_profile', {}).get('all_skills', {})
            required_skills = career_profile.get('required_skills', [])
            
            if not user_skills or not required_skills:
                return None
            
            matched_skills = 0
            total_skill_score = 0
            
            for skill in required_skills:
                if skill in user_skills:
                    skill_level = user_skills[skill]
                    total_skill_score += skill_level
                    if skill_level >= 3:  # Consider skill as matched if level >= 3
                        matched_skills += 1
            
            # Calculate compatibility based on matched skills and average skill level
            match_ratio = matched_skills / len(required_skills)
            avg_skill_level = total_skill_score / len(required_skills) if required_skills else 0
            
            return (match_ratio * 0.7) + (min(avg_skill_level / 5, 1) * 0.3)
            
        except Exception:
            return None
    
    def _calculate_preferences_compatibility(self, user_profile: Dict, career_profile: Dict) -> Optional[float]:
        """Calculate preferences compatibility score"""
        try:
            preferences = user_profile.get('preferences', {})
            
            compatibility_score = 0.0
            factors = 0
            
            # Work environment preference
            user_work_env = preferences.get('work_environment', '').lower()
            career_work_env = career_profile.get('work_environment', '').lower()
            
            if user_work_env and career_work_env:
                if user_work_env == career_work_env or 'remote' in career_work_env:
                    compatibility_score += 1.0
                elif 'hybrid' in career_work_env or 'office_remote' in career_work_env:
                    compatibility_score += 0.7
                else:
                    compatibility_score += 0.3
                factors += 1
            
            # Salary expectations
            user_salary_exp = preferences.get('salary_expectations', 0)
            career_salary_range = career_profile.get('salary_range', {})
            
            if user_salary_exp > 0 and career_salary_range:
                career_min = career_salary_range.get('min', 0)
                career_max = career_salary_range.get('max', 0)
                
                if career_min <= user_salary_exp <= career_max:
                    compatibility_score += 1.0
                elif user_salary_exp < career_min:
                    # User expectation is lower than career offers
                    compatibility_score += 0.8
                else:
                    # User expectation is higher than career offers
                    ratio = career_max / user_salary_exp if user_salary_exp > 0 else 0
                    compatibility_score += max(0, ratio)
                
                factors += 1
            
            return compatibility_score / factors if factors > 0 else None
            
        except Exception:
            return None
    
    def _rule_based_recommendations(self, user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations using rule-based approach"""
        recommendations = []
        
        for career_profile in self.career_profiles:
            compatibility_score = self._calculate_rule_based_score(user_profile, career_profile) * 100
            
            recommendation = {
                'career_id': career_profile['career_id'],
                'career_title': career_profile['title'],
                'career_category': career_profile.get('category', 'Technology'),
                'career_description': career_profile['description'],
                'final_score': compatibility_score / 100.0,  # Convert to 0-1 range for template
                'compatibility_score': round(compatibility_score, 1),
                'required_skills': career_profile['required_skills'],
                'salary_range': self._format_salary_range(career_profile['salary_range']),
                'growth_outlook': career_profile['growth_prospects'],
                'job_market_demand': career_profile['job_market_demand'],
                'education_requirements': career_profile['education_requirements'],
                'work_environment': career_profile['work_environment'],
                'match_reasons': self._generate_match_reasons(user_profile, career_profile),
                'skill_gaps': self._identify_skill_gaps(user_profile, career_profile),
                'skill_analysis': {
                    'skill_match_percentage': round(compatibility_score, 1)
                },
                'recommendation_confidence': 'Medium'
            }
            
            recommendations.append(recommendation)
        
        # Sort by compatibility score
        recommendations.sort(key=lambda x: x['compatibility_score'], reverse=True)
        return recommendations
    
    def _format_salary_range(self, salary_range: Dict) -> Dict:
        """Format salary range for display"""
        if not salary_range:
            return {"formatted": "Competitive salary"}
        
        min_salary = salary_range.get('min', 0)
        max_salary = salary_range.get('max', 0)
        currency = salary_range.get('currency', 'INR')
        
        if currency == 'INR':
            min_lpa = min_salary / 100000
            max_lpa = max_salary / 100000
            formatted = f"₹{min_lpa:.1f}-{max_lpa:.1f} LPA"
        else:
            formatted = f"{min_salary:,}-{max_salary:,} {currency}"
        
        return {
            "formatted": formatted,
            "min": min_salary,
            "max": max_salary,
            "currency": currency
        }
    
    def _generate_match_reasons(self, user_profile: Dict, career_profile: Dict) -> List[str]:
        """Generate reasons why this career matches the user"""
        reasons = []
        
        # Personality-based reasons
        user_personality = user_profile.get('personality', {}).get('personality_profile', {})
        required_traits = career_profile.get('personality_traits', {})
        
        for trait, required_level in required_traits.items():
            if trait in user_personality:
                user_score = user_personality[trait].get('score', 3)
                if required_level in ['high', 'very_high'] and user_score >= 4:
                    trait_name = trait.replace('_', ' ').title()
                    reasons.append(f"Your high {trait_name} aligns well with this role")
        
        # Skills-based reasons
        user_skills = user_profile.get('skills', {}).get('skill_profile', {}).get('all_skills', {})
        required_skills = career_profile.get('required_skills', [])
        
        matched_skills = [skill for skill in required_skills if skill in user_skills and user_skills[skill] >= 3]
        if matched_skills:
            if len(matched_skills) == 1:
                reasons.append(f"You have strong skills in {matched_skills[0]}")
            else:
                reasons.append(f"You have strong skills in {', '.join(matched_skills[:2])}")
        
        # Market demand reason
        if career_profile.get('job_market_demand') in ['High', 'Very High']:
            reasons.append("High market demand for this role")
        
        # Growth prospects reason
        if career_profile.get('growth_prospects') in ['High', 'Very High']:
            reasons.append("Excellent career growth opportunities")
        
        return reasons[:3]  # Return top 3 reasons
    
    def _identify_skill_gaps(self, user_profile: Dict, career_profile: Dict) -> List[str]:
        """Identify skill gaps for the career"""
        user_skills = user_profile.get('skills', {}).get('skill_profile', {}).get('all_skills', {})
        required_skills = career_profile.get('required_skills', [])
        
        gaps = []
        for skill in required_skills:
            if skill not in user_skills:
                gaps.append(skill)
            elif user_skills[skill] < 3:
                gaps.append(f"{skill} (needs improvement)")
        
        return gaps[:5]  # Return top 5 gaps
    
    def _fallback_recommendations(self, user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Provide fallback recommendations when main system fails"""
        return [
            {
                'career_id': 'software_engineer',
                'career_title': 'Software Engineer',
                'career_category': 'Technology',
                'career_description': 'Develop software applications and systems',
                'final_score': 0.85,
                'compatibility_score': 85.0,
                'salary_range': {'formatted': '₹6-12 LPA'},
                'growth_outlook': 'High',
                'skill_analysis': {'skill_match_percentage': 78},
                'match_reasons': ['High demand in the market', 'Good growth opportunities'],
                'skill_gaps': ['Advanced Python', 'System Design']
            },
            {
                'career_id': 'data_scientist',
                'career_title': 'Data Scientist',
                'career_category': 'Technology',
                'career_description': 'Analyze data to drive business decisions',
                'final_score': 0.78,
                'compatibility_score': 78.0,
                'salary_range': {'formatted': '₹8-15 LPA'},
                'growth_outlook': 'Very High',
                'skill_analysis': {'skill_match_percentage': 72},
                'match_reasons': ['Growing field', 'High salary potential'],
                'skill_gaps': ['Machine Learning', 'Statistics']
            }
        ]
    
    def get_career_details(self, career_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific career"""
        career_profile = self._find_career_profile(career_id)
        
        if not career_profile:
            return None
        
        # Add market insights if available
        try:
            if self.data_loader:
                market_data = self.data_loader.get_job_market_data()
                career_market_info = market_data.get(career_id, {})
            else:
                career_market_info = {}
        except:
            career_market_info = {}
        
        return {
            **career_profile,
            'market_insights': career_market_info,
            'last_updated': datetime.now().isoformat()
        }