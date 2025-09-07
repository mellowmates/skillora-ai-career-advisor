from typing import Dict, List, Any, Optional, Tuple
import json
import logging

class PersonalityAssessor:
    """Big Five personality assessment and analysis"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.personality_data = data_loader.get_personality_traits()
        
        # Big Five traits with assessment questions
        self.traits = {
            'openness': {
                'name': 'Openness to Experience',
                'description': 'Creativity, curiosity, intellectual pursuits',
                'questions': [
                    "I enjoy exploring new ideas and concepts",
                    "I appreciate art, music, and creative expressions",
                    "I like to try new and different activities"
                ]
            },
            'conscientiousness': {
                'name': 'Conscientiousness',
                'description': 'Organization, responsibility, self-discipline',
                'questions': [
                    "I am always prepared and organized",
                    "I pay attention to details and follow through on commitments",
                    "I work systematically towards my goals"
                ]
            },
            'extraversion': {
                'name': 'Extraversion',
                'description': 'Sociability, energy, positive emotions',
                'questions': [
                    "I feel energized when around other people",
                    "I enjoy being the center of attention in social situations",
                    "I prefer working in teams rather than alone"
                ]
            },
            'agreeableness': {
                'name': 'Agreeableness', 
                'description': 'Trust, cooperation, empathy',
                'questions': [
                    "I try to be cooperative and avoid conflicts",
                    "I am sympathetic and concerned about others' problems",
                    "I trust people and assume they have good intentions"
                ]
            },
            'neuroticism': {
                'name': 'Neuroticism',
                'description': 'Emotional stability, stress tolerance',
                'questions': [
                    "I often worry about things that might go wrong",
                    "I get stressed easily under pressure",
                    "My mood changes frequently throughout the day"
                ]
            }
        }
        
        self.assessment_results = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_assessment_questions(self) -> Dict[str, List[str]]:
        """Get personality assessment questions"""
        questions = {}
        for trait, data in self.traits.items():
            questions[trait] = data['questions']
        return questions
    
    def analyze_personality(self, responses: Dict[str, List[int]]) -> Dict[str, Any]:
        """
        Analyze personality based on responses to assessment questions
        
        Args:
            responses: Dict with trait names as keys and list of scores (1-10) as values
                      
        Returns:
            Dict containing personality analysis results
        """
        self.logger.info("Starting personality analysis...")
        
        # Calculate trait scores
        trait_scores = {}
        for trait, scores in responses.items():
            if trait in self.traits and scores:
                trait_scores[trait] = sum(scores) / len(scores)  # Average score
        
        # Interpret scores
        personality_profile = self._interpret_scores(trait_scores)
        
        # Get career correlations
        career_matches = self._get_career_correlations(trait_scores)
        
        # Generate detailed analysis
        detailed_analysis = self._generate_detailed_analysis(trait_scores)
        
        self.assessment_results = {
            'trait_scores': trait_scores,
            'personality_profile': personality_profile,
            'career_correlations': career_matches,
            'detailed_analysis': detailed_analysis,
            'assessment_date': self._get_current_date()
        }
        
        self.logger.info("Personality analysis completed")
        return self.assessment_results
    
    def _interpret_scores(self, trait_scores: Dict[str, float]) -> Dict[str, Dict]:
        """Interpret trait scores into personality profile"""
        profile = {}
        
        for trait, score in trait_scores.items():
            if score >= 8:
                level = "Very High"
            elif score >= 6.5:
                level = "High"
            elif score >= 4.5:
                level = "Moderate"
            elif score >= 3:
                level = "Low"
            else:
                level = "Very Low"
            
            profile[trait] = {
                'score': round(score, 2),
                'level': level,
                'description': self.traits[trait]['description'],
                'interpretation': self._get_trait_interpretation(trait, level)
            }
        
        return profile
    
    def _get_trait_interpretation(self, trait: str, level: str) -> str:
        """Get interpretation for specific trait and level"""
        interpretations = {
            'openness': {
                'Very High': "Highly creative, curious, and open to new experiences. Enjoys intellectual challenges.",
                'High': "Creative and curious, enjoys learning and exploring new ideas.",
                'Moderate': "Balanced approach to new experiences, selective about changes.",
                'Low': "Prefers familiar routines and traditional approaches.",
                'Very Low': "Strongly prefers routine and conventional methods."
            },
            'conscientiousness': {
                'Very High': "Extremely organized, disciplined, and reliable. Sets and achieves high standards.",
                'High': "Well-organized, responsible, and goal-oriented.",
                'Moderate': "Generally organized with good work habits.",
                'Low': "Sometimes disorganized, may struggle with long-term planning.",
                'Very Low': "Often disorganized and may have difficulty meeting commitments."
            },
            'extraversion': {
                'Very High': "Highly sociable, energetic, and outgoing. Thrives in social situations.",
                'High': "Sociable and energetic, enjoys interacting with others.",
                'Moderate': "Comfortable in both social and solitary situations.",
                'Low': "Prefers quieter environments and smaller groups.",
                'Very Low': "Strongly prefers solitude and quiet activities."
            },
            'agreeableness': {
                'Very High': "Extremely cooperative, trusting, and empathetic. Avoids conflicts.",
                'High': "Cooperative, trusting, and concerned about others.",
                'Moderate': "Generally cooperative but can be assertive when needed.",
                'Low': "Can be competitive and skeptical of others' motives.",
                'Very Low': "Often competitive, suspicious, and less concerned with others' needs."
            },
            'neuroticism': {
                'Very High': "Experiences frequent stress, anxiety, and emotional ups and downs.",
                'High': "Tends to worry and can be emotionally reactive to stress.",
                'Moderate': "Generally emotionally stable with occasional stress responses.",
                'Low': "Usually calm and emotionally stable under pressure.",
                'Very Low': "Very emotionally stable, rarely anxious or stressed."
            }
        }
        
        return interpretations.get(trait, {}).get(level, "No interpretation available")
    
    def _get_career_correlations(self, trait_scores: Dict[str, float]) -> List[Dict]:
        """Get career recommendations based on personality profile"""
        career_matches = []
        
        # Load personality-career correlations from data
        personality_careers = self.personality_data.get('big_five_model', {})
        
        # Calculate compatibility scores for different career types
        if 'personality_combinations' in self.personality_data:
            combinations = self.personality_data['personality_combinations']
            
            for combo_name, combo_data in combinations.items():
                compatibility_score = self._calculate_compatibility(trait_scores, combo_data['traits'])
                
                if compatibility_score > 0.6:  # Threshold for good match
                    career_matches.append({
                        'profile_type': combo_name,
                        'compatibility_score': round(compatibility_score, 3),
                        'description': combo_data.get('description', ''),
                        'best_careers': combo_data.get('best_careers', [])
                    })
        
        # Sort by compatibility score
        career_matches.sort(key=lambda x: x['compatibility_score'], reverse=True)
        
        return career_matches[:5]  # Return top 5 matches
    
    def _calculate_compatibility(self, user_scores: Dict[str, float], required_traits: Dict[str, str]) -> float:
        """Calculate compatibility between user scores and required traits"""
        total_compatibility = 0
        valid_traits = 0
        
        for trait, required_level in required_traits.items():
            if trait in user_scores:
                user_score = user_scores[trait]
                
                # Map required level to score range
                if required_level == 'high':
                    target_score = 7.5
                elif required_level == 'medium':
                    target_score = 5.0
                elif required_level == 'low':
                    target_score = 2.5
                else:
                    continue
                
                # Calculate compatibility (closer = better)
                difference = abs(user_score - target_score)
                compatibility = max(0, 1 - (difference / 5))  # Normalize to 0-1
                
                total_compatibility += compatibility
                valid_traits += 1
        
        return total_compatibility / valid_traits if valid_traits > 0 else 0
    
    def _generate_detailed_analysis(self, trait_scores: Dict[str, float]) -> Dict[str, Any]:
        """Generate detailed personality analysis"""
        analysis = {}
        
        # Overall personality summary
        dominant_traits = sorted(trait_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        analysis['dominant_traits'] = [trait for trait, score in dominant_traits]
        
        # Work style preferences
        analysis['work_style'] = self._determine_work_style(trait_scores)
        
        # Learning preferences
        analysis['learning_style'] = self._determine_learning_style(trait_scores)
        
        # Leadership potential
        analysis['leadership_potential'] = self._assess_leadership_potential(trait_scores)
        
        # Stress management
        analysis['stress_management'] = self._assess_stress_management(trait_scores)
        
        return analysis
    
    def _determine_work_style(self, scores: Dict[str, float]) -> Dict[str, str]:
        """Determine work style preferences"""
        style = {}
        
        # Team vs Individual work
        if scores.get('extraversion', 5) > 6:
            style['collaboration'] = "Prefers collaborative team environments"
        else:
            style['collaboration'] = "Works well independently or in small groups"
        
        # Structure vs Flexibility
        if scores.get('conscientiousness', 5) > 7:
            style['organization'] = "Prefers structured, well-organized work environments"
        else:
            style['organization'] = "Adaptable to various organizational structures"
        
        # Innovation vs Tradition
        if scores.get('openness', 5) > 6:
            style['innovation'] = "Enjoys innovative projects and creative problem-solving"
        else:
            style['innovation'] = "Comfortable with established procedures and methods"
        
        return style
    
    def _determine_learning_style(self, scores: Dict[str, float]) -> Dict[str, str]:
        """Determine learning preferences"""
        style = {}
        
        if scores.get('openness', 5) > 6:
            style['approach'] = "Enjoys exploring new concepts and learning methods"
        else:
            style['approach'] = "Prefers structured, traditional learning approaches"
        
        if scores.get('conscientiousness', 5) > 7:
            style['pace'] = "Benefits from systematic, well-planned learning schedules"
        else:
            style['pace'] = "Adaptable to various learning paces and formats"
        
        return style
    
    def _assess_leadership_potential(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """Assess leadership potential based on personality traits"""
        leadership_score = 0
        leadership_traits = []
        
        # Extraversion contributes to leadership
        if scores.get('extraversion', 5) > 6:
            leadership_score += 2
            leadership_traits.append("Natural communicator")
        
        # Conscientiousness for reliability
        if scores.get('conscientiousness', 5) > 6:
            leadership_score += 2
            leadership_traits.append("Reliable and organized")
        
        # Openness for vision
        if scores.get('openness', 5) > 6:
            leadership_score += 1
            leadership_traits.append("Visionary and innovative")
        
        # Agreeableness for team building
        if scores.get('agreeableness', 5) > 6:
            leadership_score += 1
            leadership_traits.append("Team-oriented")
        
        # Low neuroticism for stability
        if scores.get('neuroticism', 5) < 4:
            leadership_score += 2
            leadership_traits.append("Emotionally stable under pressure")
        
        if leadership_score >= 6:
            potential = "High"
        elif leadership_score >= 4:
            potential = "Moderate"
        else:
            potential = "Developing"
        
        return {
            'potential': potential,
            'score': leadership_score,
            'strengths': leadership_traits
        }
    
    def _assess_stress_management(self, scores: Dict[str, float]) -> Dict[str, str]:
        """Assess stress management capabilities"""
        neuroticism_score = scores.get('neuroticism', 5)
        conscientiousness_score = scores.get('conscientiousness', 5)
        
        if neuroticism_score < 3 and conscientiousness_score > 6:
            management = "Excellent stress management and resilience"
        elif neuroticism_score < 5:
            management = "Good stress management with effective coping strategies"
        elif neuroticism_score > 7:
            management = "May need stress management support and coping techniques"
        else:
            management = "Moderate stress management abilities"
        
        return {'assessment': management}
    
    def get_career_recommendations(self, personality_results: Dict) -> List[Dict]:
        """Get specific career recommendations based on personality assessment"""
        if not personality_results.get('career_correlations'):
            return []
        
        recommendations = []
        for match in personality_results['career_correlations']:
            for career in match.get('best_careers', []):
                recommendations.append({
                    'career': career,
                    'compatibility_score': match['compatibility_score'],
                    'personality_fit': match['profile_type'],
                    'reasoning': f"Strong match with {match['description']}"
                })
        
        return recommendations
    
    def _get_current_date(self) -> str:
        """Get current date in ISO format"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def save_assessment_results(self, user_id: str, results: Dict) -> bool:
        """Save personality assessment results"""
        try:
            # In a real implementation, this would save to database
            # For now, we'll log the successful assessment
            self.logger.info(f"Personality assessment completed for user {user_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving assessment results: {str(e)}")
            return False
