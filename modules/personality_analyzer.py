"""
Personality Analyzer Module
Analyzes user personality traits using rule-based assessment
"""

import json
import numpy as np
from typing import Dict, List, Any

class PersonalityAnalyzer:
    def __init__(self):
        """Initialize the personality analyzer with rule-based system"""
        self.personality_traits = [
            'openness', 'conscientiousness', 'extraversion', 
            'agreeableness', 'neuroticism'
        ]
        
        # Question mapping to personality traits
        self.question_mapping = {
            'q1': 'openness',      # "I enjoy trying new things and exploring different ideas"
            'q2': 'conscientiousness',  # "I prefer to plan ahead and organize my work carefully"
            'q3': 'extraversion',  # "I feel energized when working with large groups of people"
            'q4': 'agreeableness', # "I often put others' needs before my own"
            'q5': 'neuroticism',   # "I remain calm under pressure and stressful situations" (inverted)
            'q6': 'extraversion',  # "I prefer working independently rather than in teams" (inverted)
            'q7': 'openness',      # "I enjoy solving complex problems and puzzles"
            'q8': 'extraversion'   # "I like to take charge and lead others"
        }
        
        # Inverted questions (higher score = lower trait)
        self.inverted_questions = {'q5', 'q6'}
    
    def analyze_responses(self, responses: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze user responses to determine personality traits using rule-based system
        
        Args:
            responses: Dictionary containing user responses to personality questions
            
        Returns:
            Dictionary with personality analysis results
        """
        try:
            # Calculate Big Five scores from responses
            big_five_scores = self._calculate_big_five_scores(responses)
            
            # Generate personality characteristics
            characteristics = self._generate_characteristics(big_five_scores)
            
            # Generate work style description
            work_style = self._generate_work_style(big_five_scores)
            
            # Generate communication style
            communication_style = self._generate_communication_style(big_five_scores)
            
            # Generate leadership tendencies
            leadership_tendencies = self._generate_leadership_tendencies(big_five_scores)
            
            # Generate stress management approach
            stress_management = self._generate_stress_management(big_five_scores)
            
            personality_profile = {
                "big_five_scores": big_five_scores,
                "characteristics": characteristics,
                "work_style": work_style,
                "communication_style": communication_style,
                "leadership_tendencies": leadership_tendencies,
                "stress_management": stress_management
            }
            
            return {
                'success': True,
                'personality_profile': personality_profile,
                'recommendations': self._generate_recommendations(personality_profile)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'personality_profile': None
            }
    
    def _calculate_big_five_scores(self, responses: Dict[str, Any]) -> Dict[str, int]:
        """Calculate Big Five personality scores from responses"""
        trait_scores = {trait: [] for trait in self.personality_traits}
        
        # Process each response
        for question_id, score in responses.items():
            if question_id in self.question_mapping:
                trait = self.question_mapping[question_id]
                
                # Handle inverted questions
                if question_id in self.inverted_questions:
                    # Invert the score (1->5, 2->4, 3->3, 4->2, 5->1)
                    adjusted_score = 6 - score
                else:
                    adjusted_score = score
                
                trait_scores[trait].append(adjusted_score)
        
        # Calculate average scores and convert to 0-100 scale
        big_five_scores = {}
        for trait, scores in trait_scores.items():
            if scores:
                # Average the scores and convert to 0-100 scale
                # (score - 1) / 4 * 100, where score is 1-5
                average_score = np.mean(scores)
                scaled_score = int((average_score - 1) / 4 * 100)
                big_five_scores[trait] = max(0, min(100, scaled_score))
            else:
                # Default to neutral if no responses
                big_five_scores[trait] = 50
        
        return big_five_scores
    
    def _generate_characteristics(self, scores: Dict[str, int]) -> List[str]:
        """Generate personality characteristics based on Big Five scores"""
        characteristics = []
        
        if scores['openness'] > 70:
            characteristics.extend(['Creative', 'Innovative', 'Curious'])
        elif scores['openness'] < 30:
            characteristics.extend(['Practical', 'Traditional', 'Conventional'])
        
        if scores['conscientiousness'] > 70:
            characteristics.extend(['Organized', 'Reliable', 'Detail-oriented'])
        elif scores['conscientiousness'] < 30:
            characteristics.extend(['Flexible', 'Spontaneous', 'Adaptable'])
        
        if scores['extraversion'] > 70:
            characteristics.extend(['Outgoing', 'Energetic', 'Social'])
        elif scores['extraversion'] < 30:
            characteristics.extend(['Reserved', 'Thoughtful', 'Independent'])
        
        if scores['agreeableness'] > 70:
            characteristics.extend(['Cooperative', 'Trusting', 'Helpful'])
        elif scores['agreeableness'] < 30:
            characteristics.extend(['Competitive', 'Direct', 'Assertive'])
        
        if scores['neuroticism'] < 30:
            characteristics.extend(['Calm', 'Confident', 'Resilient'])
        elif scores['neuroticism'] > 70:
            characteristics.extend(['Sensitive', 'Emotional', 'Empathetic'])
        
        return characteristics[:6]  # Limit to 6 characteristics
    
    def _generate_work_style(self, scores: Dict[str, int]) -> str:
        """Generate work style description based on personality scores"""
        style_elements = []
        
        if scores['conscientiousness'] > 70:
            style_elements.append("methodical and organized approach")
        elif scores['conscientiousness'] < 30:
            style_elements.append("flexible and adaptive approach")
        
        if scores['extraversion'] > 70:
            style_elements.append("collaborative and team-oriented")
        elif scores['extraversion'] < 30:
            style_elements.append("independent and focused")
        
        if scores['openness'] > 70:
            style_elements.append("creative and innovative")
        elif scores['openness'] < 30:
            style_elements.append("practical and systematic")
        
        if not style_elements:
            style_elements.append("balanced and versatile")
        
        return f"Prefers a {', '.join(style_elements)} to tasks"
    
    def _generate_communication_style(self, scores: Dict[str, int]) -> str:
        """Generate communication style based on personality scores"""
        if scores['extraversion'] > 70 and scores['agreeableness'] > 60:
            return "Warm, engaging, and collaborative communication style"
        elif scores['extraversion'] > 70 and scores['agreeableness'] < 40:
            return "Direct, assertive, and results-focused communication"
        elif scores['extraversion'] < 30 and scores['agreeableness'] > 60:
            return "Thoughtful, considerate, and diplomatic communication"
        elif scores['extraversion'] < 30 and scores['agreeableness'] < 40:
            return "Concise, analytical, and objective communication"
        else:
            return "Balanced and adaptable communication style"
    
    def _generate_leadership_tendencies(self, scores: Dict[str, int]) -> str:
        """Generate leadership tendencies based on personality scores"""
        if scores['extraversion'] > 70 and scores['conscientiousness'] > 60:
            return "Natural leader with strong organizational skills and team motivation"
        elif scores['extraversion'] > 70 and scores['agreeableness'] > 60:
            return "Collaborative leader who builds consensus and supports team members"
        elif scores['conscientiousness'] > 70 and scores['neuroticism'] < 30:
            return "Reliable leader who leads by example and maintains high standards"
        elif scores['openness'] > 70 and scores['extraversion'] > 60:
            return "Innovative leader who inspires change and creative thinking"
        else:
            return "Supportive leadership style that adapts to team needs"
    
    def _generate_stress_management(self, scores: Dict[str, int]) -> str:
        """Generate stress management approach based on personality scores"""
        if scores['neuroticism'] < 30 and scores['conscientiousness'] > 60:
            return "Handles stress through planning, organization, and systematic problem-solving"
        elif scores['neuroticism'] < 30 and scores['extraversion'] > 60:
            return "Manages stress through social support and collaborative problem-solving"
        elif scores['openness'] > 70 and scores['neuroticism'] < 40:
            return "Uses creativity and new perspectives to overcome challenges"
        elif scores['agreeableness'] > 70:
            return "Seeks support from others and focuses on team well-being during stress"
        else:
            return "Uses a balanced approach combining planning, support, and adaptability"
    
    def _generate_recommendations(self, personality_profile: Dict[str, Any]) -> List[str]:
        """Generate career recommendations based on personality profile"""
        recommendations = []
        
        # Analyze Big Five scores
        scores = personality_profile.get('big_five_scores', {})
        
        if scores.get('openness', 50) > 70:
            recommendations.append("Consider creative and innovative career paths")
        
        if scores.get('conscientiousness', 50) > 70:
            recommendations.append("Well-suited for roles requiring attention to detail and reliability")
        
        if scores.get('extraversion', 50) > 70:
            recommendations.append("Thrive in people-oriented and leadership positions")
        
        if scores.get('agreeableness', 50) > 70:
            recommendations.append("Excel in collaborative and service-oriented roles")
        
        if scores.get('neuroticism', 50) < 30:
            recommendations.append("Handle high-pressure environments well")
        
        return recommendations
    
    def get_personality_questions(self) -> List[Dict[str, Any]]:
        """Get a set of personality assessment questions"""
        return [
            {
                "id": "q1",
                "question": "I enjoy trying new things and exploring different ideas",
                "type": "likert",
                "scale": 5
            },
            {
                "id": "q2", 
                "question": "I prefer to plan ahead and organize my work carefully",
                "type": "likert",
                "scale": 5
            },
            {
                "id": "q3",
                "question": "I feel energized when working with large groups of people",
                "type": "likert", 
                "scale": 5
            },
            {
                "id": "q4",
                "question": "I often put others' needs before my own",
                "type": "likert",
                "scale": 5
            },
            {
                "id": "q5",
                "question": "I remain calm under pressure and stressful situations",
                "type": "likert",
                "scale": 5
            },
            {
                "id": "q6",
                "question": "I prefer working independently rather than in teams",
                "type": "likert",
                "scale": 5
            },
            {
                "id": "q7",
                "question": "I enjoy solving complex problems and puzzles",
                "type": "likert",
                "scale": 5
            },
            {
                "id": "q8",
                "question": "I like to take charge and lead others",
                "type": "likert",
                "scale": 5
            }
        ]
