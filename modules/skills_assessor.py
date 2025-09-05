"""
Skills Assessor Module
Evaluates user skills and capabilities for career matching
"""

import json
from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SkillsAssessor:
    def __init__(self):
        """Initialize the skills assessor"""
        self.skill_categories = {
            'technical': ['programming', 'data_analysis', 'design', 'engineering'],
            'soft': ['communication', 'leadership', 'problem_solving', 'teamwork'],
            'domain': ['business', 'healthcare', 'education', 'finance'],
            'tools': ['software', 'hardware', 'platforms', 'methodologies']
        }
        
        self.skill_levels = {
            'beginner': 1,
            'intermediate': 2, 
            'advanced': 3,
            'expert': 4
        }
    
    def assess_skills(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess user skills based on input data
        
        Args:
            user_input: Dictionary containing user skill information
            
        Returns:
            Dictionary with skills assessment results
        """
        try:
            # Extract skills from different input sources
            skills_data = self._extract_skills_data(user_input)
            
            # Analyze skill levels and gaps
            skill_analysis = self._analyze_skill_levels(skills_data)
            
            # Generate skill recommendations
            recommendations = self._generate_skill_recommendations(skill_analysis)
            
            # Calculate overall skill score
            overall_score = self._calculate_overall_score(skill_analysis)
            
            return {
                'success': True,
                'skills_profile': skill_analysis,
                'overall_score': overall_score,
                'recommendations': recommendations,
                'skill_gaps': self._identify_skill_gaps(skill_analysis)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'skills_profile': None
            }
    
    def _extract_skills_data(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and structure skills data from user input"""
        skills_data = {
            'self_assessed': {},
            'experience_based': {},
            'education_based': {},
            'certification_based': {}
        }
        
        # Extract self-assessed skills
        if 'self_assessed_skills' in user_input:
            skills_data['self_assessed'] = user_input['self_assessed_skills']
        
        # Extract skills from work experience
        if 'work_experience' in user_input:
            skills_data['experience_based'] = self._extract_skills_from_experience(
                user_input['work_experience']
            )
        
        # Extract skills from education
        if 'education' in user_input:
            skills_data['education_based'] = self._extract_skills_from_education(
                user_input['education']
            )
        
        # Extract skills from certifications
        if 'certifications' in user_input:
            skills_data['certification_based'] = self._extract_skills_from_certifications(
                user_input['certifications']
            )
        
        return skills_data
    
    def _extract_skills_from_experience(self, experience: Any) -> Dict[str, int]:
        """Extract skills from work experience descriptions"""
        skills = {}
        
        # Handle both string and list inputs
        if isinstance(experience, str):
            # If it's a string, extract skills from the text
            exp_text = experience.lower()
            skills = self._extract_skills_from_text(exp_text)
        elif isinstance(experience, list):
            # If it's a list of dictionaries
            for job in experience:
                if isinstance(job, dict):
                    description = job.get('description', '').lower()
                    title = job.get('title', '').lower()
                    
                    # Map common job titles to skills
                    title_skills = self._map_title_to_skills(title)
                    for skill in title_skills:
                        skills[skill] = skills.get(skill, 0) + 1
                    
                    # Extract skills from description
                    desc_skills = self._extract_skills_from_text(description)
                    for skill, count in desc_skills.items():
                        skills[skill] = skills.get(skill, 0) + count
                elif isinstance(job, str):
                    # Handle list of strings
                    skills.update(self._extract_skills_from_text(job.lower()))
        
        return skills
    
    def _extract_skills_from_education(self, education: Any) -> Dict[str, int]:
        """Extract skills from education background"""
        skills = {}
        
        # Handle both string and list inputs
        if isinstance(education, str):
            # If it's a string, extract skills from the text
            edu_text = education.lower()
            skills = self._extract_skills_from_text(edu_text)
        elif isinstance(education, list):
            # If it's a list of dictionaries
            for edu in education:
                if isinstance(edu, dict):
                    degree = edu.get('degree', '').lower()
                    field = edu.get('field', '').lower()
                    
                    # Map degree fields to skills
                    field_skills = self._map_field_to_skills(field)
                    for skill in field_skills:
                        skills[skill] = skills.get(skill, 0) + 1
                elif isinstance(edu, str):
                    # Handle list of strings
                    skills.update(self._extract_skills_from_text(edu.lower()))
        
        return skills
    
    def _extract_skills_from_certifications(self, certifications: Any) -> Dict[str, int]:
        """Extract skills from certifications"""
        skills = {}
        
        # Handle both string and list inputs
        if isinstance(certifications, str):
            # If it's a string, extract skills from the text
            cert_text = certifications.lower()
            skills = self._extract_skills_from_text(cert_text)
        elif isinstance(certifications, list):
            # If it's a list of dictionaries
            for cert in certifications:
                if isinstance(cert, dict):
                    name = cert.get('name', '').lower()
                    if name:
                        skills[name] = skills.get(name, 0) + 1
                elif isinstance(cert, str):
                    # Handle list of strings
                    skills.update(self._extract_skills_from_text(cert.lower()))
        
        return skills
    
    def _map_title_to_skills(self, title: str) -> List[str]:
        """Map job titles to relevant skills"""
        title_skill_mapping = {
            'developer': ['programming', 'software_development', 'problem_solving'],
            'manager': ['leadership', 'project_management', 'communication'],
            'analyst': ['data_analysis', 'research', 'critical_thinking'],
            'designer': ['design', 'creativity', 'user_experience'],
            'engineer': ['engineering', 'technical_skills', 'problem_solving'],
            'consultant': ['communication', 'business_analysis', 'client_management'],
            'teacher': ['education', 'communication', 'patience'],
            'sales': ['communication', 'persuasion', 'customer_service']
        }
        
        skills = []
        for keyword, skill_list in title_skill_mapping.items():
            if keyword in title:
                skills.extend(skill_list)
        
        return skills
    
    def _map_field_to_skills(self, field: str) -> List[str]:
        """Map education fields to relevant skills"""
        field_skill_mapping = {
            'computer science': ['programming', 'algorithms', 'software_engineering'],
            'business': ['business_analysis', 'management', 'finance'],
            'engineering': ['engineering', 'mathematics', 'problem_solving'],
            'psychology': ['communication', 'research', 'human_behavior'],
            'marketing': ['marketing', 'communication', 'analytics'],
            'finance': ['finance', 'mathematics', 'analytics'],
            'medicine': ['healthcare', 'research', 'patient_care'],
            'education': ['teaching', 'communication', 'curriculum_development']
        }
        
        skills = []
        for keyword, skill_list in field_skill_mapping.items():
            if keyword in field:
                skills.extend(skill_list)
        
        return skills
    
    def _extract_skills_from_text(self, text: str) -> Dict[str, int]:
        """Extract skills from text using keyword matching"""
        skill_keywords = {
            'programming': ['programming', 'coding', 'development', 'software'],
            'data_analysis': ['data', 'analysis', 'analytics', 'statistics'],
            'leadership': ['lead', 'manage', 'supervise', 'direct'],
            'communication': ['communicate', 'present', 'write', 'speak'],
            'project_management': ['project', 'plan', 'coordinate', 'organize'],
            'design': ['design', 'create', 'visual', 'graphic'],
            'research': ['research', 'investigate', 'study', 'analyze'],
            'customer_service': ['customer', 'client', 'support', 'service']
        }
        
        found_skills = {}
        text_lower = text.lower()
        
        for skill, keywords in skill_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                found_skills[skill] = found_skills.get(skill, 0) + 1
        
        return found_skills
    
    def _analyze_skill_levels(self, skills_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and consolidate skill levels from different sources"""
        consolidated_skills = {}
        
        # Combine skills from all sources
        for source, skills in skills_data.items():
            for skill, count in skills.items():
                if skill not in consolidated_skills:
                    consolidated_skills[skill] = 0
                # Ensure count is always an integer
                if isinstance(count, str):
                    consolidated_skills[skill] += 1  # Count each occurrence as 1
                else:
                    consolidated_skills[skill] += int(count)
        
        # Convert counts to skill levels
        skill_levels = {}
        for skill, count in consolidated_skills.items():
            if count >= 3:
                skill_levels[skill] = 'expert'
            elif count >= 2:
                skill_levels[skill] = 'advanced'
            elif count >= 1:
                skill_levels[skill] = 'intermediate'
            else:
                skill_levels[skill] = 'beginner'
        
        return {
            'skills': skill_levels,
            'raw_data': consolidated_skills,
            'categories': self._categorize_skills(skill_levels)
        }
    
    def _categorize_skills(self, skills: Dict[str, str]) -> Dict[str, List[str]]:
        """Categorize skills into different types"""
        categories = {
            'technical': [],
            'soft': [],
            'domain': [],
            'tools': []
        }
        
        for skill, level in skills.items():
            # Simple categorization based on skill names
            if any(tech in skill for tech in ['programming', 'software', 'technical', 'engineering']):
                categories['technical'].append(skill)
            elif any(soft in skill for soft in ['communication', 'leadership', 'teamwork', 'management']):
                categories['soft'].append(skill)
            elif any(domain in skill for domain in ['business', 'healthcare', 'finance', 'education']):
                categories['domain'].append(skill)
            else:
                categories['tools'].append(skill)
        
        return categories
    
    def _generate_skill_recommendations(self, skill_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for skill development"""
        recommendations = []
        skills = skill_analysis['skills']
        categories = skill_analysis['categories']
        
        # Check for missing technical skills
        if not categories['technical']:
            recommendations.append("Consider developing technical skills like programming or data analysis")
        
        # Check for missing soft skills
        if not categories['soft']:
            recommendations.append("Focus on developing soft skills like communication and leadership")
        
        # Recommend advanced skills for intermediate users
        for skill, level in skills.items():
            if level == 'intermediate':
                recommendations.append(f"Consider advancing your {skill} skills to expert level")
        
        return recommendations
    
    def _calculate_overall_score(self, skill_analysis: Dict[str, Any]) -> float:
        """Calculate overall skill score (0-100)"""
        skills = skill_analysis['skills']
        
        if not skills:
            return 0.0
        
        total_score = 0
        for skill, level in skills.items():
            total_score += self.skill_levels.get(level, 1)
        
        # Normalize to 0-100 scale
        max_possible = len(skills) * 4  # 4 is the max skill level
        return (total_score / max_possible) * 100 if max_possible > 0 else 0
    
    def _identify_skill_gaps(self, skill_analysis: Dict[str, Any]) -> List[str]:
        """Identify potential skill gaps for career advancement"""
        gaps = []
        skills = skill_analysis['skills']
        
        # Common skill gaps in different career paths
        common_gaps = {
            'programming': 'Consider learning programming languages like Python or JavaScript',
            'data_analysis': 'Develop data analysis skills using tools like Excel, SQL, or Python',
            'leadership': 'Build leadership and management capabilities',
            'communication': 'Enhance written and verbal communication skills',
            'project_management': 'Learn project management methodologies and tools'
        }
        
        for gap, recommendation in common_gaps.items():
            if gap not in skills or skills[gap] in ['beginner', 'intermediate']:
                gaps.append(recommendation)
        
        return gaps
    
    def get_skill_assessment_questions(self) -> List[Dict[str, Any]]:
        """Get a set of skill assessment questions"""
        return [
            {
                "id": "skill_1",
                "question": "Rate your programming skills",
                "type": "skill_level",
                "options": ["Beginner", "Intermediate", "Advanced", "Expert"]
            },
            {
                "id": "skill_2",
                "question": "How would you rate your communication skills?",
                "type": "skill_level", 
                "options": ["Beginner", "Intermediate", "Advanced", "Expert"]
            },
            {
                "id": "skill_3",
                "question": "Rate your leadership experience",
                "type": "skill_level",
                "options": ["Beginner", "Intermediate", "Advanced", "Expert"]
            },
            {
                "id": "skill_4",
                "question": "How comfortable are you with data analysis?",
                "type": "skill_level",
                "options": ["Beginner", "Intermediate", "Advanced", "Expert"]
            },
            {
                "id": "skill_5",
                "question": "Rate your project management skills",
                "type": "skill_level",
                "options": ["Beginner", "Intermediate", "Advanced", "Expert"]
            }
        ]
