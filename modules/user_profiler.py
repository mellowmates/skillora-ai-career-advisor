from typing import Dict, List, Any, Optional
import re
from datetime import datetime

class UserProfiler:
    """Handles user profile creation and management with multi-education support"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.education_paths = data_loader.get_education_paths()
    
    def create_profile(self, profile_data: Dict) -> Dict:
        """Create comprehensive user profile"""
        profile = {
            'user_id': profile_data.get('user_id'),
            'timestamp': datetime.now().isoformat(),
            'demographics': self._process_demographics(profile_data.get('demographics', {})),
            'education': self._process_education(profile_data.get('education', {})),
            'experience': self._process_experience(profile_data.get('experience', {})),
            'preferences': self._process_preferences(profile_data.get('preferences', {})),
            'goals': self._process_goals(profile_data.get('goals', {}))
        }
        
        return profile
    
    def _process_demographics(self, demo_data: Dict) -> Dict:
        """Process demographic information"""
        return {
            'age': self._validate_age(demo_data.get('age')),
            'gender': demo_data.get('gender', '').lower(),
            'location': {
                'city': demo_data.get('city', '').title(),
                'state': demo_data.get('state', '').title(),
                'country': demo_data.get('country', 'India')
            },
            'languages': demo_data.get('languages', ['English', 'Hindi'])
        }
    
    def _process_education(self, edu_data: Dict) -> Dict:
        """Process comprehensive education data"""
        education = {
            'current_level': edu_data.get('current_level', ''),
            'completed_degrees': [],
            'ongoing_studies': {},
            'certifications': [],
            'specialized_training': []
        }
        
        # Process completed degrees
        for degree in edu_data.get('completed_degrees', []):
            processed_degree = {
                'type': degree.get('type', ''),  # B.Tech, B.E., BCA, etc.
                'specialization': degree.get('specialization', ''),
                'institution': degree.get('institution', ''),
                'year_completed': degree.get('year_completed'),
                'grade': degree.get('grade', ''),
                'relevant_projects': degree.get('projects', [])
            }
            education['completed_degrees'].append(processed_degree)
        
        # Process ongoing studies
        if 'ongoing_studies' in edu_data:
            education['ongoing_studies'] = {
                'type': edu_data['ongoing_studies'].get('type', ''),
                'specialization': edu_data['ongoing_studies'].get('specialization', ''),
                'expected_completion': edu_data['ongoing_studies'].get('expected_completion'),
                'current_year': edu_data['ongoing_studies'].get('current_year', 1)
            }
        
        # Process certifications
        for cert in edu_data.get('certifications', []):
            education['certifications'].append({
                'name': cert.get('name', ''),
                'provider': cert.get('provider', ''),
                'completion_date': cert.get('completion_date'),
                'validity_period': cert.get('validity_period'),
                'skill_areas': cert.get('skill_areas', [])
            })
        
        return education
    
    def _process_experience(self, exp_data: Dict) -> Dict:
        """Process work and project experience"""
        return {
            'total_experience': exp_data.get('total_experience', 0),  # in years
            'work_experience': self._process_work_experience(exp_data.get('work_experience', [])),
            'internships': self._process_internships(exp_data.get('internships', [])),
            'projects': self._process_projects(exp_data.get('projects', [])),
            'volunteer_work': exp_data.get('volunteer_work', [])
        }
    
    def _process_work_experience(self, work_exp: List) -> List:
        """Process work experience"""
        processed_exp = []
        for exp in work_exp:
            processed_exp.append({
                'company': exp.get('company', ''),
                'position': exp.get('position', ''),
                'duration': exp.get('duration', ''),
                'responsibilities': exp.get('responsibilities', []),
                'technologies_used': exp.get('technologies_used', []),
                'achievements': exp.get('achievements', [])
            })
        return processed_exp
    
    def _process_internships(self, internships: List) -> List:
        """Process internship experience"""
        processed_internships = []
        for internship in internships:
            processed_internships.append({
                'company': internship.get('company', ''),
                'role': internship.get('role', ''),
                'duration': internship.get('duration', ''),
                'key_learnings': internship.get('key_learnings', []),
                'mentor_feedback': internship.get('mentor_feedback', '')
            })
        return processed_internships
    
    def _process_projects(self, projects: List) -> List:
        """Process project experience"""
        processed_projects = []
        for project in projects:
            processed_projects.append({
                'name': project.get('name', ''),
                'description': project.get('description', ''),
                'technologies': project.get('technologies', []),
                'duration': project.get('duration', ''),
                'team_size': project.get('team_size', 1),
                'your_role': project.get('your_role', ''),
                'outcomes': project.get('outcomes', []),
                'github_link': project.get('github_link', ''),
                'demo_link': project.get('demo_link', '')
            })
        return processed_projects
    
    def _process_preferences(self, pref_data: Dict) -> Dict:
        """Process career preferences"""
        return {
            'work_environment': {
                'remote': pref_data.get('remote_preference', 'flexible'),
                'company_size': pref_data.get('company_size', 'no_preference'),
                'industry_preferences': pref_data.get('industry_preferences', [])
            },
            'role_preferences': {
                'leadership_interest': pref_data.get('leadership_interest', False),
                'client_interaction': pref_data.get('client_interaction', 'medium'),
                'travel_willingness': pref_data.get('travel_willingness', 'occasional')
            },
            'compensation': {
                'salary_expectation': pref_data.get('salary_expectation', {}),
                'benefits_priority': pref_data.get('benefits_priority', []),
                'equity_interest': pref_data.get('equity_interest', False)
            },
            'learning_preferences': {
                'continuous_learning': pref_data.get('continuous_learning', True),
                'mentorship_interest': pref_data.get('mentorship_interest', True),
                'conference_participation': pref_data.get('conference_participation', False)
            }
        }
    
    def _process_goals(self, goals_data: Dict) -> Dict:
        """Process career goals"""
        return {
            'short_term': goals_data.get('short_term', []),  # 1-2 years
            'medium_term': goals_data.get('medium_term', []), # 3-5 years
            'long_term': goals_data.get('long_term', []),    # 5+ years
            'skill_development_priorities': goals_data.get('skill_development_priorities', []),
            'geographic_preferences': goals_data.get('geographic_preferences', [])
        }
    
    def _validate_age(self, age) -> Optional[int]:
        """Validate and return age"""
        if age and isinstance(age, (int, str)):
            try:
                age_int = int(age)
                if 16 <= age_int <= 65:
                    return age_int
            except ValueError:
                pass
        return None
    
    def update_profile(self, user_id: str, updates: Dict) -> Dict:
        """Update existing user profile"""
        # In a real application, you'd load from database
        # For now, we'll work with session data
        updated_profile = {}
        # Add update logic here
        return updated_profile
    
    def get_education_recommendations(self, profile: Dict) -> List[Dict]:
        """Get education recommendations based on profile"""
        recommendations = []
        current_education = profile.get('education', {})
        career_goals = profile.get('goals', {})
        
        # Add logic to recommend additional education/certifications
        # based on career goals and current education level
        
        return recommendations
