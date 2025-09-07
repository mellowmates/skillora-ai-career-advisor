"""
Learning Roadmap Generator - Creates personalized skill development plans
Provides step-by-step roadmaps with timelines, resources, and milestone tracking
"""

from typing import Dict, List, Any, Optional, Tuple
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict

class LearningRoadmap:
    """Generates personalized learning roadmaps for career development"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.career_profiles = data_loader.get_career_profiles()
        self.skills_mapping = data_loader.get_skills_mapping()
        self.learning_resources = data_loader.get_learning_resources()
        
        # Skill difficulty and time estimation
        self.skill_difficulty = {
            'beginner': {'weeks': 4, 'hours_per_week': 5},
            'intermediate': {'weeks': 8, 'hours_per_week': 6},
            'advanced': {'weeks': 12, 'hours_per_week': 8},
            'expert': {'weeks': 16, 'hours_per_week': 10}
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def generate_roadmap(self, career_path: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive learning roadmap for target career
        
        Args:
            career_path: Target career identifier
            user_data: User profile including current skills, experience, etc.
            
        Returns:
            Detailed roadmap with phases, timelines, and resources
        """
        self.logger.info(f"Generating learning roadmap for {career_path}")
        
        if career_path not in self.career_profiles:
            return {'error': f'Career path {career_path} not found'}
        
        career_data = self.career_profiles[career_path]
        current_skills = set(skill.lower() for skill in user_data.get('skills', []))
        
        # Analyze skill requirements and gaps
        skill_analysis = self._analyze_skill_requirements(career_data, current_skills)
        
        # Create phased learning plan
        learning_phases = self._create_learning_phases(skill_analysis, user_data)
        
        # Generate timeline
        timeline = self._calculate_timeline(learning_phases, user_data)
        
        # Add resource recommendations
        resource_plan = self._recommend_learning_resources(learning_phases)
        
        # Create milestones and progress tracking
        milestones = self._create_milestones(learning_phases, timeline)
        
        roadmap = {
            'career_target': career_path,
            'career_title': career_data.get('title', career_path.replace('_', ' ').title()),
            'generated_date': datetime.now().isoformat(),
            'user_profile_summary': self._summarize_user_profile(user_data),
            'skill_analysis': skill_analysis,
            'learning_phases': learning_phases,
            'timeline': timeline,
            'resource_recommendations': resource_plan,
            'milestones': milestones,
            'estimated_completion': timeline.get('total_duration_weeks', 0),
            'commitment_required': self._calculate_commitment(learning_phases)
        }
        
        self.logger.info(f"Roadmap generated with {len(learning_phases)} phases")
        return roadmap
    
    def _analyze_skill_requirements(self, career_data: Dict, current_skills: set) -> Dict[str, Any]:
        """Analyze skill requirements and identify gaps"""
        required_skills = career_data.get('required_skills', {})
        
        analysis = {
            'skill_categories': {},
            'total_required_skills': 0,
            'current_skill_coverage': 0,
            'skill_gaps_by_priority': {
                'critical': [],
                'important': [],
                'preferred': []
            }
        }
        
        for category, skills_list in required_skills.items():
            if category in ['critical', 'important', 'preferred']:
                matching_skills = []
                missing_skills = []
                
                for skill in skills_list:
                    skill_lower = skill.lower()
                    analysis['total_required_skills'] += 1
                    
                    # Check if user has this skill (flexible matching)
                    has_skill = any(
                        skill_lower in current_skill or current_skill in skill_lower
                        for current_skill in current_skills
                    )
                    
                    if has_skill:
                        matching_skills.append(skill)
                        analysis['current_skill_coverage'] += 1
                    else:
                        missing_skills.append(skill)
                        analysis['skill_gaps_by_priority'][category].append(skill)
                
                analysis['skill_categories'][category] = {
                    'required': skills_list,
                    'matching': matching_skills,
                    'missing': missing_skills,
                    'coverage_percentage': (len(matching_skills) / len(skills_list) * 100) if skills_list else 0
                }
        
        # Overall coverage percentage
        analysis['overall_coverage_percentage'] = (
            analysis['current_skill_coverage'] / analysis['total_required_skills'] * 100
        ) if analysis['total_required_skills'] > 0 else 0
        
        return analysis
    
    def _create_learning_phases(self, skill_analysis: Dict, user_data: Dict) -> List[Dict[str, Any]]:
        """Create phased learning plan based on skill priorities and user profile"""
        phases = []
        
        # Phase 1: Foundation Skills (Critical skills)
        critical_skills = skill_analysis['skill_gaps_by_priority']['critical']
        if critical_skills:
            phases.append({
                'phase_number': 1,
                'phase_name': 'Foundation Building',
                'phase_description': 'Essential skills required for career entry',
                'priority': 'Critical',
                'skills_to_learn': critical_skills,
                'learning_objectives': self._create_learning_objectives(critical_skills),
                'phase_difficulty': 'Beginner to Intermediate',
                'prerequisites': self._identify_prerequisites(critical_skills)
            })
        
        # Phase 2: Core Development (Important skills)
        important_skills = skill_analysis['skill_gaps_by_priority']['important']
        if important_skills:
            phases.append({
                'phase_number': 2,
                'phase_name': 'Core Development',
                'phase_description': 'Important skills for career advancement',
                'priority': 'Important',
                'skills_to_learn': important_skills,
                'learning_objectives': self._create_learning_objectives(important_skills),
                'phase_difficulty': 'Intermediate',
                'prerequisites': phases[0]['skills_to_learn'] if phases else []
            })
        
        # Phase 3: Specialization (Preferred skills)
        preferred_skills = skill_analysis['skill_gaps_by_priority']['preferred']
        if preferred_skills:
            phases.append({
                'phase_number': 3,
                'phase_name': 'Specialization & Advanced Skills',
                'phase_description': 'Advanced skills for career excellence',
                'priority': 'Preferred',
                'skills_to_learn': preferred_skills,
                'learning_objectives': self._create_learning_objectives(preferred_skills),
                'phase_difficulty': 'Advanced',
                'prerequisites': [skill for phase in phases for skill in phase['skills_to_learn']]
            })
        
        # Customize phases based on user experience and goals
        self._customize_phases_for_user(phases, user_data)
        
        return phases
    
    def _create_learning_objectives(self, skills: List[str]) -> List[str]:
        """Create specific learning objectives for skills"""
        objectives = []
        
        for skill in skills:
            # Get skill information from skills mapping
            skill_info = self._find_skill_info(skill)
            
            if skill_info:
                # Create objective based on skill type
                if 'programming' in skill.lower() or 'development' in skill.lower():
                    objectives.append(f"Build practical projects demonstrating {skill} proficiency")
                elif 'management' in skill.lower() or 'leadership' in skill.lower():
                    objectives.append(f"Develop {skill} through case studies and simulations")
                elif 'analysis' in skill.lower() or 'data' in skill.lower():
                    objectives.append(f"Complete real-world {skill} projects with measurable outcomes")
                else:
                    objectives.append(f"Achieve intermediate-level competency in {skill}")
            else:
                objectives.append(f"Master fundamentals and practical applications of {skill}")
        
        return objectives
    
    def _identify_prerequisites(self, skills: List[str]) -> List[str]:
        """Identify prerequisites for learning specific skills"""
        prerequisites = []
        
        for skill in skills:
            skill_lower = skill.lower()
            
            # Programming prerequisites
            if any(lang in skill_lower for lang in ['python', 'java', 'javascript']):
                prerequisites.extend(['Basic programming concepts', 'Problem-solving fundamentals'])
            
            # Data science prerequisites
            elif 'data science' in skill_lower or 'machine learning' in skill_lower:
                prerequisites.extend(['Python programming', 'Statistics basics', 'Mathematics fundamentals'])
            
            # Management prerequisites
            elif 'management' in skill_lower or 'leadership' in skill_lower:
                prerequisites.extend(['Communication skills', 'Basic business understanding'])
            
            # Domain-specific prerequisites
            elif 'accounting' in skill_lower:
                prerequisites.extend(['Basic mathematics', 'Business fundamentals'])
        
        # Remove duplicates and return
        return list(set(prerequisites))
    
    def _customize_phases_for_user(self, phases: List[Dict], user_data: Dict):
        """Customize learning phases based on user's profile and constraints"""
        user_experience = user_data.get('experience_years', 0)
        available_hours_per_week = user_data.get('available_hours_per_week', 10)
        learning_style = user_data.get('learning_preferences', {})
        
        for phase in phases:
            # Adjust difficulty based on experience
            if user_experience > 5:
                if phase['phase_difficulty'] == 'Beginner to Intermediate':
                    phase['phase_difficulty'] = 'Intermediate'
                elif phase['phase_difficulty'] == 'Intermediate':
                    phase['phase_difficulty'] = 'Intermediate to Advanced'
            
            # Add time estimates
            phase['estimated_duration_weeks'] = self._estimate_phase_duration(
                phase['skills_to_learn'], phase['phase_difficulty'], available_hours_per_week
            )
            
            # Add learning approach recommendations
            phase['recommended_approach'] = self._recommend_learning_approach(
                phase['skills_to_learn'], learning_style
            )
    
    def _estimate_phase_duration(self, skills: List[str], difficulty: str, hours_per_week: int) -> int:
        """Estimate duration for learning phase"""
        base_weeks_per_skill = {
            'Beginner': 4,
            'Beginner to Intermediate': 6,
            'Intermediate': 8,
            'Intermediate to Advanced': 10,
            'Advanced': 12
        }
        
        weeks_per_skill = base_weeks_per_skill.get(difficulty, 8)
        total_base_weeks = len(skills) * weeks_per_skill
        
        # Adjust based on available time (assume 8 hours per week as baseline)
        time_adjustment = 8 / hours_per_week if hours_per_week > 0 else 1
        adjusted_weeks = int(total_base_weeks * time_adjustment)
        
        # Account for parallel learning (some skills can be learned together)
        if len(skills) > 1:
            parallel_factor = 0.8  # 20% time savings from parallel learning
            adjusted_weeks = int(adjusted_weeks * parallel_factor)
        
        return max(adjusted_weeks, 2)  # Minimum 2 weeks per phase
    
    def _recommend_learning_approach(self, skills: List[str], learning_style: Dict) -> Dict[str, Any]:
        """Recommend learning approach based on skills and user preferences"""
        approach = {
            'primary_method': 'Mixed Learning',
            'recommended_resources': [],
            'practice_focus': 'Project-based learning',
            'assessment_method': 'Portfolio development'
        }
        
        # Analyze skill types
        technical_skills = [s for s in skills if any(term in s.lower() for term in ['programming', 'development', 'technical'])]
        soft_skills = [s for s in skills if any(term in s.lower() for term in ['communication', 'leadership', 'management'])]
        
        if technical_skills:
            approach['primary_method'] = 'Hands-on coding and projects'
            approach['practice_focus'] = 'Build real-world applications'
            approach['recommended_resources'].extend(['Online coding platforms', 'GitHub repositories', 'Technical documentation'])
        
        if soft_skills:
            approach['recommended_resources'].extend(['Case studies', 'Role-playing exercises', 'Peer feedback sessions'])
        
        # Customize based on learning preferences
        preferred_style = learning_style.get('style', 'visual')
        if preferred_style == 'visual':
            approach['recommended_resources'].append('Video tutorials and infographics')
        elif preferred_style == 'hands-on':
            approach['practice_focus'] = 'Interactive workshops and labs'
        elif preferred_style == 'reading':
            approach['recommended_resources'].append('Comprehensive textbooks and articles')
        
        return approach
    
    def _calculate_timeline(self, learning_phases: List[Dict], user_data: Dict) -> Dict[str, Any]:
        """Calculate detailed timeline for the learning roadmap"""
        start_date = datetime.now()
        current_date = start_date
        
        timeline = {
            'start_date': start_date.date().isoformat(),
            'phases': [],
            'total_duration_weeks': 0,
            'total_duration_months': 0,
            'commitment_hours_per_week': user_data.get('available_hours_per_week', 10)
        }
        
        for phase in learning_phases:
            duration_weeks = phase.get('estimated_duration_weeks', 8)
            end_date = current_date + timedelta(weeks=duration_weeks)
            
            phase_timeline = {
                'phase_number': phase['phase_number'],
                'phase_name': phase['phase_name'],
                'start_date': current_date.date().isoformat(),
                'end_date': end_date.date().isoformat(),
                'duration_weeks': duration_weeks,
                'skills_timeline': self._create_skills_timeline(
                    phase['skills_to_learn'], current_date, duration_weeks
                )
            }
            
            timeline['phases'].append(phase_timeline)
            timeline['total_duration_weeks'] += duration_weeks
            current_date = end_date
        
        timeline['completion_date'] = current_date.date().isoformat()
        timeline['total_duration_months'] = timeline['total_duration_weeks'] / 4.33  # Average weeks per month
        
        return timeline
    
    def _create_skills_timeline(self, skills: List[str], start_date: datetime, total_weeks: int) -> List[Dict]:
        """Create timeline for individual skills within a phase"""
        skills_timeline = []
        
        # Distribute skills across the phase duration
        weeks_per_skill = total_weeks / len(skills) if skills else total_weeks
        current_date = start_date
        
        for i, skill in enumerate(skills):
            skill_duration = max(int(weeks_per_skill), 1)
            skill_end_date = current_date + timedelta(weeks=skill_duration)
            
            skills_timeline.append({
                'skill': skill,
                'start_date': current_date.date().isoformat(),
                'end_date': skill_end_date.date().isoformat(),
                'duration_weeks': skill_duration,
                'learning_milestones': self._create_skill_milestones(skill, skill_duration)
            })
            
            # Allow for some overlap in skill learning
            overlap_weeks = max(int(skill_duration * 0.2), 1)
            current_date = skill_end_date - timedelta(weeks=overlap_weeks)
        
        return skills_timeline
    
    def _create_skill_milestones(self, skill: str, duration_weeks: int) -> List[Dict]:
        """Create learning milestones for individual skills"""
        milestones = []
        
        # Week 1: Fundamentals
        milestones.append({
            'week': 1,
            'milestone': f'Understand {skill} fundamentals and basic concepts',
            'deliverable': 'Complete introductory course or tutorial',
            'assessment': 'Basic knowledge quiz or summary'
        })
        
        # Mid-point: Practice
        mid_week = max(duration_weeks // 2, 2)
        milestones.append({
            'week': mid_week,
            'milestone': f'Apply {skill} in practical exercises',
            'deliverable': 'Complete hands-on project or assignments',
            'assessment': 'Project review and feedback'
        })
        
        # Final week: Mastery demonstration
        if duration_weeks > 2:
            milestones.append({
                'week': duration_weeks,
                'milestone': f'Demonstrate {skill} proficiency',
                'deliverable': 'Portfolio project or certification',
                'assessment': 'Peer review or professional evaluation'
            })
        
        return milestones
    
    def _recommend_learning_resources(self, learning_phases: List[Dict]) -> Dict[str, Any]:
        """Recommend specific learning resources for each phase"""
        resource_plan = {
            'by_phase': [],
            'overall_recommendations': {
                'free_resources': [],
                'paid_resources': [],
                'books': [],
                'online_platforms': [],
                'communities': []
            }
        }
        
        for phase in learning_phases:
            phase_resources = {
                'phase_number': phase['phase_number'],
                'phase_name': phase['phase_name'],
                'resources': []
            }
            
            for skill in phase['skills_to_learn']:
                skill_resources = self._get_skill_resources(skill)
                phase_resources['resources'].extend(skill_resources)
            
            # Remove duplicates
            unique_resources = []
            seen_titles = set()
            for resource in phase_resources['resources']:
                title = resource.get('name') or resource.get('title', '')
                if title not in seen_titles:
                    unique_resources.append(resource)
                    seen_titles.add(title)
            
            phase_resources['resources'] = unique_resources
            resource_plan['by_phase'].append(phase_resources)
        
        # Compile overall recommendations
        self._compile_overall_recommendations(resource_plan)
        
        return resource_plan
    
    def _get_skill_resources(self, skill: str) -> List[Dict]:
        """Get learning resources for a specific skill"""
        resources = []
        
        # Find skill in skills mapping
        skill_info = self._find_skill_info(skill)
        
        if skill_info and 'learning_resources' in skill_info:
            return skill_info['learning_resources']
        
        # Fallback: General recommendations based on skill type
        skill_lower = skill.lower()
        
        if any(term in skill_lower for term in ['python', 'programming']):
            resources.extend([
                {'name': 'Python.org Tutorial', 'type': 'online', 'cost': 'free', 'url': 'https://docs.python.org/3/tutorial/'},
                {'name': 'Codecademy Python', 'type': 'interactive', 'cost': 'freemium'},
                {'name': 'Automate the Boring Stuff', 'type': 'book', 'cost': 'free/paid'}
            ])
        
        elif 'data' in skill_lower or 'analytics' in skill_lower:
            resources.extend([
                {'name': 'Kaggle Learn', 'type': 'online', 'cost': 'free'},
                {'name': 'DataCamp', 'type': 'interactive', 'cost': 'paid'},
                {'name': 'Coursera Data Science', 'type': 'course', 'cost': 'paid'}
            ])
        
        elif any(term in skill_lower for term in ['communication', 'leadership', 'management']):
            resources.extend([
                {'name': 'Toastmasters International', 'type': 'community', 'cost': 'membership'},
                {'name': 'LinkedIn Learning', 'type': 'online', 'cost': 'subscription'},
                {'name': 'Harvard Business Review', 'type': 'articles', 'cost': 'subscription'}
            ])
        
        return resources
    
    def _find_skill_info(self, skill_name: str) -> Optional[Dict]:
        """Find skill information in skills mapping"""
        skill_name_lower = skill_name.lower()
        
        for category, skills_data in self.skills_mapping.items():
            for skill_key, skill_info in skills_data.items():
                skill_key_lower = skill_key.lower()
                skill_display_name = skill_info.get('name', '').lower()
                
                if (skill_name_lower in skill_key_lower or 
                    skill_key_lower in skill_name_lower or
                    skill_name_lower in skill_display_name):
                    return skill_info
        
        return None
    
    def _compile_overall_recommendations(self, resource_plan: Dict):
        """Compile overall resource recommendations"""
        all_resources = []
        for phase in resource_plan['by_phase']:
            all_resources.extend(phase['resources'])
        
        # Categorize resources
        for resource in all_resources:
            resource_type = resource.get('type', '').lower()
            cost = resource.get('cost', '').lower()
            
            if cost in ['free', 'freemium']:
                resource_plan['overall_recommendations']['free_resources'].append(resource)
            else:
                resource_plan['overall_recommendations']['paid_resources'].append(resource)
            
            if resource_type == 'book':
                resource_plan['overall_recommendations']['books'].append(resource)
            elif resource_type in ['online', 'interactive', 'course']:
                resource_plan['overall_recommendations']['online_platforms'].append(resource)
            elif resource_type == 'community':
                resource_plan['overall_recommendations']['communities'].append(resource)
        
        # Remove duplicates from each category
        for category in resource_plan['overall_recommendations']:
            unique_resources = []
            seen_names = set()
            for resource in resource_plan['overall_recommendations'][category]:
                name = resource.get('name', '')
                if name not in seen_names:
                    unique_resources.append(resource)
                    seen_names.add(name)
            resource_plan['overall_recommendations'][category] = unique_resources
    
    def _create_milestones(self, learning_phases: List[Dict], timeline: Dict) -> List[Dict]:
        """Create major milestones for the learning journey"""
        milestones = []
        
        # Phase completion milestones
        for i, phase in enumerate(learning_phases):
            phase_timeline = timeline['phases'][i]
            
            milestones.append({
                'milestone_id': f'phase_{phase["phase_number"]}_complete',
                'title': f'{phase["phase_name"]} Completion',
                'description': f'Successfully complete all skills in {phase["phase_name"]} phase',
                'target_date': phase_timeline['end_date'],
                'type': 'phase_completion',
                'success_criteria': [
                    f'Master {len(phase["skills_to_learn"])} skills',
                    'Complete all recommended projects',
                    'Pass skill assessments',
                    'Build portfolio demonstrating competency'
                ],
                'reward': f'Ready to progress to next phase or apply for {phase["priority"].lower()}-level positions'
            })
        
        # Mid-journey checkpoint
        if len(learning_phases) > 1:
            mid_phase = len(learning_phases) // 2
            mid_date = timeline['phases'][mid_phase - 1]['end_date']
            
            milestones.append({
                'milestone_id': 'mid_journey_checkpoint',
                'title': 'Mid-Journey Assessment',
                'description': 'Evaluate progress and adjust learning plan if needed',
                'target_date': mid_date,
                'type': 'checkpoint',
                'success_criteria': [
                    'Complete self-assessment of learned skills',
                    'Review and update career goals',
                    'Seek feedback from mentors or peers',
                    'Adjust remaining learning plan if necessary'
                ],
                'reward': 'Clear path forward with validated progress'
            })
        
        # Final completion milestone
        completion_date = timeline.get('completion_date')
        if completion_date:
            milestones.append({
                'milestone_id': 'journey_complete',
                'title': 'Career Readiness Achievement',
                'description': 'Complete transformation into target career professional',
                'target_date': completion_date,
                'type': 'final_completion',
                'success_criteria': [
                    'Master all required skills for target career',
                    'Build comprehensive professional portfolio',
                    'Network with industry professionals',
                    'Secure job interviews or career opportunities'
                ],
                'reward': 'Ready to successfully transition into target career'
            })
        
        return sorted(milestones, key=lambda x: x['target_date'])
    
    def _calculate_commitment(self, learning_phases: List[Dict]) -> Dict[str, Any]:
        """Calculate time commitment required for the learning journey"""
        total_skills = sum(len(phase['skills_to_learn']) for phase in learning_phases)
        total_weeks = sum(phase.get('estimated_duration_weeks', 8) for phase in learning_phases)
        
        return {
            'total_skills_to_learn': total_skills,
            'total_duration_weeks': total_weeks,
            'total_duration_months': round(total_weeks / 4.33, 1),
            'average_hours_per_week': 10,  # Default assumption
            'total_learning_hours': total_weeks * 10,
            'commitment_level': 'High' if total_weeks > 40 else 'Medium' if total_weeks > 20 else 'Low',
            'recommendation': self._get_commitment_recommendation(total_weeks)
        }
    
    def _get_commitment_recommendation(self, total_weeks: int) -> str:
        """Get recommendation based on time commitment"""
        if total_weeks > 52:
            return "This is a long-term commitment (1+ year). Consider breaking into smaller goals and maintain consistent progress."
        elif total_weeks > 26:
            return "This requires significant commitment (6+ months). Set up a consistent study schedule and seek support."
        elif total_weeks > 12:
            return "Moderate commitment required (3+ months). Focus on building strong learning habits."
        else:
            return "Short-term intensive learning (under 3 months). Maintain high focus and momentum."
    
    def _summarize_user_profile(self, user_data: Dict) -> Dict[str, Any]:
        """Create a summary of user profile relevant to learning roadmap"""
        return {
            'current_skills_count': len(user_data.get('skills', [])),
            'experience_years': user_data.get('experience_years', 0),
            'education_level': user_data.get('education', {}).get('current_level', 'Not specified'),
            'available_hours_per_week': user_data.get('available_hours_per_week', 10),
            'learning_goals': user_data.get('goals', {}),
            'preferred_learning_style': user_data.get('learning_preferences', {}).get('style', 'mixed')
        }
    
    def update_progress(self, roadmap: Dict, skill_progress: Dict[str, Dict]) -> Dict:
        """Update roadmap with user's learning progress"""
        updated_roadmap = roadmap.copy()
        
        # Update skill progress in timeline
        for phase in updated_roadmap.get('timeline', {}).get('phases', []):
            for skill_timeline in phase.get('skills_timeline', []):
                skill_name = skill_timeline['skill']
                if skill_name in skill_progress:
                    progress_data = skill_progress[skill_name]
                    skill_timeline['progress'] = {
                        'completion_percentage': progress_data.get('completion_percentage', 0),
                        'hours_spent': progress_data.get('hours_spent', 0),
                        'last_updated': progress_data.get('last_updated', datetime.now().isoformat()),
                        'status': progress_data.get('status', 'not_started'),  # not_started, in_progress, completed
                        'notes': progress_data.get('notes', '')
                    }
        
        # Update milestone progress
        for milestone in updated_roadmap.get('milestones', []):
            milestone_id = milestone['milestone_id']
            if milestone_id in skill_progress:
                milestone['progress'] = skill_progress[milestone_id]
        
        # Recalculate overall completion
        updated_roadmap['overall_progress'] = self._calculate_overall_progress(updated_roadmap)
        
        return updated_roadmap
    
    def _calculate_overall_progress(self, roadmap: Dict) -> Dict[str, Any]:
        """Calculate overall progress for the learning roadmap"""
        total_skills = 0
        completed_skills = 0
        in_progress_skills = 0
        
        for phase in roadmap.get('timeline', {}).get('phases', []):
            for skill_timeline in phase.get('skills_timeline', []):
                total_skills += 1
                progress = skill_timeline.get('progress', {})
                status = progress.get('status', 'not_started')
                
                if status == 'completed':
                    completed_skills += 1
                elif status == 'in_progress':
                    in_progress_skills += 1
        
        completion_percentage = (completed_skills / total_skills * 100) if total_skills > 0 else 0
        
        return {
            'total_skills': total_skills,
            'completed_skills': completed_skills,
            'in_progress_skills': in_progress_skills,
            'not_started_skills': total_skills - completed_skills - in_progress_skills,
            'completion_percentage': round(completion_percentage, 1),
            'estimated_time_remaining': self._estimate_remaining_time(roadmap, completion_percentage)
        }
    
    def _estimate_remaining_time(self, roadmap: Dict, completion_percentage: float) -> str:
        """Estimate remaining time based on current progress"""
        total_weeks = roadmap.get('timeline', {}).get('total_duration_weeks', 0)
        remaining_percentage = 100 - completion_percentage
        remaining_weeks = int(total_weeks * remaining_percentage / 100)
        
        if remaining_weeks < 4:
            return f"{remaining_weeks} weeks remaining"
        elif remaining_weeks < 52:
            return f"{remaining_weeks // 4} months remaining"
        else:
            return f"{remaining_weeks // 52} years remaining"
    
    def export_roadmap(self, roadmap: Dict, format: str = 'json') -> str:
        """Export roadmap in specified format"""
        if format.lower() == 'json':
            return json.dumps(roadmap, indent=2, default=str)
        elif format.lower() == 'summary':
            return self._create_text_summary(roadmap)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _create_text_summary(self, roadmap: Dict) -> str:
        """Create a text summary of the roadmap"""
        summary = f"""
LEARNING ROADMAP SUMMARY
========================

Career Target: {roadmap['career_title']}
Generated: {roadmap['generated_date']}

TIMELINE OVERVIEW:
- Total Duration: {roadmap['timeline']['total_duration_weeks']} weeks ({roadmap['timeline']['total_duration_months']:.1f} months)
- Start Date: {roadmap['timeline']['start_date']}
- Completion Date: {roadmap['timeline']['completion_date']}
- Required Commitment: {roadmap['commitment_required']['average_hours_per_week']} hours/week

LEARNING PHASES:
"""
        
        for phase in roadmap['learning_phases']:
            summary += f"""
Phase {phase['phase_number']}: {phase['phase_name']}
- Priority: {phase['priority']}
- Skills: {', '.join(phase['skills_to_learn'])}
- Duration: {phase['estimated_duration_weeks']} weeks
"""
        
        summary += f"""
MAJOR MILESTONES:
"""
        for milestone in roadmap['milestones']:
            summary += f"- {milestone['title']} (by {milestone['target_date']})\n"
        
        return summary
