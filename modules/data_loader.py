"""
Data Loader Module for Skillora AI Career Advisor
Provides centralized data management with caching and error handling
"""

import json
import os
import csv
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

class DataLoader:
    """
    Centralized data loader for all Skillora AI data files
    Implements caching and graceful error handling for missing files
    """
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        self.cache = {}
        self._ensure_data_directory()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"DataLoader initialized with data directory: {data_dir}")
    
    def _ensure_data_directory(self):
        """Ensure data directory exists"""
        if not os.path.exists(self.data_dir):
            self.logger.warning(f"Data directory {self.data_dir} does not exist. Creating...")
            os.makedirs(self.data_dir, exist_ok=True)
    
    def _load_json(self, filename: str) -> Dict[str, Any]:
        """
        Load JSON file with caching and error handling
        
        Args:
            filename: Name of JSON file to load
            
        Returns:
            Dict containing loaded data or empty dict if file not found/invalid
        """
        # Check cache first
        if filename in self.cache:
            self.logger.debug(f"Loading {filename} from cache")
            return self.cache[filename]
        
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            self.logger.warning(f"Data file {filepath} not found. Returning empty dict.")
            return {}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.cache[filename] = data
                self.logger.info(f"Successfully loaded {filename} ({len(data) if isinstance(data, (dict, list)) else 'N/A'} items)")
                return data
                
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error in {filename}: {str(e)}")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading {filename}: {str(e)}")
            return {}
    
    def _load_csv(self, filename: str) -> List[Dict[str, Any]]:
        """
        Load CSV file and return as list of dictionaries
        
        Args:
            filename: Name of CSV file to load
            
        Returns:
            List of dictionaries representing CSV rows
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            self.logger.warning(f"CSV file {filepath} not found. Returning empty list.")
            return []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                data = list(reader)
                self.logger.info(f"Successfully loaded {filename} ({len(data)} rows)")
                return data
                
        except Exception as e:
            self.logger.error(f"Error reading CSV {filename}: {str(e)}")
            return []
    
    def get_career_profiles(self) -> Dict[str, Any]:
        """
        Load detailed career information with skills requirements
        
        Returns:
            Dict with career IDs as keys and career data as values
        """
        return self._load_json('career_profiles.json')
    
    def get_skills_mapping(self) -> Dict[str, Any]:
        """
        Load skills-to-career mappings with priorities and learning resources
        
        Returns:
            Dict containing skill categories and mappings
        """
        return self._load_json('skills_mapping.json')
    
    def get_job_market_data(self) -> Dict[str, Any]:
        """
        Load Indian job market statistics and trends
        
        Returns:
            Dict containing market data, growth rates, and sector information
        """
        return self._load_json('job_market_data.json')
    
    def get_personality_traits(self) -> Dict[str, Any]:
        """
        Load personality traits data for Big Five model
        
        Returns:
            Dict containing personality-career correlations and assessment data
        """
        return self._load_json('personality_traits.json')
    
    def get_education_paths(self) -> Dict[str, Any]:
        """
        Load education pathways and requirements
        
        Returns:
            Dict containing formal education, professional courses, and pathways
        """
        return self._load_json('education_paths.json')
    
    def get_salary_data(self) -> Dict[str, Any]:
        """
        Load India-specific salary ranges and compensation data
        
        Returns:
            Dict containing salary ranges by role, location, and experience
        """
        return self._load_json('salary_data.json')
    
    def get_learning_resources(self) -> Dict[str, Any]:
        """
        Load learning resources including courses, books, and certifications
        
        Returns:
            Dict containing categorized learning resources
        """
        return self._load_json('learning_resources.json')
    
    def get_chatbot_knowledge(self) -> Dict[str, Any]:
        """
        Load chatbot knowledge base and response templates
        
        Returns:
            Dict containing intents, responses, and conversation flows
        """
        return self._load_json('chatbot_knowledge.json')
    
    def get_india_locations(self) -> Dict[str, Any]:
        """
        Load Indian cities and states with market data
        
        Returns:
            Dict containing location data with job market information
        """
        return self._load_json('india_locations.json')
    
    def get_kaggle_dataset(self) -> List[Dict[str, Any]]:
        """
        Load raw Kaggle dataset for ML training
        
        Returns:
            List of dictionaries representing career profiles from Kaggle data
        """
        return self._load_csv('processed/preprocessed_data.csv')
    
    def get_career_by_id(self, career_id: str) -> Optional[Dict[str, Any]]:
        """
        Get specific career profile by ID
        
        Args:
            career_id: Career identifier
            
        Returns:
            Dict containing career data or None if not found
        """
        careers = self.get_career_profiles()
        return careers.get(career_id)
    
    def get_skills_by_category(self, category: str) -> Dict[str, Any]:
        """
        Get skills by specific category
        
        Args:
            category: Skill category (e.g., 'technical_skills', 'soft_skills')
            
        Returns:
            Dict containing skills in the specified category
        """
        skills_mapping = self.get_skills_mapping()
        return skills_mapping.get(category, {})
    
    def get_location_data(self, location: str) -> Optional[Dict[str, Any]]:
        """
        Get specific location data
        
        Args:
            location: Location name (city or state)
            
        Returns:
            Dict containing location data or None if not found
        """
        locations = self.get_india_locations()
        
        # Check major cities first
        major_cities = locations.get('major_cities', {})
        if location.lower() in major_cities:
            return major_cities[location.lower()]
        
        # Check tier-2 cities
        tier2_cities = locations.get('tier2_cities', {})
        if location.lower() in tier2_cities:
            return tier2_cities[location.lower()]
        
        return None
    
    def get_salary_range(self, role: str, experience_level: str = 'mid') -> Optional[Dict[str, Any]]:
        """
        Get salary range for specific role and experience level
        
        Args:
            role: Job role identifier
            experience_level: Experience level (entry, mid, senior, lead)
            
        Returns:
            Dict containing salary range or None if not found
        """
        salary_data = self.get_salary_data()
        role_based_salary = salary_data.get('role_based_salary', {})
        
        if role in role_based_salary:
            return role_based_salary[role].get(experience_level)
        
        return None
    
    def search_careers_by_skills(self, skills: List[str]) -> List[Dict[str, Any]]:
        """
        Search careers that match given skills
        
        Args:
            skills: List of skill names
            
        Returns:
            List of matching career profiles with match scores
        """
        careers = self.get_career_profiles()
        matching_careers = []
        
        skills_lower = [skill.lower() for skill in skills]
        
        for career_id, career_data in careers.items():
            required_skills = career_data.get('required_skills', {})
            all_required = []
            
            # Collect all required skills
            for category in ['critical', 'important', 'preferred']:
                all_required.extend([s.lower() for s in required_skills.get(category, [])])
            
            # Calculate match score
            matches = sum(1 for skill in skills_lower if any(skill in req_skill or req_skill in skill for req_skill in all_required))
            match_score = matches / len(all_required) if all_required else 0
            
            if match_score > 0:
                matching_careers.append({
                    'career_id': career_id,
                    'career_title': career_data.get('title', career_id.replace('_', ' ').title()),
                    'match_score': match_score,
                    'matching_skills_count': matches
                })
        
        # Sort by match score
        matching_careers.sort(key=lambda x: x['match_score'], reverse=True)
        return matching_careers
    
    def get_trending_skills(self) -> List[Dict[str, Any]]:
        """
        Get trending skills based on market demand data
        
        Returns:
            List of trending skills with demand information
        """
        market_data = self.get_job_market_data()
        skill_trends = market_data.get('skill_demand_trends', {})
        
        trending_skills = []
        
        # Get most in-demand skills
        most_in_demand = skill_trends.get('most_in_demand', [])
        trending_skills.extend(most_in_demand)
        
        # Get emerging skills
        emerging_skills = skill_trends.get('emerging_skills', [])
        trending_skills.extend(emerging_skills)
        
        return trending_skills
    
    def get_education_requirements(self, career_id: str) -> Optional[Dict[str, Any]]:
        """
        Get education requirements for specific career
        
        Args:
            career_id: Career identifier
            
        Returns:
            Dict containing education requirements or None if not found
        """
        career = self.get_career_by_id(career_id)
        if career:
            return career.get('education', {})
        return None
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """
        Validate integrity of all data files
        
        Returns:
            Dict containing validation results
        """
        validation_results = {
            'files_checked': 0,
            'files_valid': 0,
            'files_missing': [],
            'files_invalid': [],
            'total_records': 0
        }
        
        data_files = [
            'career_profiles.json',
            'skills_mapping.json', 
            'job_market_data.json',
            'personality_traits.json',
            'education_paths.json',
            'salary_data.json',
            'learning_resources.json',
            'chatbot_knowledge.json',
            'india_locations.json'
        ]
        
        for filename in data_files:
            validation_results['files_checked'] += 1
            filepath = os.path.join(self.data_dir, filename)
            
            if not os.path.exists(filepath):
                validation_results['files_missing'].append(filename)
                continue
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    validation_results['files_valid'] += 1
                    
                    if isinstance(data, (dict, list)):
                        validation_results['total_records'] += len(data)
                        
            except Exception as e:
                validation_results['files_invalid'].append({
                    'filename': filename,
                    'error': str(e)
                })
        
        # Check CSV file separately
        csv_file = 'processed/preprocessed_data.csv'
        csv_path = os.path.join(self.data_dir, csv_file)
        validation_results['files_checked'] += 1
        
        if os.path.exists(csv_path):
            try:
                csv_data = self._load_csv(csv_file)
                validation_results['files_valid'] += 1
                validation_results['total_records'] += len(csv_data)
            except Exception as e:
                validation_results['files_invalid'].append({
                    'filename': csv_file,
                    'error': str(e)
                })
        else:
            validation_results['files_missing'].append(csv_file)
        
        self.logger.info(f"Data validation completed: {validation_results['files_valid']}/{validation_results['files_checked']} files valid")
        return validation_results
    
    def clear_cache(self):
        """Clear the data cache to force reload on next access"""
        self.cache.clear()
        self.logger.info("Data cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached data
        
        Returns:
            Dict containing cache statistics
        """
        return {
            'cached_files': list(self.cache.keys()),
            'cache_size': len(self.cache),
            'memory_usage': sum(len(str(data)) for data in self.cache.values())
        }
    
    def export_data_summary(self) -> Dict[str, Any]:
        """
        Export summary of all loaded data
        
        Returns:
            Dict containing data summary statistics
        """
        summary = {
            'careers_count': len(self.get_career_profiles()),
            'skills_categories': len(self.get_skills_mapping()),
            'locations_count': len(self.get_india_locations().get('major_cities', {})),
            'learning_resources_count': len(self.get_learning_resources()),
            'kaggle_records_count': len(self.get_kaggle_dataset()),
            'data_integrity': self.validate_data_integrity()
        }
        
        return summary

# Usage examples and testing
def test_data_loader():
    """Test function to verify DataLoader functionality"""
    loader = DataLoader()
    
    # Test basic loading
    careers = loader.get_career_profiles()
    skills = loader.get_skills_mapping()
    
    print(f"Loaded {len(careers)} career profiles")
    print(f"Loaded {len(skills)} skill categories")
    
    # Test specific queries
    software_eng = loader.get_career_by_id('software_engineer')
    if software_eng:
        print(f"Software Engineer title: {software_eng.get('title')}")
    
    # Test skill search
    matching_careers = loader.search_careers_by_skills(['Python', 'Machine Learning'])
    print(f"Found {len(matching_careers)} careers matching Python + ML skills")
    
    # Test validation
    validation = loader.validate_data_integrity()
    print(f"Data validation: {validation['files_valid']}/{validation['files_checked']} files valid")
    
    return loader

if __name__ == "__main__":
    # Run tests if executed directly
    test_data_loader()
