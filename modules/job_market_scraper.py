"""
Job Market Scraper Module
Scrapes and analyzes job market data from various sources
"""

import requests
from bs4 import BeautifulSoup
import json
import time
from typing import Dict, List, Any, Optional
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from datetime import datetime, timedelta

class JobMarketScraper:
    def __init__(self):
        """Initialize the job market scraper"""
        self.job_sites = {
            'indeed': 'https://www.indeed.com/jobs',
            'linkedin': 'https://www.linkedin.com/jobs/search',
            'glassdoor': 'https://www.glassdoor.com/Job/jobs.htm',
            'ziprecruiter': 'https://www.ziprecruiter.com/jobs-search'
        }
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        self.job_data = []
    
    def scrape_job_market(self, keywords: List[str], location: str = "", 
                         max_pages: int = 5) -> Dict[str, Any]:
        """
        Scrape job market data for given keywords and location
        
        Args:
            keywords: List of job keywords to search for
            location: Location to search in
            max_pages: Maximum number of pages to scrape
            
        Returns:
            Dictionary with scraped job data and analysis
        """
        try:
            all_jobs = []
            
            for keyword in keywords:
                print(f"Scraping jobs for keyword: {keyword}")
                
                # Scrape from different job sites
                indeed_jobs = self._scrape_indeed(keyword, location, max_pages)
                linkedin_jobs = self._scrape_linkedin(keyword, location, max_pages)
                
                all_jobs.extend(indeed_jobs)
                all_jobs.extend(linkedin_jobs)
                
                # Add delay between requests
                time.sleep(2)
            
            # Analyze the collected data
            analysis = self._analyze_job_market(all_jobs)
            
            return {
                'success': True,
                'total_jobs': len(all_jobs),
                'jobs': all_jobs,
                'analysis': analysis,
                'scraped_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'jobs': [],
                'analysis': None
            }
    
    def _scrape_indeed(self, keyword: str, location: str, max_pages: int) -> List[Dict[str, Any]]:
        """Scrape job data from Indeed"""
        jobs = []
        
        try:
            for page in range(max_pages):
                start = page * 10
                url = f"{self.job_sites['indeed']}?q={keyword}&l={location}&start={start}"
                
                response = requests.get(url, headers=self.headers)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                job_cards = soup.find_all('div', class_='job_seen_beacon')
                
                for card in job_cards:
                    job_data = self._extract_indeed_job_data(card)
                    if job_data:
                        jobs.append(job_data)
                
                time.sleep(1)  # Rate limiting
                
        except Exception as e:
            print(f"Error scraping Indeed: {str(e)}")
        
        return jobs
    
    def _extract_indeed_job_data(self, card) -> Optional[Dict[str, Any]]:
        """Extract job data from Indeed job card"""
        try:
            title_elem = card.find('h2', class_='jobTitle')
            company_elem = card.find('span', class_='companyName')
            location_elem = card.find('div', class_='companyLocation')
            salary_elem = card.find('div', class_='salary-snippet')
            
            if not title_elem:
                return None
            
            title = title_elem.get_text(strip=True)
            company = company_elem.get_text(strip=True) if company_elem else "Unknown"
            location = location_elem.get_text(strip=True) if location_elem else "Unknown"
            salary = salary_elem.get_text(strip=True) if salary_elem else "Not specified"
            
            # Extract job link
            link_elem = title_elem.find('a')
            job_url = f"https://www.indeed.com{link_elem['href']}" if link_elem else ""
            
            return {
                'title': title,
                'company': company,
                'location': location,
                'salary': salary,
                'url': job_url,
                'source': 'indeed',
                'scraped_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error extracting Indeed job data: {str(e)}")
            return None
    
    def _scrape_linkedin(self, keyword: str, location: str, max_pages: int) -> List[Dict[str, Any]]:
        """Scrape job data from LinkedIn (simplified version)"""
        jobs = []
        
        try:
            # Note: LinkedIn has strict anti-scraping measures
            # This is a simplified version that would need proper handling
            # in a production environment
            
            url = f"{self.job_sites['linkedin']}?keywords={keyword}&location={location}"
            
            # For demo purposes, return mock data
            # In production, you'd need proper LinkedIn API access or advanced scraping
            mock_jobs = [
                {
                    'title': f'{keyword} Developer',
                    'company': 'Tech Company Inc.',
                    'location': location or 'Remote',
                    'salary': '$80,000 - $120,000',
                    'url': 'https://linkedin.com/jobs/view/123456',
                    'source': 'linkedin',
                    'scraped_at': datetime.now().isoformat()
                }
            ]
            
            jobs.extend(mock_jobs)
            
        except Exception as e:
            print(f"Error scraping LinkedIn: {str(e)}")
        
        return jobs
    
    def _analyze_job_market(self, jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the scraped job market data"""
        if not jobs:
            return {
                'total_jobs': 0,
                'average_salary': 0,
                'top_companies': [],
                'top_locations': [],
                'skill_demand': {},
                'trends': {}
            }
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(jobs)
        
        # Calculate average salary (simplified)
        salaries = []
        for job in jobs:
            salary_text = job.get('salary', '')
            if salary_text and salary_text != 'Not specified':
                # Extract numeric values from salary text
                import re
                numbers = re.findall(r'\d+', salary_text.replace(',', ''))
                if numbers:
                    avg_salary = sum(map(int, numbers)) / len(numbers)
                    salaries.append(avg_salary)
        
        average_salary = sum(salaries) / len(salaries) if salaries else 0
        
        # Top companies
        top_companies = df['company'].value_counts().head(10).to_dict()
        
        # Top locations
        top_locations = df['location'].value_counts().head(10).to_dict()
        
        # Skill demand analysis
        skill_demand = self._analyze_skill_demand(jobs)
        
        # Market trends
        trends = self._analyze_trends(jobs)
        
        return {
            'total_jobs': len(jobs),
            'average_salary': round(average_salary, 2),
            'top_companies': top_companies,
            'top_locations': top_locations,
            'skill_demand': skill_demand,
            'trends': trends,
            'salary_range': self._calculate_salary_range(salaries)
        }
    
    def _analyze_skill_demand(self, jobs: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze skill demand from job titles and descriptions"""
        skill_keywords = {
            'python': ['python', 'django', 'flask'],
            'javascript': ['javascript', 'node.js', 'react', 'angular'],
            'java': ['java', 'spring', 'hibernate'],
            'data_analysis': ['data', 'analytics', 'sql', 'tableau'],
            'machine_learning': ['machine learning', 'ai', 'tensorflow', 'pytorch'],
            'cloud': ['aws', 'azure', 'gcp', 'cloud'],
            'devops': ['devops', 'docker', 'kubernetes', 'ci/cd'],
            'mobile': ['ios', 'android', 'react native', 'flutter'],
            'web_development': ['html', 'css', 'frontend', 'backend'],
            'project_management': ['project management', 'agile', 'scrum']
        }
        
        skill_counts = {skill: 0 for skill in skill_keywords.keys()}
        
        for job in jobs:
            title = job.get('title', '').lower()
            for skill, keywords in skill_keywords.items():
                if any(keyword in title for keyword in keywords):
                    skill_counts[skill] += 1
        
        return skill_counts
    
    def _analyze_trends(self, jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze job market trends"""
        # This would typically involve time-series analysis
        # For now, return basic trend information
        
        remote_jobs = sum(1 for job in jobs if 'remote' in job.get('location', '').lower())
        total_jobs = len(jobs)
        
        return {
            'remote_work_percentage': round((remote_jobs / total_jobs) * 100, 2) if total_jobs > 0 else 0,
            'growth_indicators': {
                'high_demand_skills': ['python', 'data_analysis', 'cloud'],
                'emerging_skills': ['machine_learning', 'devops', 'mobile']
            }
        }
    
    def _calculate_salary_range(self, salaries: List[float]) -> Dict[str, float]:
        """Calculate salary range statistics"""
        if not salaries:
            return {'min': 0, 'max': 0, 'median': 0}
        
        salaries.sort()
        return {
            'min': min(salaries),
            'max': max(salaries),
            'median': salaries[len(salaries) // 2]
        }
    
    def get_job_recommendations(self, user_skills: Dict[str, str], 
                              location: str = "") -> Dict[str, Any]:
        """
        Get job recommendations based on user skills
        
        Args:
            user_skills: Dictionary of user skills and levels
            location: Preferred job location
            
        Returns:
            Dictionary with job recommendations
        """
        try:
            # Extract top skills
            top_skills = [skill for skill, level in user_skills.items() 
                         if level in ['advanced', 'expert']]
            
            if not top_skills:
                top_skills = list(user_skills.keys())[:3]  # Take first 3 skills
            
            # Scrape jobs for these skills
            job_data = self.scrape_job_market(top_skills, location, max_pages=3)
            
            if not job_data['success']:
                return job_data
            
            # Filter and rank jobs based on skill match
            recommendations = self._rank_jobs_by_skills(job_data['jobs'], user_skills)
            
            return {
                'success': True,
                'recommendations': recommendations,
                'market_analysis': job_data['analysis']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'recommendations': []
            }
    
    def _rank_jobs_by_skills(self, jobs: List[Dict[str, Any]], 
                           user_skills: Dict[str, str]) -> List[Dict[str, Any]]:
        """Rank jobs based on skill match with user profile"""
        ranked_jobs = []
        
        for job in jobs:
            title = job.get('title', '').lower()
            match_score = 0
            
            # Calculate skill match score
            for skill, level in user_skills.items():
                if skill.lower() in title:
                    # Higher score for higher skill levels
                    level_scores = {'beginner': 1, 'intermediate': 2, 'advanced': 3, 'expert': 4}
                    match_score += level_scores.get(level, 1)
            
            if match_score > 0:
                job['match_score'] = match_score
                ranked_jobs.append(job)
        
        # Sort by match score
        ranked_jobs.sort(key=lambda x: x['match_score'], reverse=True)
        
        return ranked_jobs[:20]  # Return top 20 matches
    
    def save_job_data(self, filename: str = "job_market_data.json") -> bool:
        """Save scraped job data to file"""
        try:
            with open(f"data/{filename}", 'w') as f:
                json.dump(self.job_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving job data: {str(e)}")
            return False
    
    def load_job_data(self, filename: str = "job_market_data.json") -> bool:
        """Load job data from file"""
        try:
            with open(f"data/{filename}", 'r') as f:
                self.job_data = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading job data: {str(e)}")
            return False
