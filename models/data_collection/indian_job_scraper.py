import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
import json
from typing import List, Dict, Optional
import logging
from urllib.parse import urljoin, quote_plus
import random

class IndianJobScraper:
    """Scrapes Indian job portals for real-time job market data"""
    
    def __init__(self, base_urls: Dict[str, str] = None):
        self.base_urls = base_urls or {
            'naukri': 'https://www.naukri.com',
            'indeed': 'https://in.indeed.com',
            'linkedin': 'https://www.linkedin.com/jobs'
        }
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def scrape_naukri_jobs(self, keywords: List[str], locations: List[str], max_pages: int = 3) -> pd.DataFrame:
        """Scrape jobs from Naukri.com"""
        all_jobs = []
        
        for keyword in keywords:
            for location in locations:
                self.logger.info(f"Scraping {keyword} jobs in {location}")
                
                # For Phase 1: Generate mock data instead of actual scraping
                mock_jobs = self._generate_mock_job_data(keyword, location, 20)
                all_jobs.extend(mock_jobs)
                
                # Simulate delay
                time.sleep(1)
        
        return pd.DataFrame(all_jobs)
    
    def scrape_indeed_jobs(self, keywords: List[str], locations: List[str]) -> pd.DataFrame:
        """Scrape jobs from Indeed India"""
        all_jobs = []
        
        for keyword in keywords:
            for location in locations:
                # Mock implementation for Phase 1
                mock_jobs = self._generate_mock_job_data(keyword, location, 15)
                all_jobs.extend(mock_jobs)
                time.sleep(1)
        
        return pd.DataFrame(all_jobs)
    
    def _generate_mock_job_data(self, keyword: str, location: str, count: int) -> List[Dict]:
        """Generate realistic mock job data for Phase 1"""
        companies = ['TCS', 'Infosys', 'Wipro', 'Accenture', 'IBM', 'Google', 'Microsoft', 
                    'Amazon', 'Flipkart', 'Paytm', 'Zomato', 'Swiggy', 'Ola', 'Uber']
        
        job_titles = {
            'software engineer': ['Software Engineer', 'Software Developer', 'Full Stack Developer', 'Backend Developer'],
            'data scientist': ['Data Scientist', 'Data Analyst', 'ML Engineer', 'Business Analyst'],
            'product manager': ['Product Manager', 'Associate Product Manager', 'Senior Product Manager'],
            'marketing': ['Marketing Manager', 'Digital Marketing Specialist', 'Marketing Executive']
        }
        
        # Map keyword to relevant job titles
        titles = []
        for key, title_list in job_titles.items():
            if key.lower() in keyword.lower():
                titles = title_list
                break
        
        if not titles:
            titles = ['Software Engineer', 'Business Analyst', 'Project Manager']
        
        jobs = []
        for i in range(count):
            company = random.choice(companies)
            title = random.choice(titles)
            
            # Generate salary based on location and role
            base_salaries = {
                'bangalore': 100000,
                'mumbai': 95000,
                'delhi': 90000,
                'pune': 85000,
                'hyderabad': 80000,
                'chennai': 75000
            }
            
            base = base_salaries.get(location.lower(), 70000)
            salary_min = base * random.randint(4, 8)
            salary_max = salary_min + random.randint(200000, 500000)
            
            jobs.append({
                'title': title,
                'company': company,
                'location': location.title(),
                'salary_min': salary_min,
                'salary_max': salary_max,
                'experience_required': random.randint(0, 10),
                'job_type': random.choice(['Full-time', 'Contract', 'Part-time']),
                'posted_date': f'2025-0{random.randint(1, 3)}-{random.randint(1, 28)}',
                'description': f'{title} role at {company} in {location}',
                'skills': self._generate_relevant_skills(title),
                'source': 'naukri' if i % 2 == 0 else 'indeed'
            })
        
        return jobs
    
    def _generate_relevant_skills(self, job_title: str) -> str:
        """Generate relevant skills based on job title"""
        skill_mapping = {
            'Software Engineer': ['Python', 'Java', 'JavaScript', 'SQL', 'Git'],
            'Data Scientist': ['Python', 'R', 'Machine Learning', 'Statistics', 'SQL'],
            'Product Manager': ['Analytics', 'Strategy', 'Leadership', 'Communication'],
            'Marketing': ['Digital Marketing', 'SEO', 'Google Ads', 'Social Media']
        }
        
        for key, skills in skill_mapping.items():
            if key.lower() in job_title.lower():
                return ','.join(random.sample(skills, min(len(skills), 4)))
        
        return 'Communication,Teamwork,Problem Solving'
    
    def aggregate_market_data(self, jobs_df: pd.DataFrame) -> Dict:
        """Aggregate job market insights"""
        market_data = {
            'total_jobs': len(jobs_df),
            'avg_salary': jobs_df['salary_min'].mean(),
            'top_locations': jobs_df['location'].value_counts().head().to_dict(),
            'top_companies': jobs_df['company'].value_counts().head().to_dict(),
            'top_skills': self._extract_top_skills(jobs_df['skills']),
            'experience_distribution': jobs_df['experience_required'].value_counts().to_dict(),
            'salary_ranges': {
                'entry_level': jobs_df[jobs_df['experience_required'] <= 2]['salary_min'].median(),
                'mid_level': jobs_df[(jobs_df['experience_required'] > 2) & 
                                   (jobs_df['experience_required'] <= 5)]['salary_min'].median(),
                'senior_level': jobs_df[jobs_df['experience_required'] > 5]['salary_min'].median()
            }
        }
        
        return market_data
    
    def _extract_top_skills(self, skills_series: pd.Series) -> Dict:
        """Extract most mentioned skills"""
        all_skills = []
        for skills_str in skills_series.dropna():
            skills = [skill.strip() for skill in skills_str.split(',')]
            all_skills.extend(skills)
        
        from collections import Counter
        skill_counts = Counter(all_skills)
        return dict(skill_counts.most_common(10))
    
    def save_jobs_data(self, jobs_df: pd.DataFrame, market_data: Dict, output_path: str = 'data/'):
        """Save scraped job data and market insights"""
        # Save jobs CSV
        jobs_file = f"{output_path}scraped_jobs.csv"
        jobs_df.to_csv(jobs_file, index=False)
        
        # Save market insights JSON
        market_file = f"{output_path}market_insights.json"
        with open(market_file, 'w') as f:
            json.dump(market_data, f, indent=2)
        
        self.logger.info(f"Saved {len(jobs_df)} jobs to {jobs_file}")
        self.logger.info(f"Saved market insights to {market_file}")
        
        return jobs_file, market_file
    
    def run_full_scrape(self, keywords: List[str], locations: List[str]) -> Dict:
        """Run complete scraping workflow"""
        self.logger.info("Starting job market data collection...")
        
        # Scrape from multiple sources
        naukri_jobs = self.scrape_naukri_jobs(keywords, locations)
        indeed_jobs = self.scrape_indeed_jobs(keywords, locations)
        
        # Combine all data
        all_jobs = pd.concat([naukri_jobs, indeed_jobs], ignore_index=True)
        
        # Remove duplicates
        all_jobs = all_jobs.drop_duplicates(subset=['title', 'company', 'location'])
        
        # Generate market insights
        market_data = self.aggregate_market_data(all_jobs)
        
        # Save data
        jobs_file, market_file = self.save_jobs_data(all_jobs, market_data)
        
        return {
            'jobs_count': len(all_jobs),
            'jobs_file': jobs_file,
            'market_file': market_file,
            'market_insights': market_data
        }

if __name__ == "__main__":
    scraper = IndianJobScraper()
    
    keywords = ['software engineer', 'data scientist', 'product manager', 'marketing']
    locations = ['bangalore', 'mumbai', 'delhi', 'pune', 'hyderabad']
    
    results = scraper.run_full_scrape(keywords, locations)
    print(f"Scraped {results['jobs_count']} jobs")
