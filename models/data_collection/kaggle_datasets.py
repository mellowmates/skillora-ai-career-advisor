import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import json
import logging

class KaggleDatasetsDownloader:
    """Download and preprocess Kaggle career-related datasets"""
    
    def __init__(self, download_path='data/kaggle_raw/', processed_path='data/'):
        self.download_path = download_path
        self.processed_path = processed_path
        os.makedirs(self.download_path, exist_ok=True)
        os.makedirs(self.processed_path, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def download_career_datasets(self) -> List[str]:
        """Download relevant career datasets from Kaggle"""
        # For Phase 1: Mock implementation
        datasets = [
            'indeed-job-postings-2021',
            'data-science-job-salaries',
            'job-skills-dataset',
            'indian-startup-ecosystem'
        ]
        
        downloaded_files = []
        for dataset in datasets:
            self.logger.info(f"Downloading {dataset} from Kaggle...")
            # In real implementation: kaggle.api.dataset_download_files(dataset)
            self.logger.info(f"Mock: Downloaded {dataset}")
            downloaded_files.append(f"{self.download_path}{dataset}.csv")
            
        return downloaded_files
    
    def preprocess_job_postings(self, file_path: str) -> pd.DataFrame:
        """Preprocess job postings data"""
        # For Phase 1: Create sample data structure
        sample_data = {
            'job_title': ['Software Engineer', 'Data Scientist', 'Product Manager', 'Marketing Specialist'],
            'company': ['TCS', 'Infosys', 'Flipkart', 'Zomato'],
            'location': ['Bangalore', 'Mumbai', 'Delhi', 'Pune'],
            'salary_min': [500000, 800000, 1200000, 400000],
            'salary_max': [800000, 1500000, 2000000, 700000],
            'experience_required': [2, 3, 5, 1],
            'skills': ['Python,Java,SQL', 'Python,ML,Statistics', 'Analytics,Strategy,Leadership', 'Digital Marketing,SEO,Analytics']
        }
        
        df = pd.DataFrame(sample_data)
        
        # Basic preprocessing
        df['salary_avg'] = (df['salary_min'] + df['salary_max']) / 2
        df['skills_list'] = df['skills'].apply(lambda x: x.split(','))
        df = df.dropna()
        
        return df
    
    def preprocess_salary_data(self, file_path: str) -> pd.DataFrame:
        """Preprocess salary dataset"""
        sample_salary_data = {
            'job_title': ['Software Engineer', 'Data Scientist', 'Product Manager', 'Marketing Manager'],
            'experience_level': ['Mid', 'Senior', 'Senior', 'Mid'],
            'salary': [1200000, 1800000, 2500000, 1000000],
            'company_location': ['Bangalore', 'Mumbai', 'Delhi', 'Chennai'],
            'company_size': ['Large', 'Medium', 'Large', 'Startup']
        }
        
        df = pd.DataFrame(sample_salary_data)
        return df
    
    def combine_datasets(self, datasets: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple datasets into unified format"""
        combined_data = []
        
        for df in datasets:
            # Standardize column names and format
            standardized = self._standardize_dataframe(df)
            combined_data.append(standardized)
        
        final_df = pd.concat(combined_data, ignore_index=True)
        return final_df
    
    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize dataframe format"""
        # Map different column names to standard format
        column_mapping = {
            'job_title': 'title',
            'position': 'title', 
            'role': 'title',
            'company_name': 'company',
            'organization': 'company',
            'salary_avg': 'salary',
            'compensation': 'salary',
            'pay': 'salary'
        }
        
        # Rename columns
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = 'processed_kaggle_data.csv'):
        """Save processed data to CSV"""
        file_path = os.path.join(self.processed_path, filename)
        df.to_csv(file_path, index=False)
        self.logger.info(f"Processed data saved to {file_path}")
        
        # Also save JSON version for easier loading
        json_path = file_path.replace('.csv', '.json')
        df.to_json(json_path, orient='records', indent=2)
        
    def generate_sample_data(self) -> pd.DataFrame:
        """Generate comprehensive sample data for Phase 1"""
        np.random.seed(42)
        
        careers = ['software_engineer', 'data_scientist', 'product_manager', 
                  'digital_marketing_specialist', 'chartered_accountant']
        locations = ['bangalore', 'mumbai', 'delhi', 'pune', 'hyderabad', 'chennai']
        education_levels = ['B.Tech', 'B.E.', 'BCA', 'MBA', 'M.Tech', 'B.Com', 'CA']
        
        sample_size = 500
        data = []
        
        for i in range(sample_size):
            career = np.random.choice(careers)
            location = np.random.choice(locations)
            education = np.random.choice(education_levels)
            experience = np.random.randint(0, 15)
            
            # Generate salary based on career and experience
            base_salary = {
                'software_engineer': 600000,
                'data_scientist': 800000, 
                'product_manager': 1000000,
                'digital_marketing_specialist': 500000,
                'chartered_accountant': 700000
            }.get(career, 500000)
            
            salary = base_salary + (experience * 80000) + np.random.randint(-100000, 200000)
            salary = max(250000, salary)  # Minimum salary
            
            # Generate skills based on career
            skill_sets = {
                'software_engineer': ['Python', 'Java', 'JavaScript', 'SQL', 'Git', 'AWS'],
                'data_scientist': ['Python', 'R', 'Statistics', 'Machine Learning', 'SQL', 'Tableau'],
                'product_manager': ['Analytics', 'Strategy', 'Leadership', 'Communication', 'Roadmapping'],
                'digital_marketing_specialist': ['SEO', 'Google Ads', 'Social Media', 'Analytics', 'Content Marketing'],
                'chartered_accountant': ['Accounting', 'Taxation', 'Audit', 'Financial Analysis', 'Compliance']
            }
            
            skills = np.random.choice(skill_sets[career], size=np.random.randint(3, 6), replace=False).tolist()
            
            data.append({
                'id': i + 1,
                'name': f'Profile {i + 1}',
                'description': f'{career.replace("_", " ").title()} with {experience} years experience',
                'skills': ','.join(skills),
                'career_path': career,
                'education_level': education,
                'experience_years': experience,
                'salary': salary,
                'location': location,
                'industry': 'technology' if career in ['software_engineer', 'data_scientist'] else 'business'
            })
        
        return pd.DataFrame(data)

if __name__ == "__main__":
    downloader = KaggleDatasetsDownloader()
    sample_df = downloader.generate_sample_data()
    downloader.save_processed_data(sample_df)
    print(f"Generated {len(sample_df)} sample career profiles")
