import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import logging

class DataPreprocessor:
    """Prepares and cleans data for ML training"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.mlb = MultiLabelBinarizer()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_and_combine_data(self, data_sources: Dict[str, str]) -> pd.DataFrame:
        """Load and combine data from multiple sources"""
        dataframes = []
        
        for source_name, file_path in data_sources.items():
            self.logger.info(f"Loading data from {source_name}: {file_path}")
            
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                self.logger.error(f"Unsupported file format: {file_path}")
                continue
                
            df['data_source'] = source_name
            dataframes.append(df)
        
        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            self.logger.info(f"Combined dataset shape: {combined_df.shape}")
            return combined_df
        else:
            self.logger.error("No valid data sources found")
            return pd.DataFrame()
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform basic data cleaning"""
        self.logger.info("Starting data cleaning...")
        
        initial_rows = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        self.logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Clean text columns
        text_columns = ['name', 'description', 'skills', 'career_path']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(self._clean_text)
        
        # Standardize categorical values
        df = self._standardize_categorical_values(df)
        
        # Validate data types and ranges
        df = self._validate_data(df)
        
        self.logger.info(f"Cleaned dataset shape: {df.shape}")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on column type and importance"""
        
        # Strategy for different types of columns
        strategies = {
            'salary': lambda x: x.fillna(x.median()),
            'experience_years': lambda x: x.fillna(0),
            'location': lambda x: x.fillna('india'),
            'industry': lambda x: x.fillna('general'),
            'education_level': lambda x: x.fillna('graduate'),
            'skills': lambda x: x.fillna('general_skills')
        }
        
        for column, strategy in strategies.items():
            if column in df.columns:
                df[column] = strategy(df[column])
        
        # Forward fill for remaining columns
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Drop rows with critical missing data
        critical_columns = ['career_path', 'skills']
        for col in critical_columns:
            if col in df.columns:
                df = df.dropna(subset=[col])
        
        return df
    
    def _clean_text(self, text: str) -> str:
        """Clean and standardize text data"""
        if pd.isna(text) or text == 'nan':
            return ''
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep relevant ones
        text = re.sub(r'[^\w\s\,\.\-\+\#]', '', text)
        
        return text
    
    def _standardize_categorical_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize categorical values for consistency"""
        
        # Standardize career paths
        if 'career_path' in df.columns:
            career_mapping = {
                'software developer': 'software_engineer',
                'software dev': 'software_engineer',
                'programmer': 'software_engineer',
                'data analyst': 'data_scientist',
                'ml engineer': 'data_scientist',
                'marketing specialist': 'digital_marketing_specialist',
                'marketing manager': 'digital_marketing_specialist'
            }
            
            df['career_path'] = df['career_path'].replace(career_mapping)
        
        # Standardize locations
        if 'location' in df.columns:
            location_mapping = {
                'bengaluru': 'bangalore',
                'bombay': 'mumbai',
                'new delhi': 'delhi',
                'ncr': 'delhi',
                'hyderabad': 'hyderabad',
                'madras': 'chennai'
            }
            
            df['location'] = df['location'].replace(location_mapping)
        
        # Standardize education levels
        if 'education_level' in df.columns:
            education_mapping = {
                'bachelor': 'bachelors',
                'master': 'masters',
                'phd': 'doctorate',
                'b.tech': 'bachelors',
                'b.e.': 'bachelors',
                'm.tech': 'masters',
                'mba': 'masters'
            }
            
            df['education_level'] = df['education_level'].replace(education_mapping)
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data ranges and fix obvious errors"""
        
        # Validate salary ranges
        if 'salary' in df.columns:
            df = df[df['salary'] > 0]  # Remove zero/negative salaries
            df = df[df['salary'] < 10000000]  # Remove unrealistic salaries
        
        # Validate experience
        if 'experience_years' in df.columns:
            df['experience_years'] = df['experience_years'].clip(0, 50)
        
        # Validate age if present
        if 'age' in df.columns:
            df['age'] = df['age'].clip(18, 70)
        
        return df
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for better model performance"""
        self.logger.info("Starting feature engineering...")
        
        # Skills-based features
        if 'skills' in df.columns:
            df['skills_count'] = df['skills'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
            
            # Technical skills indicator
            technical_skills = ['python', 'java', 'javascript', 'sql', 'machine learning', 'data science']
            df['has_technical_skills'] = df['skills'].apply(
                lambda x: any(skill in str(x).lower() for skill in technical_skills)
            ).astype(int)
        
        # Experience-based features
        if 'experience_years' in df.columns:
            df['experience_level'] = pd.cut(
                df['experience_years'], 
                bins=[0, 2, 5, 10, float('inf')], 
                labels=['entry', 'mid', 'senior', 'expert']
            )
        
        # Salary-based features
        if 'salary' in df.columns:
            df['salary_range'] = pd.cut(
                df['salary'],
                bins=[0, 500000, 1000000, 2000000, float('inf')],
                labels=['low', 'medium', 'high', 'very_high']
            )
        
        # Location-based features (tier city classification)
        if 'location' in df.columns:
            tier1_cities = ['bangalore', 'mumbai', 'delhi', 'pune', 'hyderabad', 'chennai']
            df['city_tier'] = df['location'].apply(
                lambda x: 'tier1' if str(x).lower() in tier1_cities else 'tier2'
            )
        
        self.logger.info(f"Feature engineering complete. New shape: {df.shape}")
        return df
    
    def prepare_ml_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Prepare features for machine learning models"""
        self.logger.info("Preparing ML features...")
        
        feature_info = {}
        
        # Encode categorical variables
        categorical_columns = ['career_path', 'location', 'industry', 'education_level', 'experience_level', 'city_tier']
        
        for col in categorical_columns:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                feature_info[f'{col}_mapping'] = dict(zip(
                    self.label_encoders[col].classes_,
                    self.label_encoders[col].transform(self.label_encoders[col].classes_)
                ))
        
        # Process skills as multi-label features
        if 'skills' in df.columns:
            skills_lists = df['skills'].apply(lambda x: str(x).split(',') if pd.notna(x) else [])
            skills_binary = self.mlb.fit_transform(skills_lists)
            skills_df = pd.DataFrame(skills_binary, columns=[f'skill_{skill}' for skill in self.mlb.classes_])
            df = pd.concat([df, skills_df], axis=1)
            feature_info['skills_features'] = list(self.mlb.classes_)
        
        # Select numerical features for scaling
        numerical_features = ['salary', 'experience_years', 'skills_count', 'age']
        numerical_features = [col for col in numerical_features if col in df.columns]
        
        if numerical_features:
            df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
            feature_info['scaled_features'] = numerical_features
        
        # Define feature columns for ML
        feature_columns = []
        
        # Add encoded categorical features
        feature_columns.extend([f'{col}_encoded' for col in categorical_columns if col in df.columns])
        
        # Add numerical features
        feature_columns.extend(numerical_features)
        
        # Add skills features
        if 'skills' in df.columns:
            feature_columns.extend([f'skill_{skill}' for skill in self.mlb.classes_])
        
        # Add engineered features
        if 'has_technical_skills' in df.columns:
            feature_columns.append('has_technical_skills')
        
        feature_info['feature_columns'] = feature_columns
        
        self.logger.info(f"ML features prepared. Feature count: {len(feature_columns)}")
        return df, feature_info
    
    def split_data(self, df: pd.DataFrame, target_columns: List[str], 
                  feature_columns: List[str], test_size: float = 0.2) -> Dict:
        """Split data into train/test sets"""
        
        X = df[feature_columns]
        splits = {}
        
        for target_col in target_columns:
            if target_col in df.columns:
                y = df[target_col]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y if target_col != 'salary' else None
                )
                
                splits[target_col] = {
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test
                }
        
        self.logger.info(f"Data split complete for targets: {target_columns}")
        return splits
    
    def save_preprocessed_data(self, df: pd.DataFrame, feature_info: Dict, 
                              output_path: str = 'data/processed/'):
        """Save preprocessed data and metadata"""
        import os
        os.makedirs(output_path, exist_ok=True)
        
        # Save main dataset
        df.to_csv(f'{output_path}preprocessed_data.csv', index=False)
        
        # Save feature information
        with open(f'{output_path}feature_info.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_info = {}
            for key, value in feature_info.items():
                if isinstance(value, np.ndarray):
                    serializable_info[key] = value.tolist()
                else:
                    serializable_info[key] = value
            
            json.dump(serializable_info, f, indent=2)
        
        # Save encoders
        import pickle
        with open(f'{output_path}encoders.pkl', 'wb') as f:
            pickle.dump({
                'label_encoders': self.label_encoders,
                'scaler': self.scaler,
                'mlb': self.mlb
            }, f)
        
        self.logger.info(f"Preprocessed data saved to {output_path}")
    
    def run_full_preprocessing(self, data_sources: Dict[str, str], 
                             target_columns: List[str] = None) -> Dict:
        """Run complete preprocessing pipeline"""
        target_columns = target_columns or ['career_path', 'salary']
        
        # Load and combine data
        df = self.load_and_combine_data(data_sources)
        
        # Clean data
        df = self.clean_data(df)
        
        # Feature engineering
        df = self.feature_engineering(df)
        
        # Prepare ML features
        df, feature_info = self.prepare_ml_features(df)
        
        # Split data
        splits = self.split_data(df, target_columns, feature_info['feature_columns'])
        
        # Save preprocessed data
        self.save_preprocessed_data(df, feature_info)
        
        return {
            'dataset': df,
            'feature_info': feature_info,
            'splits': splits,
            'preprocessing_complete': True
        }

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    # Example usage - using actual raw data files
    data_sources = {
        'career_recommendations': 'data/raw/AI-based Career Recommendation System.csv',
        'salary_data': 'data/raw/Salary Data.csv'
    }
    
    results = preprocessor.run_full_preprocessing(data_sources)
    print(f"Preprocessing complete. Dataset shape: {results['dataset'].shape}")
