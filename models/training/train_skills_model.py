#!/usr/bin/env python3
"""
Skills Recommendation Model Training Script
Trains a model to recommend skills based on career paths and user profiles
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, hamming_loss
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkillsModelTrainer:
    def __init__(self, data_path='data/processed/', models_path='models/trained_models/'):
        self.data_path = data_path
        self.models_path = models_path
        self.model = None
        self.mlb = MultiLabelBinarizer()
        
        # Ensure directories exist
        os.makedirs(models_path, exist_ok=True)
        
    def load_processed_data(self):
        """Load preprocessed data"""
        try:
            # Load main dataset
            df = pd.read_csv(f'{self.data_path}preprocessed_data.csv')
            logger.info(f"Loaded dataset with shape: {df.shape}")
            
            # Load feature info
            with open(f'{self.data_path}feature_info.json', 'r') as f:
                feature_info = json.load(f)
            
            return df, feature_info
        except FileNotFoundError as e:
            logger.error(f"Processed data not found: {e}")
            logger.info("Please run data preprocessing first")
            return None, None
    
    def prepare_skills_features(self, df, feature_info):
        """Prepare features specifically for skills recommendation"""
        # Define skills-relevant features
        skills_features = [
            'career_path', 'experience_years', 'education_level', 'industry_type',
            'current_role', 'location_tier'
        ]
        
        # Create synthetic features if they don't exist
        if 'career_path' not in df.columns:
            careers = [
                'Software Engineer', 'Data Scientist', 'Product Manager', 'Business Analyst',
                'Marketing Manager', 'Financial Analyst', 'HR Manager', 'Operations Manager',
                'Designer', 'Sales Manager', 'Consultant', 'DevOps Engineer'
            ]
            df['career_path'] = np.random.choice(careers, size=len(df))
        
        if 'experience_years' not in df.columns:
            df['experience_years'] = np.random.randint(0, 15, size=len(df))
        
        if 'education_level' not in df.columns:
            df['education_level'] = np.random.choice(['Bachelor', 'Master', 'PhD'], size=len(df))
        
        if 'industry_type' not in df.columns:
            industries = ['IT', 'Finance', 'Healthcare', 'Education', 'Manufacturing', 'Consulting']
            df['industry_type'] = np.random.choice(industries, size=len(df))
        
        if 'current_role' not in df.columns:
            df['current_role'] = df['career_path']  # Use career_path as current_role
        
        if 'location_tier' not in df.columns:
            df['location_tier'] = np.random.choice(['Tier1', 'Tier2', 'Tier3'], size=len(df))
        
        # Create skills target if it doesn't exist
        if 'required_skills' not in df.columns:
            # Define skill sets for different careers
            career_skills = {
                'Software Engineer': ['Python', 'Java', 'JavaScript', 'SQL', 'Git', 'Agile', 'Problem Solving'],
                'Data Scientist': ['Python', 'R', 'SQL', 'Machine Learning', 'Statistics', 'Tableau', 'Excel'],
                'Product Manager': ['Product Strategy', 'Agile', 'Analytics', 'Communication', 'Leadership', 'Market Research'],
                'Business Analyst': ['SQL', 'Excel', 'Analytics', 'Communication', 'Process Improvement', 'Documentation'],
                'Marketing Manager': ['Digital Marketing', 'Analytics', 'Communication', 'Content Creation', 'SEO', 'Social Media'],
                'Financial Analyst': ['Excel', 'Financial Modeling', 'SQL', 'Analytics', 'Accounting', 'Presentation'],
                'HR Manager': ['Communication', 'Leadership', 'Recruitment', 'Employee Relations', 'HR Analytics', 'Training'],
                'Operations Manager': ['Process Improvement', 'Leadership', 'Analytics', 'Project Management', 'Communication'],
                'Designer': ['Design Thinking', 'Figma', 'Adobe Creative Suite', 'Prototyping', 'User Research', 'Communication'],
                'DevOps Engineer': ['AWS', 'Docker', 'Kubernetes', 'CI/CD', 'Linux', 'Python', 'Monitoring'],
                'Sales Manager': ['Communication', 'Negotiation', 'CRM', 'Leadership', 'Analytics', 'Presentation'],
                'Consultant': ['Problem Solving', 'Communication', 'Analytics', 'Presentation', 'Industry Knowledge']
            }
            
            skills_list = []
            for _, row in df.iterrows():
                career = row['career_path']
                base_skills = career_skills.get(career, ['Communication', 'Problem Solving', 'Teamwork'])
                
                # Add some randomness and experience-based skills
                num_skills = min(len(base_skills), 3 + row['experience_years'] // 2)
                selected_skills = np.random.choice(base_skills, size=num_skills, replace=False).tolist()
                
                # Add some general skills based on experience
                if row['experience_years'] > 5:
                    selected_skills.extend(['Leadership', 'Mentoring'])
                if row['experience_years'] > 10:
                    selected_skills.extend(['Strategic Planning', 'Team Management'])
                
                skills_list.append(selected_skills)
            
            df['required_skills'] = skills_list
        else:
            # If skills column exists but is string, convert to list
            if isinstance(df['required_skills'].iloc[0], str):
                df['required_skills'] = df['required_skills'].apply(
                    lambda x: [skill.strip() for skill in x.split(',') if skill.strip()]
                )
        
        # Select available features
        available_features = [col for col in skills_features if col in df.columns]
        logger.info(f"Using features: {available_features}")
        
        return df[available_features], df['required_skills']
    
    def train_model(self, X, y):
        """Train the skills recommendation model"""
        # Encode categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        X_encoded = X.copy()
        
        encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            encoders[col] = le
        
        # Encode skills using MultiLabelBinarizer
        y_encoded = self.mlb.fit_transform(y)
        
        logger.info(f"Number of unique skills: {len(self.mlb.classes_)}")
        logger.info(f"Top 10 skills: {list(self.mlb.classes_[:10])}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training set size: {X_train.shape[0]}")
        logger.info(f"Test set size: {X_test.shape[0]}")
        
        # Train Multi-output Random Forest
        base_model = RandomForestClassifier(
            n_estimators=50,  # Reduced for multi-output
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model = MultiOutputClassifier(base_model, n_jobs=-1)
        
        logger.info("Training skills recommendation model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        # Calculate Hamming Loss (lower is better)
        hamming = hamming_loss(y_test, y_pred)
        
        # Calculate accuracy for each skill
        skill_accuracies = []
        for i, skill in enumerate(self.mlb.classes_):
            accuracy = (y_test[:, i] == y_pred[:, i]).mean()
            skill_accuracies.append(accuracy)
        
        avg_accuracy = np.mean(skill_accuracies)
        
        logger.info(f"Model Performance:")
        logger.info(f"  Hamming Loss: {hamming:.4f}")
        logger.info(f"  Average Skill Accuracy: {avg_accuracy:.4f}")
        
        # Feature importance (average across all outputs)
        feature_importances = []
        for estimator in self.model.estimators_:
            feature_importances.append(estimator.feature_importances_)
        
        avg_importance = np.mean(feature_importances, axis=0)
        
        feature_importance = pd.DataFrame({
            'feature': X_encoded.columns,
            'importance': avg_importance
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 5 Most Important Features:")
        print(feature_importance.head())
        
        return {
            'model': self.model,
            'mlb': self.mlb,
            'feature_encoders': encoders,
            'feature_names': list(X_encoded.columns),
            'skills': list(self.mlb.classes_),
            'hamming_loss': hamming,
            'avg_accuracy': avg_accuracy,
            'feature_importance': feature_importance.to_dict('records')
        }
    
    def save_model(self, model_info):
        """Save the trained model"""
        model_path = f'{self.models_path}skills_model.pkl'
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_info, f)
        
        logger.info(f"Skills model saved to: {model_path}")
        
        # Save model metadata
        metadata = {
            'model_type': 'skills_recommendation',
            'hamming_loss': model_info['hamming_loss'],
            'avg_accuracy': model_info['avg_accuracy'],
            'features': model_info['feature_names'],
            'skills': model_info['skills'],
            'num_skills': len(model_info['skills']),
            'feature_importance': model_info['feature_importance']
        }
        
        with open(f'{self.models_path}skills_model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return model_path
    
    def run_training(self):
        """Run the complete training pipeline"""
        logger.info("Starting skills model training...")
        
        # Load data
        df, feature_info = self.load_processed_data()
        if df is None:
            return None
        
        # Prepare features
        X, y = self.prepare_skills_features(df, feature_info)
        
        # Train model
        model_info = self.train_model(X, y)
        
        # Save model
        model_path = self.save_model(model_info)
        
        logger.info("Skills model training completed successfully!")
        return model_path

if __name__ == "__main__":
    trainer = SkillsModelTrainer()
    model_path = trainer.run_training()
    
    if model_path:
        print(f"\n✅ Skills model training completed!")
        print(f"📁 Model saved at: {model_path}")
    else:
        print("\n❌ Training failed. Please check the logs.")