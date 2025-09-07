#!/usr/bin/env python3
"""
Career Recommendation Model Training Script
Trains a model to predict suitable career paths based on user profiles
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CareerModelTrainer:
    def __init__(self, data_path='data/processed/', models_path='models/trained_models/'):
        self.data_path = data_path
        self.models_path = models_path
        self.model = None
        self.label_encoder = LabelEncoder()
        
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
    
    def prepare_career_features(self, df, feature_info):
        """Prepare features specifically for career prediction"""
        # Define career-relevant features
        career_features = [
            'education_level', 'experience_years', 'skills_count',
            'personality_openness', 'personality_conscientiousness', 
            'personality_extraversion', 'personality_agreeableness',
            'personality_neuroticism', 'location_tier'
        ]
        
        # Create synthetic features if they don't exist
        if 'education_level' not in df.columns:
            df['education_level'] = np.random.choice(['Bachelor', 'Master', 'PhD'], size=len(df))
        
        if 'experience_years' not in df.columns:
            df['experience_years'] = np.random.randint(0, 15, size=len(df))
        
        if 'skills_count' not in df.columns:
            df['skills_count'] = np.random.randint(3, 20, size=len(df))
        
        # Add personality features (neutral if not available)
        personality_traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        for trait in personality_traits:
            col_name = f'personality_{trait}'
            if col_name not in df.columns:
                df[col_name] = np.random.normal(5, 1.5, size=len(df))  # Normal distribution around neutral
        
        if 'location_tier' not in df.columns:
            df['location_tier'] = np.random.choice(['Tier1', 'Tier2', 'Tier3'], size=len(df))
        
        # Ensure we have a target column
        if 'career_path' not in df.columns:
            # Create synthetic career paths based on common Indian careers
            careers = [
                'Software Engineer', 'Data Scientist', 'Product Manager', 'Business Analyst',
                'Marketing Manager', 'Financial Analyst', 'HR Manager', 'Operations Manager',
                'Sales Manager', 'Consultant', 'Teacher', 'Doctor', 'Engineer', 'Designer'
            ]
            df['career_path'] = np.random.choice(careers, size=len(df))
        
        # Select available features
        available_features = [col for col in career_features if col in df.columns]
        logger.info(f"Using features: {available_features}")
        
        return df[available_features], df['career_path']
    
    def train_model(self, X, y):
        """Train the career recommendation model"""
        # Encode categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        X_encoded = X.copy()
        
        encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            encoders[col] = le
        
        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        logger.info(f"Training set size: {X_train.shape[0]}")
        logger.info(f"Test set size: {X_test.shape[0]}")
        
        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        logger.info("Training career recommendation model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model accuracy: {accuracy:.4f}")
        logger.info("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.label_encoder.classes_))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_encoded.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 5 Most Important Features:")
        print(feature_importance.head())
        
        return {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_encoders': encoders,
            'feature_names': list(X_encoded.columns),
            'accuracy': accuracy,
            'feature_importance': feature_importance.to_dict('records')
        }
    
    def save_model(self, model_info):
        """Save the trained model"""
        model_path = f'{self.models_path}career_model.pkl'
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_info, f)
        
        logger.info(f"Career model saved to: {model_path}")
        
        # Save model metadata
        metadata = {
            'model_type': 'career_recommendation',
            'accuracy': model_info['accuracy'],
            'features': model_info['feature_names'],
            'target_classes': list(model_info['label_encoder'].classes_),
            'feature_importance': model_info['feature_importance']
        }
        
        with open(f'{self.models_path}career_model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return model_path
    
    def run_training(self):
        """Run the complete training pipeline"""
        logger.info("Starting career model training...")
        
        # Load data
        df, feature_info = self.load_processed_data()
        if df is None:
            return None
        
        # Prepare features
        X, y = self.prepare_career_features(df, feature_info)
        
        # Train model
        model_info = self.train_model(X, y)
        
        # Save model
        model_path = self.save_model(model_info)
        
        logger.info("Career model training completed successfully!")
        return model_path

if __name__ == "__main__":
    trainer = CareerModelTrainer()
    model_path = trainer.run_training()
    
    if model_path:
        print(f"\n✅ Career model training completed!")
        print(f"📁 Model saved at: {model_path}")
    else:
        print("\n❌ Training failed. Please check the logs.")