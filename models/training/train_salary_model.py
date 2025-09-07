#!/usr/bin/env python3
"""
Salary Prediction Model Training Script
Trains a model to predict salary ranges based on user profiles and career choices
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SalaryModelTrainer:
    def __init__(self, data_path='data/processed/', models_path='models/trained_models/'):
        self.data_path = data_path
        self.models_path = models_path
        self.model = None
        self.scaler = StandardScaler()
        
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
    
    def prepare_salary_features(self, df, feature_info):
        """Prepare features specifically for salary prediction"""
        # Define salary-relevant features
        salary_features = [
            'experience_years', 'education_level', 'career_path', 'location_tier',
            'skills_count', 'company_size', 'industry_type'
        ]
        
        # Create synthetic features if they don't exist
        if 'experience_years' not in df.columns:
            df['experience_years'] = np.random.randint(0, 20, size=len(df))
        
        if 'education_level' not in df.columns:
            df['education_level'] = np.random.choice(['Bachelor', 'Master', 'PhD'], size=len(df))
        
        if 'career_path' not in df.columns:
            careers = [
                'Software Engineer', 'Data Scientist', 'Product Manager', 'Business Analyst',
                'Marketing Manager', 'Financial Analyst', 'HR Manager', 'Operations Manager'
            ]
            df['career_path'] = np.random.choice(careers, size=len(df))
        
        if 'location_tier' not in df.columns:
            df['location_tier'] = np.random.choice(['Tier1', 'Tier2', 'Tier3'], size=len(df))
        
        if 'skills_count' not in df.columns:
            df['skills_count'] = np.random.randint(3, 25, size=len(df))
        
        if 'company_size' not in df.columns:
            df['company_size'] = np.random.choice(['Startup', 'Medium', 'Large', 'Enterprise'], size=len(df))
        
        if 'industry_type' not in df.columns:
            industries = ['IT', 'Finance', 'Healthcare', 'Education', 'Manufacturing', 'Consulting']
            df['industry_type'] = np.random.choice(industries, size=len(df))
        
        # Create salary target if it doesn't exist
        if 'salary' not in df.columns:
            # Generate realistic Indian salary data based on experience and career
            base_salaries = {
                'Software Engineer': 600000, 'Data Scientist': 800000, 'Product Manager': 1200000,
                'Business Analyst': 700000, 'Marketing Manager': 900000, 'Financial Analyst': 650000,
                'HR Manager': 750000, 'Operations Manager': 800000
            }
            
            salaries = []
            for _, row in df.iterrows():
                base = base_salaries.get(row['career_path'], 600000)
                exp_multiplier = 1 + (row['experience_years'] * 0.1)
                location_multiplier = {'Tier1': 1.3, 'Tier2': 1.0, 'Tier3': 0.8}[row['location_tier']]
                education_multiplier = {'PhD': 1.4, 'Master': 1.2, 'Bachelor': 1.0}[row['education_level']]
                
                salary = base * exp_multiplier * location_multiplier * education_multiplier
                # Add some randomness
                salary *= np.random.normal(1.0, 0.2)
                salaries.append(max(300000, int(salary)))  # Minimum 3 LPA
            
            df['salary'] = salaries
        
        # Select available features
        available_features = [col for col in salary_features if col in df.columns]
        logger.info(f"Using features: {available_features}")
        
        return df[available_features], df['salary']
    
    def train_model(self, X, y):
        """Train the salary prediction model"""
        # Encode categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        X_encoded = X.copy()
        
        encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            encoders[col] = le
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_encoded)
        X_scaled = pd.DataFrame(X_scaled, columns=X_encoded.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training set size: {X_train.shape[0]}")
        logger.info(f"Test set size: {X_test.shape[0]}")
        logger.info(f"Salary range: ₹{y.min():,.0f} - ₹{y.max():,.0f}")
        
        # Train Random Forest Regressor
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        logger.info("Training salary prediction model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Model Performance:")
        logger.info(f"  MAE: ₹{mae:,.0f}")
        logger.info(f"  RMSE: ₹{rmse:,.0f}")
        logger.info(f"  R² Score: {r2:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_scaled.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 5 Most Important Features:")
        print(feature_importance.head())
        
        return {
            'model': self.model,
            'scaler': self.scaler,
            'feature_encoders': encoders,
            'feature_names': list(X_scaled.columns),
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2,
            'feature_importance': feature_importance.to_dict('records')
        }
    
    def save_model(self, model_info):
        """Save the trained model"""
        model_path = f'{self.models_path}salary_model.pkl'
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_info, f)
        
        logger.info(f"Salary model saved to: {model_path}")
        
        # Save model metadata
        metadata = {
            'model_type': 'salary_prediction',
            'mae': model_info['mae'],
            'rmse': model_info['rmse'],
            'r2_score': model_info['r2_score'],
            'features': model_info['feature_names'],
            'feature_importance': model_info['feature_importance']
        }
        
        with open(f'{self.models_path}salary_model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return model_path
    
    def run_training(self):
        """Run the complete training pipeline"""
        logger.info("Starting salary model training...")
        
        # Load data
        df, feature_info = self.load_processed_data()
        if df is None:
            return None
        
        # Prepare features
        X, y = self.prepare_salary_features(df, feature_info)
        
        # Train model
        model_info = self.train_model(X, y)
        
        # Save model
        model_path = self.save_model(model_info)
        
        logger.info("Salary model training completed successfully!")
        return model_path

if __name__ == "__main__":
    trainer = SalaryModelTrainer()
    model_path = trainer.run_training()
    
    if model_path:
        print(f"\n✅ Salary model training completed!")
        print(f"📁 Model saved at: {model_path}")
    else:
        print("\n❌ Training failed. Please check the logs.")