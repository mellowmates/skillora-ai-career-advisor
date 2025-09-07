import pandas as pd
import numpy as np
import os
import json
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Import model classes
from career_recommendation_model import CareerRecommendationModel
from skills_prediction_model import SkillsPredictionModel
from salary_prediction_model import SalaryPredictionModel

# Import data preprocessor
import sys
sys.path.append('../data_collection/')
from data_preprocessor import DataPreprocessor

class ModelTrainer:
    """Main training pipeline for all Skillora ML models"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        
        # Initialize models
        self.career_model = CareerRecommendationModel()
        self.skills_model = SkillsPredictionModel()
        self.salary_model = SalaryPredictionModel()
        
        # Initialize preprocessor
        self.preprocessor = DataPreprocessor()
        
        # Training results
        self.training_results = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        os.makedirs(self.config['model_output_dir'], exist_ok=True)
        os.makedirs(self.config['results_output_dir'], exist_ok=True)
    
    def _get_default_config(self) -> Dict:
        """Get default training configuration"""
        return {
            'data_sources': {
                'career_data': 'data/raw/AI-based Career Recommendation System.csv',
                'salary_data': 'data/raw/Salary Data.csv'
            },
            'model_output_dir': 'models/trained_models/',
            'results_output_dir': 'models/training_results/',
            'test_size': 0.2,
            'validation_size': 0.2,
            'target_columns': {
                'career': 'career_path',
                'salary': 'salary',
                'skills': ['skill_python', 'skill_java', 'skill_sql', 'skill_machine learning', 'skill_communication']
            },
            'hyperparameter_tuning': False,
            'cross_validation': True,
            'save_models': True,
            'generate_reports': True
        }
    
    def prepare_data(self) -> Dict:
        """Prepare and preprocess all data for training"""
        self.logger.info("Starting data preparation...")
        
        # Run preprocessing pipeline
        preprocessing_results = self.preprocessor.run_full_preprocessing(
            data_sources=self.config['data_sources'],
            target_columns=list(self.config['target_columns'].values())
        )
        
        self.dataset = preprocessing_results['dataset']
        self.feature_info = preprocessing_results['feature_info']
        self.data_splits = preprocessing_results['splits']
        
        self.logger.info(f"Data preparation complete. Dataset shape: {self.dataset.shape}")
        return preprocessing_results
    
    def train_career_recommendation_model(self) -> Dict:
        """Train the career recommendation model"""
        self.logger.info("Training career recommendation model...")
        
        career_target = self.config['target_columns']['career']
        
        if career_target not in self.data_splits:
            self.logger.error(f"Career target '{career_target}' not found in data splits")
            return {}
        
        # Get training data
        split_data = self.data_splits[career_target]
        X_train, X_test = split_data['X_train'], split_data['X_test']
        y_train, y_test = split_data['y_train'], split_data['y_test']
        
        # Further split training data for validation
        from sklearn.model_selection import train_test_split
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train, y_train, test_size=self.config['validation_size'], random_state=42
        )
        
        # Hyperparameter tuning if enabled
        if self.config['hyperparameter_tuning']:
            tuning_results = self.career_model.tune_hyperparameters(X_train, y_train)
            self.logger.info(f"Best parameters: {tuning_results['best_params']}")
        
        # Train the model
        training_metrics = self.career_model.train(X_train_sub, y_train_sub, X_val, y_val)
        
        # Evaluate on test set
        test_metrics = self.career_model.evaluate_model(X_test, y_test)
        
        # Combine results
        results = {
            'model_type': 'career_recommendation',
            'training_metrics': training_metrics,
            'test_metrics': test_metrics,
            'feature_importance': self.career_model.get_feature_importance(15),
            'model_params': self.career_model.model_params
        }
        
        # Save model if configured
        if self.config['save_models']:
            model_path = os.path.join(self.config['model_output_dir'], 'career_classifier.pkl')
            self.career_model.save_model(model_path)
            results['model_path'] = model_path
        
        self.training_results['career_model'] = results
        self.logger.info(f"Career model training complete. Test accuracy: {test_metrics['accuracy']:.4f}")
        
        return results
    
    def train_skills_prediction_model(self) -> Dict:
        """Train the skills prediction model"""
        self.logger.info("Training skills prediction model...")
        
        # Get skills columns from feature info
        skills_features = [col for col in self.dataset.columns if col.startswith('skill_')]
        
        if not skills_features:
            self.logger.error("No skill features found in dataset")
            return {}
        
        # Prepare skills data
        feature_columns = self.feature_info['feature_columns']
        X = self.dataset[feature_columns]
        y_skills = self.dataset[skills_features]
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_skills, test_size=self.config['test_size'], random_state=42
        )
        
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train, y_train, test_size=self.config['validation_size'], random_state=42
        )
        
        # Train the model
        training_metrics = self.skills_model.train(X_train_sub, y_train_sub, X_val, y_val)
        
        # Evaluate on test set
        test_metrics = self.skills_model.evaluate_model(X_test, y_test)
        
        # Get feature importance
        feature_importance = self.skills_model.get_feature_importance_per_skill()
        
        results = {
            'model_type': 'skills_prediction',
            'training_metrics': training_metrics,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance,
            'skills_count': len(skills_features),
            'skills_names': skills_features
        }
        
        # Save model if configured
        if self.config['save_models']:
            model_path = os.path.join(self.config['model_output_dir'], 'skills_predictor.pkl')
            self.skills_model.save_model(model_path)
            results['model_path'] = model_path
        
        self.training_results['skills_model'] = results
        self.logger.info(f"Skills model training complete. Test accuracy: {test_metrics['overall_accuracy']:.4f}")
        
        return results
    
    def train_salary_prediction_model(self) -> Dict:
        """Train the salary prediction model"""
        self.logger.info("Training salary prediction model...")
        
        salary_target = self.config['target_columns']['salary']
        
        if salary_target not in self.data_splits:
            self.logger.error(f"Salary target '{salary_target}' not found in data splits")
            return {}
        
        # Get training data
        split_data = self.data_splits[salary_target]
        X_train, X_test = split_data['X_train'], split_data['X_test']
        y_train, y_test = split_data['y_train'], split_data['y_test']
        
        # Further split for validation
        from sklearn.model_selection import train_test_split
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train, y_train, test_size=self.config['validation_size'], random_state=42
        )
        
        # Hyperparameter tuning if enabled
        if self.config['hyperparameter_tuning']:
            tuning_results = self.salary_model.tune_hyperparameters(X_train, y_train)
            self.logger.info(f"Best parameters: {tuning_results['best_params']}")
        
        # Train the model
        training_metrics = self.salary_model.train(X_train_sub, y_train_sub, X_val, y_val)
        
        # Evaluate on test set
        test_metrics = self.salary_model.evaluate_model(X_test, y_test)
        
        results = {
            'model_type': 'salary_prediction',
            'training_metrics': training_metrics,
            'test_metrics': test_metrics,
            'feature_importance': self.salary_model.get_feature_importance(15),
            'salary_stats': self.salary_model.salary_stats
        }
        
        # Save model if configured
        if self.config['save_models']:
            model_path = os.path.join(self.config['model_output_dir'], 'salary_estimator.pkl')
            self.salary_model.save_model(model_path)
            results['model_path'] = model_path
        
        self.training_results['salary_model'] = results
        self.logger.info(f"Salary model training complete. Test R²: {test_metrics['r2_score']:.4f}")
        
        return results
    
    def train_all_models(self) -> Dict:
        """Train all models in sequence"""
        self.logger.info("Starting training pipeline for all models...")
        
        start_time = datetime.now()
        
        # Prepare data
        self.prepare_data()
        
        # Train individual models
        career_results = self.train_career_recommendation_model()
        skills_results = self.train_skills_prediction_model()
        salary_results = self.train_salary_prediction_model()
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        # Compile overall results
        overall_results = {
            'training_timestamp': start_time.isoformat(),
            'training_duration_seconds': training_duration.total_seconds(),
            'dataset_info': {
                'total_samples': len(self.dataset),
                'feature_count': len(self.feature_info['feature_columns']),
                'data_sources': list(self.config['data_sources'].keys())
            },
            'model_results': {
                'career_model': career_results,
                'skills_model': skills_results,
                'salary_model': salary_results
            },
            'config_used': self.config
        }
        
        # Generate training report if configured
        if self.config['generate_reports']:
            self.generate_training_report(overall_results)
        
        self.logger.info(f"All models training complete in {training_duration.total_seconds():.2f} seconds")
        
        return overall_results
    
    def generate_training_report(self, results: Dict):
        """Generate comprehensive training report"""
        report_path = os.path.join(
            self.config['results_output_dir'], 
            f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        # Save detailed results
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate summary report
        summary_path = os.path.join(
            self.config['results_output_dir'],
            "training_summary.txt"
        )
        
        with open(summary_path, 'w') as f:
            f.write("Skillora ML Models Training Summary\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Training Date: {results['training_timestamp']}\n")
            f.write(f"Training Duration: {results['training_duration_seconds']:.2f} seconds\n")
            f.write(f"Dataset Size: {results['dataset_info']['total_samples']} samples\n")
            f.write(f"Feature Count: {results['dataset_info']['feature_count']}\n\n")
            
            # Model performance summary
            f.write("Model Performance Summary:\n")
            f.write("-" * 25 + "\n")
            
            if 'career_model' in results['model_results']:
                career_acc = results['model_results']['career_model']['test_metrics']['accuracy']
                f.write(f"Career Recommendation Model - Test Accuracy: {career_acc:.4f}\n")
            
            if 'skills_model' in results['model_results']:
                skills_acc = results['model_results']['skills_model']['test_metrics']['overall_accuracy']
                f.write(f"Skills Prediction Model - Test Accuracy: {skills_acc:.4f}\n")
            
            if 'salary_model' in results['model_results']:
                salary_r2 = results['model_results']['salary_model']['test_metrics']['r2_score']
                f.write(f"Salary Prediction Model - Test R²: {salary_r2:.4f}\n")
        
        self.logger.info(f"Training report generated: {report_path}")
        self.logger.info(f"Training summary: {summary_path}")
    
    def validate_models(self) -> Dict:
        """Validate all trained models with sample predictions"""
        validation_results = {}
        
        if not self.dataset.empty:
            # Get a sample for validation
            sample_data = self.dataset.iloc[:5][self.feature_info['feature_columns']]
            
            # Test career model
            if self.career_model.is_trained:
                try:
                    career_predictions = self.career_model.get_recommendations(sample_data)
                    validation_results['career_model'] = {
                        'status': 'success',
                        'sample_predictions': len(career_predictions)
                    }
                except Exception as e:
                    validation_results['career_model'] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            # Test skills model
            if self.skills_model.is_trained:
                try:
                    skills_predictions = self.skills_model.get_skill_recommendations(sample_data)
                    validation_results['skills_model'] = {
                        'status': 'success',
                        'sample_predictions': len(skills_predictions)
                    }
                except Exception as e:
                    validation_results['skills_model'] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            # Test salary model
            if self.salary_model.is_trained:
                try:
                    salary_predictions = self.salary_model.predict_salary_range(sample_data)
                    validation_results['salary_model'] = {
                        'status': 'success',
                        'sample_predictions': len(salary_predictions)
                    }
                except Exception as e:
                    validation_results['salary_model'] = {
                        'status': 'error',
                        'error': str(e)
                    }
        
        return validation_results

def main():
    """Main training script"""
    # Custom configuration (can be loaded from file)
    config = {
        'data_sources': {
            'career_data': 'data/raw/AI-based Career Recommendation System.csv',
            'salary_data': 'data/raw/Salary Data.csv'
        },
        'hyperparameter_tuning': True,
        'generate_reports': True
    }
    
    # Initialize and run trainer
    trainer = ModelTrainer(config)
    
    # Train all models
    results = trainer.train_all_models()
    
    # Validate models
    validation = trainer.validate_models()
    
    print("Training Pipeline Complete!")
    print(f"Results saved to: {trainer.config['results_output_dir']}")
    
    return results

if __name__ == "__main__":
    results = main()
