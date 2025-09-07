from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           mean_squared_error, mean_absolute_error, r2_score, hamming_loss, 
                           jaccard_score, precision_recall_fscore_support)
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import os

class ModelEvaluation:
    """Comprehensive evaluation system for Skillora ML models"""
    
    def __init__(self, models_dir: str = 'models/trained_models/', output_dir: str = 'models/evaluation_results/'):
        self.models_dir = models_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.evaluation_results = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_model(self, model_path: str):
        """Load a trained model from file"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            self.logger.info(f"Successfully loaded model from {model_path}")
            return model_data
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {str(e)}")
            return None
    
    def evaluate_career_model(self, model_path: str, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate career recommendation model"""
        self.logger.info("Evaluating career recommendation model...")
        
        model_data = self.load_model(model_path)
        if not model_data:
            return {}
        
        model = model_data['model']
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # Advanced metrics
        top_3_accuracy = self._calculate_top_k_accuracy(y_test, y_proba, model.classes_, k=3)
        class_distribution = pd.Series(y_test).value_counts().to_dict()
        
        # Per-class performance
        class_performance = {}
        for class_name in model.classes_:
            if class_name in report:
                class_performance[class_name] = {
                    'precision': report[class_name]['precision'],
                    'recall': report[class_name]['recall'],
                    'f1_score': report[class_name]['f1-score'],
                    'support': report[class_name]['support']
                }
        
        # Prediction confidence analysis
        confidence_stats = {
            'mean_confidence': float(np.mean(np.max(y_proba, axis=1))),
            'std_confidence': float(np.std(np.max(y_proba, axis=1))),
            'min_confidence': float(np.min(np.max(y_proba, axis=1))),
            'max_confidence': float(np.max(np.max(y_proba, axis=1)))
        }
        
        # Feature importance if available
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            feature_names = X_test.columns if hasattr(X_test, 'columns') else [f'feature_{i}' for i in range(X_test.shape[1])]
            feature_importance = dict(zip(feature_names, model.feature_importances_))
            # Get top 10 most important features
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])
        
        evaluation_results = {
            'model_type': 'career_recommendation',
            'accuracy': accuracy,
            'top_3_accuracy': top_3_accuracy,
            'macro_avg_f1': report['macro avg']['f1-score'],
            'weighted_avg_f1': report['weighted avg']['f1-score'],
            'class_performance': class_performance,
            'class_distribution': class_distribution,
            'confidence_stats': confidence_stats,
            'confusion_matrix': cm.tolist(),
            'feature_importance': feature_importance,
            'total_predictions': len(y_test),
            'unique_classes': len(model.classes_),
            'class_names': model.classes_.tolist()
        }
        
        return evaluation_results
    
    def evaluate_skills_model(self, model_path: str, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict:
        """Evaluate skills prediction model"""
        self.logger.info("Evaluating skills prediction model...")
        
        model_data = self.load_model(model_path)
        if not model_data:
            return {}
        
        model = model_data['model']
        skill_names = model_data.get('skill_names', [f'skill_{i}' for i in range(y_test.shape[1])])
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Multi-label metrics
        overall_accuracy = accuracy_score(y_test, y_pred)
        hamming_loss_score = hamming_loss(y_test, y_pred)
        jaccard_score_avg = jaccard_score(y_test, y_pred, average='macro', zero_division=0)
        
        # Per-skill evaluation
        skill_performance = {}
        for i, skill in enumerate(skill_names):
            if i < y_test.shape[1]:
                skill_true = y_test.iloc[:, i] if hasattr(y_test, 'iloc') else y_test[:, i]
                skill_pred = y_pred[:, i]
                
                skill_acc = accuracy_score(skill_true, skill_pred)
                
                try:
                    precision, recall, f1, support = precision_recall_fscore_support(
                        skill_true, skill_pred, average='binary', zero_division=0
                    )
                    
                    skill_performance[skill] = {
                        'accuracy': skill_acc,
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1_score': float(f1),
                        'support': int(np.sum(skill_true))
                    }
                except:
                    skill_performance[skill] = {
                        'accuracy': skill_acc,
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1_score': 0.0,
                        'support': int(np.sum(skill_true))
                    }
        
        # Overall performance metrics
        valid_skills = [s for s in skill_performance.values() if s['f1_score'] > 0]
        avg_metrics = {
            'avg_precision': np.mean([s['precision'] for s in valid_skills]) if valid_skills else 0,
            'avg_recall': np.mean([s['recall'] for s in valid_skills]) if valid_skills else 0,
            'avg_f1': np.mean([s['f1_score'] for s in valid_skills]) if valid_skills else 0
        }
        
        # Skills coverage analysis
        skills_coverage = {
            'total_skills': len(skill_names),
            'skills_with_positive_examples': sum(1 for s in skill_performance.values() if s['support'] > 0),
            'well_performing_skills': sum(1 for s in skill_performance.values() if s['f1_score'] > 0.5),
            'coverage_percentage': (sum(1 for s in skill_performance.values() if s['support'] > 0) / len(skill_names) * 100) if skill_names else 0
        }
        
        evaluation_results = {
            'model_type': 'skills_prediction',
            'overall_accuracy': overall_accuracy,
            'hamming_loss': hamming_loss_score,
            'jaccard_score': jaccard_score_avg,
            'average_metrics': avg_metrics,
            'per_skill_performance': skill_performance,
            'skills_coverage': skills_coverage,
            'skills_count': len(skill_names),
            'total_predictions': len(y_test)
        }
        
        return evaluation_results
    
    def evaluate_salary_model(self, model_path: str, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate salary prediction model"""
        self.logger.info("Evaluating salary prediction model...")
        
        model_data = self.load_model(model_path)
        if not model_data:
            return {}
        
        model = model_data['model']
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Percentage errors
        percentage_errors = np.abs(y_pred - y_test) / np.maximum(y_test, 1) * 100  # Avoid division by zero
        mape = np.mean(percentage_errors)
        median_ape = np.median(percentage_errors)
        
        # Prediction accuracy within ranges
        within_10_percent = np.sum(percentage_errors <= 10) / len(percentage_errors) * 100
        within_20_percent = np.sum(percentage_errors <= 20) / len(percentage_errors) * 100
        within_30_percent = np.sum(percentage_errors <= 30) / len(percentage_errors) * 100
        
        # Salary range analysis
        salary_ranges = {
            'entry_level': (0, 500000),
            'mid_level': (500000, 1000000),
            'senior_level': (1000000, 2000000),
            'executive_level': (2000000, float('inf'))
        }
        
        range_performance = {}
        for range_name, (min_sal, max_sal) in salary_ranges.items():
            mask = (y_test >= min_sal) & (y_test < max_sal)
            if np.sum(mask) > 5:  # Only evaluate if we have enough samples
                range_mape = np.mean(percentage_errors[mask])
                try:
                    range_r2 = r2_score(y_test[mask], y_pred[mask])
                except:
                    range_r2 = 0
                
                range_performance[range_name] = {
                    'count': int(np.sum(mask)),
                    'mape': float(range_mape),
                    'r2': float(range_r2),
                    'avg_actual': float(np.mean(y_test[mask])),
                    'avg_predicted': float(np.mean(y_pred[mask]))
                }
        
        # Feature importance if available
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            feature_names = X_test.columns if hasattr(X_test, 'columns') else [f'feature_{i}' for i in range(X_test.shape[1])]
            feature_importance = dict(zip(feature_names, model.feature_importances_))
            # Get top 10 most important features
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])
        
        evaluation_results = {
            'model_type': 'salary_prediction',
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mape': mape,
            'median_ape': median_ape,
            'within_10_percent': within_10_percent,
            'within_20_percent': within_20_percent,
            'within_30_percent': within_30_percent,
            'range_performance': range_performance,
            'feature_importance': feature_importance,
            'prediction_stats': {
                'mean_actual': float(np.mean(y_test)),
                'mean_predicted': float(np.mean(y_pred)),
                'std_actual': float(np.std(y_test)),
                'std_predicted': float(np.std(y_pred)),
                'min_actual': float(np.min(y_test)),
                'max_actual': float(np.max(y_test)),
                'min_predicted': float(np.min(y_pred)),
                'max_predicted': float(np.max(y_pred))
            },
            'total_predictions': len(y_test)
        }
        
        return evaluation_results
    
    def _calculate_top_k_accuracy(self, y_true: pd.Series, y_proba: np.ndarray, classes: np.ndarray, k: int = 3) -> float:
        """Calculate top-k accuracy for multi-class classification"""
        if len(classes) < k:
            k = len(classes)
        
        top_k_predictions = np.argsort(y_proba, axis=1)[:, -k:]
        
        correct = 0
        for i, true_label in enumerate(y_true):
            true_label_idx = np.where(classes == true_label)[0]
            if len(true_label_idx) > 0 and true_label_idx[0] in top_k_predictions[i]:
                correct += 1
        
        return correct / len(y_true)
    
    def compare_models(self, evaluation_results: Dict) -> Dict:
        """Compare performance across different models"""
        comparison = {
            'model_comparison': {},
            'best_performers': {},
            'improvement_suggestions': []
        }
        
        # Extract key metrics for comparison
        if 'career_model' in evaluation_results:
            career_metrics = evaluation_results['career_model']
            comparison['model_comparison']['career_model'] = {
                'accuracy': career_metrics.get('accuracy', 0),
                'f1_score': career_metrics.get('weighted_avg_f1', 0),
                'confidence': career_metrics.get('confidence_stats', {}).get('mean_confidence', 0)
            }
        
        if 'skills_model' in evaluation_results:
            skills_metrics = evaluation_results['skills_model']
            comparison['model_comparison']['skills_model'] = {
                'accuracy': skills_metrics.get('overall_accuracy', 0),
                'f1_score': skills_metrics.get('average_metrics', {}).get('avg_f1', 0),
                'coverage': skills_metrics.get('skills_coverage', {}).get('coverage_percentage', 0)
            }
        
        if 'salary_model' in evaluation_results:
            salary_metrics = evaluation_results['salary_model']
            comparison['model_comparison']['salary_model'] = {
                'r2_score': salary_metrics.get('r2_score', 0),
                'mape': salary_metrics.get('mape', 100),
                'within_20_percent': salary_metrics.get('within_20_percent', 0)
            }
        
        # Identify best performers
        if 'career_model' in comparison['model_comparison']:
            if comparison['model_comparison']['career_model']['accuracy'] > 0.8:
                comparison['best_performers']['career_model'] = 'Excellent accuracy'
            elif comparison['model_comparison']['career_model']['accuracy'] > 0.7:
                comparison['best_performers']['career_model'] = 'Good accuracy'
        
        if 'skills_model' in comparison['model_comparison']:
            if comparison['model_comparison']['skills_model']['f1_score'] > 0.7:
                comparison['best_performers']['skills_model'] = 'Good multi-label performance'
        
        if 'salary_model' in comparison['model_comparison']:
            if comparison['model_comparison']['salary_model']['r2_score'] > 0.8:
                comparison['best_performers']['salary_model'] = 'Excellent prediction accuracy'
            elif comparison['model_comparison']['salary_model']['within_20_percent'] > 70:
                comparison['best_performers']['salary_model'] = 'Good practical accuracy'
        
        # Generate improvement suggestions
        if 'career_model' in comparison['model_comparison']:
            if comparison['model_comparison']['career_model']['accuracy'] < 0.7:
                comparison['improvement_suggestions'].append('Career model: Consider feature engineering or hyperparameter tuning')
        
        if 'skills_model' in comparison['model_comparison']:
            if comparison['model_comparison']['skills_model']['f1_score'] < 0.6:
                comparison['improvement_suggestions'].append('Skills model: Address class imbalance or improve feature selection')
        
        if 'salary_model' in comparison['model_comparison']:
            if comparison['model_comparison']['salary_model']['r2_score'] < 0.7:
                comparison['improvement_suggestions'].append('Salary model: Include more relevant features or try different algorithms')
        
        return comparison
    
    def generate_evaluation_report(self, career_results: Dict, skills_results: Dict, salary_results: Dict) -> Dict:
        """Generate comprehensive evaluation report"""
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'career_model_results': career_results,
            'skills_model_results': skills_results,
            'salary_model_results': salary_results,
            'model_comparison': self.compare_models({
                'career_model': career_results,
                'skills_model': skills_results,
                'salary_model': salary_results
            }),
            'summary': {
                'total_models_evaluated': sum(1 for r in [career_results, skills_results, salary_results] if r),
                'evaluation_status': 'complete' if all([career_results, skills_results, salary_results]) else 'partial'
            }
        }
        
        return report
    
    def save_evaluation_results(self, results: Dict, filename: str = None):
        """Save evaluation results to file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'model_evaluation_report_{timestamp}.json'
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_numpy(results)
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        self.logger.info(f"Evaluation results saved to: {filepath}")
        return filepath
    
    def create_evaluation_summary(self, results: Dict) -> str:
        """Create a human-readable summary of evaluation results"""
        summary = "Skillora ML Models Evaluation Summary\n"
        summary += "=" * 50 + "\n\n"
        
        if 'career_model_results' in results and results['career_model_results']:
            career = results['career_model_results']
            summary += f"Career Recommendation Model:\n"
            summary += f"  - Accuracy: {career.get('accuracy', 0):.3f}\n"
            summary += f"  - Top-3 Accuracy: {career.get('top_3_accuracy', 0):.3f}\n"
            summary += f"  - F1 Score: {career.get('weighted_avg_f1', 0):.3f}\n"
            summary += f"  - Confidence: {career.get('confidence_stats', {}).get('mean_confidence', 0):.3f}\n\n"
        
        if 'skills_model_results' in results and results['skills_model_results']:
            skills = results['skills_model_results']
            summary += f"Skills Prediction Model:\n"
            summary += f"  - Overall Accuracy: {skills.get('overall_accuracy', 0):.3f}\n"
            summary += f"  - Average F1 Score: {skills.get('average_metrics', {}).get('avg_f1', 0):.3f}\n"
            summary += f"  - Skills Coverage: {skills.get('skills_coverage', {}).get('coverage_percentage', 0):.1f}%\n\n"
        
        if 'salary_model_results' in results and results['salary_model_results']:
            salary = results['salary_model_results']
            summary += f"Salary Prediction Model:\n"
            summary += f"  - R² Score: {salary.get('r2_score', 0):.3f}\n"
            summary += f"  - MAPE: {salary.get('mape', 0):.1f}%\n"
            summary += f"  - Within 20% accuracy: {salary.get('within_20_percent', 0):.1f}%\n\n"
        
        if 'model_comparison' in results:
            comparison = results['model_comparison']
            if comparison.get('best_performers'):
                summary += "Best Performers:\n"
                for model, performance in comparison['best_performers'].items():
                    summary += f"  - {model}: {performance}\n"
                summary += "\n"
            
            if comparison.get('improvement_suggestions'):
                summary += "Improvement Suggestions:\n"
                for suggestion in comparison['improvement_suggestions']:
                    summary += f"  - {suggestion}\n"
        
        return summary

def run_complete_evaluation():
    """Run complete evaluation workflow"""
    evaluator = ModelEvaluation()
    
    # Model paths
    models = {
        'career': 'models/trained_models/career_classifier.pkl',
        'skills': 'models/trained_models/skills_predictor.pkl',
        'salary': 'models/trained_models/salary_estimator.pkl'
    }
    
    # Load test data (this would be your actual test data)
    # For demonstration, we'll create mock test data
    try:
        # Attempt to load real test data
        X_test = pd.read_csv('data/X_test.csv')
        y_test_career = pd.read_csv('data/y_test_career.csv').squeeze()
        y_test_skills = pd.read_csv('data/y_test_skills.csv')
        y_test_salary = pd.read_csv('data/y_test_salary.csv').squeeze()
        
        # Run evaluations
        career_results = evaluator.evaluate_career_model(models['career'], X_test, y_test_career)
        skills_results = evaluator.evaluate_skills_model(models['skills'], X_test, y_test_skills)
        salary_results = evaluator.evaluate_salary_model(models['salary'], X_test, y_test_salary)
        
        # Generate comprehensive report
        full_report = evaluator.generate_evaluation_report(career_results, skills_results, salary_results)
        
        # Save results
        report_path = evaluator.save_evaluation_results(full_report)
        
        # Create summary
        summary = evaluator.create_evaluation_summary(full_report)
        print(summary)
        
        return full_report
        
    except FileNotFoundError:
        print("Test data files not found. Please ensure you have:")
        print("- data/X_test.csv")
        print("- data/y_test_career.csv") 
        print("- data/y_test_skills.csv")
        print("- data/y_test_salary.csv")
        return None

if __name__ == "__main__":
    results = run_complete_evaluation()
