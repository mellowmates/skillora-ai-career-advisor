from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, hamming_loss
import pandas as pd
import numpy as np
import pickle
import json
from typing import Dict, List, Tuple, Optional
import logging

class SkillsPredictionModel:
    """Predicts required skills for given career or profile"""
    
    def __init__(self, base_estimator_params: Dict = None):
        # Default parameters for base estimator
        default_params = {
            'n_estimators': 100,
            'random_state': 42,
            'max_depth': 10,
            'min_samples_split': 5
        }
        
        if base_estimator_params:
            default_params.update(base_estimator_params)
        
        # Use RandomForest as base estimator for multi-output classification
        base_estimator = RandomForestClassifier(**default_params)
        self.model = MultiOutputClassifier(base_estimator, n_jobs=-1)
        
        self.base_params = default_params
        self.is_trained = False
        self.skill_names = []
        self.performance_metrics = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame, 
              X_val: pd.DataFrame = None, y_val: pd.DataFrame = None) -> Dict:
        """Train the skills prediction model"""
        self.logger.info("Starting skills prediction model training...")
        
        # Store skill names
        self.skill_names = list(y_train.columns)
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate on training set
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        train_hamming_loss = hamming_loss(y_train, train_pred)
        
        metrics = {
            'train_accuracy': train_accuracy,
            'train_hamming_loss': train_hamming_loss,
            'feature_count': len(X_train.columns),
            'skills_count': len(self.skill_names),
            'samples_count': len(X_train)
        }
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
            val_hamming_loss = hamming_loss(y_val, val_pred)
            
            metrics['val_accuracy'] = val_accuracy
            metrics['val_hamming_loss'] = val_hamming_loss
            
            # Per-skill performance
            skill_performance = {}
            for i, skill in enumerate(self.skill_names):
                skill_acc = accuracy_score(y_val.iloc[:, i], val_pred[:, i])
                skill_performance[skill] = skill_acc
            
            metrics['per_skill_accuracy'] = skill_performance
        
        self.performance_metrics = metrics
        
        self.logger.info(f"Training complete. Train accuracy: {train_accuracy:.4f}")
        if 'val_accuracy' in metrics:
            self.logger.info(f"Validation accuracy: {metrics['val_accuracy']:.4f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict required skills (binary predictions)"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict skill probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get probabilities for each skill (each estimator returns probabilities)
        probas = []
        for i, estimator in enumerate(self.model.estimators_):
            # Get probability of positive class (skill required)
            skill_proba = estimator.predict_proba(X)[:, 1]  # Probability of class 1
            probas.append(skill_proba)
        
        return np.array(probas).T  # Transpose to get (samples, skills)
    
    def get_skill_recommendations(self, X: pd.DataFrame, 
                                threshold: float = 0.5, 
                                top_k: int = None) -> List[Dict]:
        """Get skill recommendations with confidence scores"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")
        
        probabilities = self.predict_proba(X)
        
        recommendations = []
        
        for i, probs in enumerate(probabilities):
            # Create skill-probability pairs
            skill_probs = list(zip(self.skill_names, probs))
            
            # Filter by threshold
            recommended_skills = [
                {'skill': skill, 'confidence': float(prob), 'required': prob >= threshold}
                for skill, prob in skill_probs
            ]
            
            # Sort by confidence
            recommended_skills.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Limit to top-k if specified
            if top_k:
                recommended_skills = recommended_skills[:top_k]
            
            recommendations.append(recommended_skills)
        
        return recommendations
    
    def get_skills_gap_analysis(self, X: pd.DataFrame, 
                               current_skills: List[str],
                               threshold: float = 0.5) -> Dict:
        """Analyze skill gaps for a profile"""
        if not self.is_trained:
            raise ValueError("Model must be trained before analysis")
        
        # Get skill recommendations
        recommendations = self.get_skill_recommendations(X, threshold=threshold)
        
        if not recommendations:
            return {}
        
        profile_recommendations = recommendations[0]  # Assuming single profile
        
        # Identify required skills
        required_skills = [
            rec['skill'] for rec in profile_recommendations 
            if rec['required']
        ]
        
        # Calculate gaps
        skill_gaps = [skill for skill in required_skills if skill not in current_skills]
        existing_skills = [skill for skill in current_skills if skill in required_skills]
        
        # Prioritize gaps by confidence
        prioritized_gaps = [
            rec for rec in profile_recommendations
            if rec['skill'] in skill_gaps
        ]
        
        gap_analysis = {
            'total_required_skills': len(required_skills),
            'existing_skills': existing_skills,
            'existing_skills_count': len(existing_skills),
            'skill_gaps': skill_gaps,
            'skill_gaps_count': len(skill_gaps),
            'gap_percentage': len(skill_gaps) / len(required_skills) if required_skills else 0,
            'prioritized_gaps': prioritized_gaps[:10],  # Top 10 priority gaps
            'skill_match_score': len(existing_skills) / len(required_skills) if required_skills else 0
        }
        
        return gap_analysis
    
    def get_feature_importance_per_skill(self) -> Dict:
        """Get feature importance for each skill prediction"""
        if not self.is_trained:
            return {}
        
        importance_dict = {}
        
        for i, estimator in enumerate(self.model.estimators_):
            skill_name = self.skill_names[i]
            if hasattr(estimator, 'feature_importances_'):
                importance_dict[skill_name] = estimator.feature_importances_.tolist()
        
        return importance_dict
    
    def predict_skills_for_career(self, career_path: str, 
                                 career_features: pd.DataFrame) -> Dict:
        """Predict skills specifically for a given career path"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get skill predictions
        skill_recommendations = self.get_skill_recommendations(career_features, top_k=15)
        
        if not skill_recommendations:
            return {}
        
        career_skills = skill_recommendations[0]
        
        # Categorize skills by importance/confidence
        critical_skills = [s for s in career_skills if s['confidence'] >= 0.8]
        important_skills = [s for s in career_skills if 0.6 <= s['confidence'] < 0.8]
        nice_to_have_skills = [s for s in career_skills if 0.4 <= s['confidence'] < 0.6]
        
        return {
            'career_path': career_path,
            'total_skills': len(career_skills),
            'critical_skills': critical_skills,
            'important_skills': important_skills,
            'nice_to_have_skills': nice_to_have_skills,
            'all_skills': career_skills
        }
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict:
        """Comprehensive model evaluation"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Overall metrics
        accuracy = accuracy_score(y_test, y_pred)
        hamming_loss_score = hamming_loss(y_test, y_pred)
        
        # Per-skill evaluation
        skill_metrics = {}
        for i, skill in enumerate(self.skill_names):
            skill_true = y_test.iloc[:, i]
            skill_pred = y_pred[:, i]
            
            skill_acc = accuracy_score(skill_true, skill_pred)
            
            # Classification report for this skill
            try:
                skill_report = classification_report(skill_true, skill_pred, output_dict=True)
                skill_metrics[skill] = {
                    'accuracy': skill_acc,
                    'precision': skill_report['1']['precision'] if '1' in skill_report else 0,
                    'recall': skill_report['1']['recall'] if '1' in skill_report else 0,
                    'f1_score': skill_report['1']['f1-score'] if '1' in skill_report else 0
                }
            except:
                skill_metrics[skill] = {
                    'accuracy': skill_acc,
                    'precision': 0,
                    'recall': 0,
                    'f1_score': 0
                }
        
        # Average metrics across skills
        avg_precision = np.mean([metrics['precision'] for metrics in skill_metrics.values()])
        avg_recall = np.mean([metrics['recall'] for metrics in skill_metrics.values()])
        avg_f1 = np.mean([metrics['f1_score'] for metrics in skill_metrics.values()])
        
        evaluation_results = {
            'overall_accuracy': accuracy,
            'hamming_loss': hamming_loss_score,
            'average_precision': avg_precision,
            'average_recall': avg_recall,
            'average_f1_score': avg_f1,
            'per_skill_metrics': skill_metrics,
            'skills_count': len(self.skill_names)
        }
        
        return evaluation_results
    
    def save_model(self, filepath: str, include_metadata: bool = True):
        """Save trained model and metadata"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'base_params': self.base_params,
            'skill_names': self.skill_names,
            'performance_metrics': self.performance_metrics,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Skills prediction model saved to {filepath}")
        
        # Save metadata
        if include_metadata:
            metadata_filepath = filepath.replace('.pkl', '_metadata.json')
            metadata = {
                'model_type': 'SkillsPredictionModel',
                'base_params': self.base_params,
                'skills_count': len(self.skill_names),
                'skill_names': self.skill_names,
                'performance_metrics': self.performance_metrics
            }
            
            with open(metadata_filepath, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def load_model(self, filepath: str):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.base_params = model_data['base_params']
        self.skill_names = model_data['skill_names']
        self.performance_metrics = model_data['performance_metrics']
        self.is_trained = model_data['is_trained']
        
        self.logger.info(f"Skills prediction model loaded from {filepath}")

if __name__ == "__main__":
    # Example usage
    model = SkillsPredictionModel()
    
    # Mock training data
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.rand(1000, 15), columns=[f'feature_{i}' for i in range(15)])
    
    # Mock multi-label target (skills)
    skills = ['Python', 'SQL', 'Machine Learning', 'Communication', 'Leadership']
    y_train = pd.DataFrame(np.random.randint(0, 2, (1000, len(skills))), columns=skills)
    
    # Train model
    metrics = model.train(X_train, y_train)
    print(f"Training completed with accuracy: {metrics['train_accuracy']:.4f}")
    
    # Save model
    model.save_model('models/trained_models/skills_predictor.pkl')
