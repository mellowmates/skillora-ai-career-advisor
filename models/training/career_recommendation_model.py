from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import pandas as pd
import numpy as np
import pickle
import json
from typing import Dict, List, Tuple, Optional
import logging

class CareerRecommendationModel:
    """Machine Learning model for career path recommendation"""
    
    def __init__(self, model_params: Dict = None):
        # Default parameters optimized for career recommendation
        default_params = {
            'n_estimators': 100,
            'random_state': 42,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'class_weight': 'balanced'
        }
        
        if model_params:
            default_params.update(model_params)
            
        self.model = RandomForestClassifier(**default_params)
        self.model_params = default_params
        self.is_trained = False
        self.feature_importance = None
        self.performance_metrics = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict:
        """Train the career recommendation model"""
        self.logger.info("Starting model training...")
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate feature importance
        self.feature_importance = dict(zip(X_train.columns, self.model.feature_importances_))
        
        # Evaluate on training set
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        metrics = {
            'train_accuracy': train_accuracy,
            'feature_count': len(X_train.columns),
            'samples_count': len(X_train)
        }
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
            metrics['val_accuracy'] = val_accuracy
            
            # Detailed classification report
            report = classification_report(y_val, val_pred, output_dict=True)
            metrics['classification_report'] = report
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
        metrics['cv_mean_accuracy'] = cv_scores.mean()
        metrics['cv_std_accuracy'] = cv_scores.std()
        
        self.performance_metrics = metrics
        
        self.logger.info(f"Training complete. Train accuracy: {train_accuracy:.4f}")
        if 'val_accuracy' in metrics:
            self.logger.info(f"Validation accuracy: {metrics['val_accuracy']:.4f}")
            
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict career paths"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict career path probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def get_recommendations(self, X: pd.DataFrame, top_k: int = 3) -> List[Dict]:
        """Get top-k career recommendations with confidence scores"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")
        
        probabilities = self.predict_proba(X)
        classes = self.model.classes_
        
        recommendations = []
        
        for i, probs in enumerate(probabilities):
            # Get top-k predictions
            top_indices = np.argsort(probs)[-top_k:][::-1]
            
            profile_recommendations = []
            for idx in top_indices:
                profile_recommendations.append({
                    'career_path': classes[idx],
                    'confidence': float(probs[idx]),
                    'rank': len(profile_recommendations) + 1
                })
            
            recommendations.append(profile_recommendations)
        
        return recommendations
    
    def get_feature_importance(self, top_n: int = 20) -> Dict:
        """Get top-n most important features"""
        if not self.feature_importance:
            return {}
        
        # Sort features by importance
        sorted_features = sorted(
            self.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return dict(sorted_features[:top_n])
    
    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Perform hyperparameter tuning using GridSearchCV"""
        self.logger.info("Starting hyperparameter tuning...")
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42, class_weight='balanced'),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.model_params = grid_search.best_params_
        self.is_trained = True
        
        tuning_results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_
        }
        
        self.logger.info(f"Hyperparameter tuning complete. Best score: {grid_search.best_score_:.4f}")
        return tuning_results
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Comprehensive model evaluation"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # Per-class analysis
        class_performance = {}
        for class_name in self.model.classes_:
            if class_name in report:
                class_performance[class_name] = {
                    'precision': report[class_name]['precision'],
                    'recall': report[class_name]['recall'],
                    'f1_score': report[class_name]['f1-score'],
                    'support': report[class_name]['support']
                }
        
        evaluation_results = {
            'accuracy': accuracy,
            'macro_avg_f1': report['macro avg']['f1-score'],
            'weighted_avg_f1': report['weighted avg']['f1-score'],
            'class_performance': class_performance,
            'confusion_matrix': cm.tolist(),
            'class_names': self.model.classes_.tolist()
        }
        
        return evaluation_results
    
    def save_model(self, filepath: str, include_metadata: bool = True):
        """Save trained model and metadata"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'model_params': self.model_params,
            'feature_importance': self.feature_importance,
            'performance_metrics': self.performance_metrics,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {filepath}")
        
        # Save metadata as JSON for easy inspection
        if include_metadata:
            metadata_filepath = filepath.replace('.pkl', '_metadata.json')
            metadata = {
                'model_type': 'CareerRecommendationModel',
                'model_params': self.model_params,
                'performance_metrics': self.performance_metrics,
                'feature_count': len(self.feature_importance) if self.feature_importance else 0,
                'top_features': self.get_feature_importance(10)
            }
            
            with open(metadata_filepath, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def load_model(self, filepath: str):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_params = model_data['model_params']
        self.feature_importance = model_data['feature_importance']
        self.performance_metrics = model_data['performance_metrics']
        self.is_trained = model_data['is_trained']
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def explain_prediction(self, X: pd.DataFrame, sample_idx: int = 0) -> Dict:
        """Explain prediction for a specific sample"""
        if not self.is_trained:
            raise ValueError("Model must be trained before explaining predictions")
        
        # Get prediction and probabilities
        sample = X.iloc[sample_idx:sample_idx+1]
        prediction = self.predict(sample)[0]
        probabilities = self.predict_proba(sample)[0]
        
        # Feature contributions (simplified)
        feature_values = sample.iloc[0]
        important_features = self.get_feature_importance(10)
        
        explanation = {
            'predicted_career': prediction,
            'confidence': float(max(probabilities)),
            'all_probabilities': dict(zip(self.model.classes_, probabilities)),
            'top_contributing_features': {}
        }
        
        # Get top contributing features for this prediction
        for feature, importance in important_features.items():
            if feature in feature_values.index:
                explanation['top_contributing_features'][feature] = {
                    'importance': importance,
                    'value': float(feature_values[feature])
                }
        
        return explanation

if __name__ == "__main__":
    # Example usage
    model = CareerRecommendationModel()
    
    # Mock training data for demonstration
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.rand(1000, 20), columns=[f'feature_{i}' for i in range(20)])
    y_train = pd.Series(np.random.choice(['software_engineer', 'data_scientist', 'product_manager'], 1000))
    
    # Train model
    metrics = model.train(X_train, y_train)
    print(f"Training completed with accuracy: {metrics['train_accuracy']:.4f}")
    
    # Save model
    model.save_model('models/trained_models/career_classifier.pkl')
