from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import pandas as pd
import numpy as np
import pickle
import json
from typing import Dict, List, Tuple, Optional
import logging

class SalaryPredictionModel:
    """Predicts salary range based on profile features"""
    
    def __init__(self, model_params: Dict = None):
        # Default parameters optimized for salary prediction
        default_params = {
            'n_estimators': 100,
            'random_state': 42,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt'
        }
        
        if model_params:
            default_params.update(model_params)
        
        self.model = RandomForestRegressor(**default_params)
        self.model_params = default_params
        self.is_trained = False
        self.feature_importance = None
        self.performance_metrics = {}
        self.salary_stats = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict:
        """Train the salary prediction model"""
        self.logger.info("Starting salary prediction model training...")
        
        # Store salary statistics for later use
        self.salary_stats = {
            'mean': float(y_train.mean()),
            'median': float(y_train.median()),
            'std': float(y_train.std()),
            'min': float(y_train.min()),
            'max': float(y_train.max()),
            'q25': float(y_train.quantile(0.25)),
            'q75': float(y_train.quantile(0.75))
        }
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate feature importance
        self.feature_importance = dict(zip(X_train.columns, self.model.feature_importances_))
        
        # Evaluate on training set
        train_pred = self.model.predict(X_train)
        train_mse = mean_squared_error(y_train, train_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)
        train_rmse = np.sqrt(train_mse)
        
        metrics = {
            'train_mse': train_mse,
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'feature_count': len(X_train.columns),
            'samples_count': len(X_train)
        }
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_mse = mean_squared_error(y_val, val_pred)
            val_mae = mean_absolute_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)
            val_rmse = np.sqrt(val_mse)
            
            metrics.update({
                'val_mse': val_mse,
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'val_r2': val_r2
            })
            
            # Percentage error analysis
            percentage_errors = np.abs(val_pred - y_val) / y_val * 100
            metrics['val_mape'] = float(np.mean(percentage_errors))  # Mean Absolute Percentage Error
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='r2')
        metrics['cv_mean_r2'] = cv_scores.mean()
        metrics['cv_std_r2'] = cv_scores.std()
        
        self.performance_metrics = metrics
        
        self.logger.info(f"Training complete. Train R²: {train_r2:.4f}, Train RMSE: {train_rmse:.2f}")
        if 'val_r2' in metrics:
            self.logger.info(f"Validation R²: {metrics['val_r2']:.4f}, Validation RMSE: {metrics['val_rmse']:.2f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict salary values"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_salary_range(self, X: pd.DataFrame, 
                           confidence_interval: float = 0.8) -> List[Dict]:
        """Predict salary with confidence intervals"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get base predictions
        predictions = self.predict(X)
        
        # Estimate prediction intervals using quantile regression approach
        # For simplicity, use standard deviation from training
        std_factor = 1.28 if confidence_interval == 0.8 else 1.96  # 80% or 95% CI
        
        results = []
        for pred in predictions:
            # Estimate uncertainty based on model's training performance
            uncertainty = self.performance_metrics.get('val_rmse', self.performance_metrics.get('train_rmse', pred * 0.2))
            
            lower_bound = max(0, pred - (std_factor * uncertainty * 0.5))
            upper_bound = pred + (std_factor * uncertainty * 0.5)
            
            # Categorize salary level
            salary_level = self._categorize_salary(pred)
            
            results.append({
                'predicted_salary': float(pred),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'confidence_interval': confidence_interval,
                'salary_level': salary_level,
                'formatted_range': f"₹{lower_bound:,.0f} - ₹{upper_bound:,.0f}"
            })
        
        return results
    
    def _categorize_salary(self, salary: float) -> str:
        """Categorize salary into levels based on market standards"""
        if salary < 500000:
            return "Entry Level"
        elif salary < 1000000:
            return "Mid Level"
        elif salary < 2000000:
            return "Senior Level"
        elif salary < 3500000:
            return "Lead/Principal Level"
        else:
            return "Executive Level"
    
    def get_salary_insights(self, X: pd.DataFrame, profile_info: Dict = None) -> Dict:
        """Get comprehensive salary insights for a profile"""
        if not self.is_trained:
            raise ValueError("Model must be trained before analysis")
        
        # Get salary predictions
        salary_predictions = self.predict_salary_range(X, confidence_interval=0.8)
        
        if not salary_predictions:
            return {}
        
        prediction = salary_predictions[0]  # Assuming single profile
        predicted_salary = prediction['predicted_salary']
        
        # Compare with market statistics
        percentile_position = self._calculate_percentile_position(predicted_salary)
        
        # Identify top contributing factors
        top_factors = self.get_feature_importance(top_n=5)
        
        insights = {
            'predicted_salary': predicted_salary,
            'salary_range': prediction['formatted_range'],
            'salary_level': prediction['salary_level'],
            'market_percentile': percentile_position,
            'comparison_to_median': (predicted_salary / self.salary_stats['median'] - 1) * 100,
            'top_salary_factors': top_factors,
            'salary_growth_potential': self._estimate_growth_potential(predicted_salary, profile_info)
        }
        
        return insights
    
    def _calculate_percentile_position(self, salary: float) -> float:
        """Calculate what percentile this salary represents"""
        # Rough estimation based on salary distribution
        if salary <= self.salary_stats['q25']:
            return 25.0
        elif salary <= self.salary_stats['median']:
            return 50.0
        elif salary <= self.salary_stats['q75']:
            return 75.0
        else:
            # Estimate higher percentiles
            if salary <= self.salary_stats['mean'] + self.salary_stats['std']:
                return 85.0
            elif salary <= self.salary_stats['mean'] + 2 * self.salary_stats['std']:
                return 95.0
            else:
                return 99.0
    
    def _estimate_growth_potential(self, current_salary: float, profile_info: Dict = None) -> Dict:
        """Estimate salary growth potential"""
        # Simple growth estimation based on current salary level
        if current_salary < 500000:
            potential_growth = {
                '1_year': current_salary * 1.15,
                '3_years': current_salary * 1.4,
                '5_years': current_salary * 1.8
            }
        elif current_salary < 1000000:
            potential_growth = {
                '1_year': current_salary * 1.12,
                '3_years': current_salary * 1.35,
                '5_years': current_salary * 1.6
            }
        else:
            potential_growth = {
                '1_year': current_salary * 1.08,
                '3_years': current_salary * 1.25,
                '5_years': current_salary * 1.4
            }
        
        return {
            'growth_rates': {
                '1_year': f"₹{potential_growth['1_year']:,.0f} (+{((potential_growth['1_year']/current_salary - 1)*100):.1f}%)",
                '3_years': f"₹{potential_growth['3_years']:,.0f} (+{((potential_growth['3_years']/current_salary - 1)*100):.1f}%)",
                '5_years': f"₹{potential_growth['5_years']:,.0f} (+{((potential_growth['5_years']/current_salary - 1)*100):.1f}%)"
            },
            'growth_factors': ['Skill development', 'Experience gain', 'Role progression', 'Industry growth']
        }
    
    def get_feature_importance(self, top_n: int = 20) -> Dict:
        """Get top-n most important features for salary prediction"""
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
        """Perform hyperparameter tuning"""
        self.logger.info("Starting hyperparameter tuning...")
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            cv=5,
            scoring='r2',
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
        
        self.logger.info(f"Hyperparameter tuning complete. Best R² score: {grid_search.best_score_:.4f}")
        return tuning_results
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Comprehensive model evaluation"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate various metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Percentage errors
        percentage_errors = np.abs(y_pred - y_test) / y_test * 100
        mape = np.mean(percentage_errors)
        
        # Prediction accuracy within ranges
        within_10_percent = np.sum(percentage_errors <= 10) / len(percentage_errors) * 100
        within_20_percent = np.sum(percentage_errors <= 20) / len(percentage_errors) * 100
        
        evaluation_results = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2,
            'mape': mape,
            'within_10_percent_accuracy': within_10_percent,
            'within_20_percent_accuracy': within_20_percent,
            'prediction_stats': {
                'mean_prediction': float(np.mean(y_pred)),
                'mean_actual': float(np.mean(y_test)),
                'prediction_std': float(np.std(y_pred)),
                'actual_std': float(np.std(y_test))
            }
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
            'salary_stats': self.salary_stats,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Salary prediction model saved to {filepath}")
        
        # Save metadata
        if include_metadata:
            metadata_filepath = filepath.replace('.pkl', '_metadata.json')
            metadata = {
                'model_type': 'SalaryPredictionModel',
                'model_params': self.model_params,
                'performance_metrics': self.performance_metrics,
                'salary_stats': self.salary_stats,
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
        self.salary_stats = model_data['salary_stats']
        self.is_trained = model_data['is_trained']
        
        self.logger.info(f"Salary prediction model loaded from {filepath}")

if __name__ == "__main__":
    # Example usage
    model = SalaryPredictionModel()
    
    # Mock training data
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.rand(1000, 15), columns=[f'feature_{i}' for i in range(15)])
    y_train = pd.Series(np.random.normal(800000, 300000, 1000))  # Mock salaries
    
    # Train model
    metrics = model.train(X_train, y_train)
    print(f"Training completed with R² score: {metrics['train_r2']:.4f}")
    
    # Save model
    model.save_model('models/trained_models/salary_estimator.pkl')
