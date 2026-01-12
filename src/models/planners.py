"""Machine learning models for financial planning assistant."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Base class for all financial planning models."""
    
    def __init__(self, model_name: str) -> None:
        """Initialize base model.
        
        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
        self.model = None
        self.is_fitted = False
        self.feature_importance_ = None
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        pass
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'feature_importance_': self.feature_importance_
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.feature_importance_ = model_data.get('feature_importance_')
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")


class BudgetOptimizer(BaseModel):
    """Model for optimizing budget allocation and expense prediction."""
    
    def __init__(self, algorithm: str = "xgboost") -> None:
        """Initialize budget optimizer.
        
        Args:
            algorithm: ML algorithm to use ('xgboost', 'lightgbm', 'random_forest')
        """
        super().__init__(f"budget_optimizer_{algorithm}")
        self.algorithm = algorithm
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the underlying model based on algorithm."""
        if self.algorithm == "xgboost":
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        elif self.algorithm == "lightgbm":
            self.model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
        elif self.algorithm == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the budget optimizer model.
        
        Args:
            X: Feature matrix
            y: Target variable (expense amounts or savings rates)
        """
        logger.info(f"Training {self.model_name}")
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = pd.Series(
                self.model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)
        
        logger.info("Budget optimizer training completed")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict budget-related metrics.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def predict_expense_recommendations(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict optimal expense recommendations.
        
        Args:
            X: Feature matrix
            
        Returns:
            DataFrame with expense recommendations by category
        """
        predictions = self.predict(X)
        
        # Create recommendations based on predictions
        recommendations = pd.DataFrame({
            'predicted_expense': predictions,
            'recommendation': self._generate_recommendations(predictions)
        })
        
        return recommendations
    
    def _generate_recommendations(self, predictions: np.ndarray) -> List[str]:
        """Generate human-readable recommendations."""
        recommendations = []
        
        for pred in predictions:
            if pred > 0.8:
                recommendations.append("Reduce expenses significantly")
            elif pred > 0.6:
                recommendations.append("Consider reducing expenses")
            elif pred > 0.4:
                recommendations.append("Maintain current spending")
            else:
                recommendations.append("You can increase spending")
        
        return recommendations


class GoalTracker(BaseModel):
    """Model for tracking progress toward financial goals."""
    
    def __init__(self, algorithm: str = "linear_regression") -> None:
        """Initialize goal tracker.
        
        Args:
            algorithm: ML algorithm to use
        """
        super().__init__(f"goal_tracker_{algorithm}")
        self.algorithm = algorithm
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the underlying model."""
        if self.algorithm == "linear_regression":
            self.model = LinearRegression()
        elif self.algorithm == "ridge":
            self.model = Ridge(alpha=1.0)
        elif self.algorithm == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the goal tracker model.
        
        Args:
            X: Feature matrix
            y: Target variable (goal progress or time to completion)
        """
        logger.info(f"Training {self.model_name}")
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        logger.info("Goal tracker training completed")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict goal-related metrics.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def predict_goal_completion_time(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict time to complete financial goals.
        
        Args:
            X: Feature matrix
            
        Returns:
            DataFrame with completion time predictions
        """
        predictions = self.predict(X)
        
        results = pd.DataFrame({
            'predicted_months': predictions,
            'confidence': self._calculate_confidence(predictions),
            'recommendation': self._generate_goal_recommendations(predictions)
        })
        
        return results
    
    def _calculate_confidence(self, predictions: np.ndarray) -> List[str]:
        """Calculate confidence levels for predictions."""
        confidence_levels = []
        
        for pred in predictions:
            if pred < 6:
                confidence_levels.append("High")
            elif pred < 12:
                confidence_levels.append("Medium")
            else:
                confidence_levels.append("Low")
        
        return confidence_levels
    
    def _generate_goal_recommendations(self, predictions: np.ndarray) -> List[str]:
        """Generate goal-specific recommendations."""
        recommendations = []
        
        for pred in predictions:
            if pred < 6:
                recommendations.append("Goal achievable with current savings rate")
            elif pred < 12:
                recommendations.append("Consider increasing monthly contributions")
            else:
                recommendations.append("Goal may need adjustment or higher savings rate")
        
        return recommendations


class RiskAssessor(BaseModel):
    """Model for assessing financial risk tolerance and stability."""
    
    def __init__(self, algorithm: str = "random_forest") -> None:
        """Initialize risk assessor.
        
        Args:
            algorithm: ML algorithm to use
        """
        super().__init__(f"risk_assessor_{algorithm}")
        self.algorithm = algorithm
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the underlying model."""
        if self.algorithm == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42
            )
        elif self.algorithm == "logistic_regression":
            self.model = LogisticRegression(random_state=42)
        elif self.algorithm == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the risk assessor model.
        
        Args:
            X: Feature matrix
            y: Target variable (risk tolerance levels)
        """
        logger.info(f"Training {self.model_name}")
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = pd.Series(
                self.model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)
        
        logger.info("Risk assessor training completed")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict risk tolerance levels.
        
        Args:
            X: Feature matrix
            
        Returns:
            Risk tolerance predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def predict_risk_profile(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict comprehensive risk profile.
        
        Args:
            X: Feature matrix
            
        Returns:
            DataFrame with risk assessment results
        """
        predictions = self.predict(X)
        probabilities = self.model.predict_proba(X)
        
        risk_levels = ['Conservative', 'Moderate', 'Aggressive']
        
        results = pd.DataFrame({
            'risk_tolerance': [risk_levels[pred] for pred in predictions],
            'confidence': np.max(probabilities, axis=1),
            'recommendation': self._generate_risk_recommendations(predictions)
        })
        
        return results
    
    def _generate_risk_recommendations(self, predictions: np.ndarray) -> List[str]:
        """Generate risk-specific recommendations."""
        recommendations = []
        
        for pred in predictions:
            if pred == 0:  # Conservative
                recommendations.append("Focus on low-risk investments and emergency fund")
            elif pred == 1:  # Moderate
                recommendations.append("Balanced approach with mix of investments")
            else:  # Aggressive
                recommendations.append("Consider higher-risk, higher-reward investments")
        
        return recommendations


class ModelEnsemble:
    """Ensemble of financial planning models."""
    
    def __init__(self) -> None:
        """Initialize model ensemble."""
        self.models: Dict[str, BaseModel] = {}
        self.is_fitted = False
    
    def add_model(self, name: str, model: BaseModel) -> None:
        """Add a model to the ensemble.
        
        Args:
            name: Name of the model
            model: Model instance
        """
        self.models[name] = model
    
    def fit_all(self, X: pd.DataFrame, y_dict: Dict[str, pd.Series]) -> None:
        """Fit all models in the ensemble.
        
        Args:
            X: Feature matrix
            y_dict: Dictionary mapping model names to target variables
        """
        logger.info("Training model ensemble")
        
        for name, model in self.models.items():
            if name in y_dict:
                model.fit(X, y_dict[name])
        
        self.is_fitted = True
        logger.info("Model ensemble training completed")
    
    def predict_all(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make predictions with all models.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary mapping model names to predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        predictions = {}
        for name, model in self.models.items():
            if model.is_fitted:
                predictions[name] = model.predict(X)
        
        return predictions
    
    def save_ensemble(self, output_dir: str) -> None:
        """Save all models in the ensemble.
        
        Args:
            output_dir: Directory to save models
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            if model.is_fitted:
                model.save_model(output_path / f"{name}.joblib")
        
        logger.info(f"Ensemble saved to {output_dir}")
    
    def load_ensemble(self, model_dir: str) -> None:
        """Load all models in the ensemble.
        
        Args:
            model_dir: Directory containing saved models
        """
        model_path = Path(model_dir)
        
        for model_file in model_path.glob("*.joblib"):
            model_name = model_file.stem
            
            if model_name in self.models:
                self.models[model_name].load_model(str(model_file))
        
        self.is_fitted = True
        logger.info(f"Ensemble loaded from {model_dir}")
