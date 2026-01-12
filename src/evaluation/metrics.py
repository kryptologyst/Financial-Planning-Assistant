"""Evaluation metrics and backtesting for financial planning models."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

logger = logging.getLogger(__name__)


class FinancialMetrics:
    """Calculate financial-specific evaluation metrics."""
    
    @staticmethod
    def calculate_savings_rate_accuracy(
        actual: pd.Series, 
        predicted: pd.Series
    ) -> Dict[str, float]:
        """Calculate savings rate prediction accuracy.
        
        Args:
            actual: Actual savings rates
            predicted: Predicted savings rates
            
        Returns:
            Dictionary of accuracy metrics
        """
        mse = mean_squared_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        
        # Financial-specific metrics
        savings_rate_error = np.abs(actual - predicted).mean()
        target_achievement = (predicted >= 0.2).mean()  # 20% savings target
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'savings_rate_error': savings_rate_error,
            'target_achievement_rate': target_achievement
        }
    
    @staticmethod
    def calculate_goal_progress_metrics(
        actual_progress: pd.Series,
        predicted_progress: pd.Series,
        target_dates: pd.Series
    ) -> Dict[str, float]:
        """Calculate goal progress prediction metrics.
        
        Args:
            actual_progress: Actual progress percentages
            predicted_progress: Predicted progress percentages
            target_dates: Target completion dates
            
        Returns:
            Dictionary of progress metrics
        """
        progress_error = np.abs(actual_progress - predicted_progress).mean()
        
        # Time-based metrics
        current_date = datetime.now()
        days_remaining = (target_dates - current_date).dt.days
        predicted_completion_rate = (predicted_progress >= 1.0).mean()
        
        return {
            'progress_error': progress_error,
            'completion_rate': predicted_completion_rate,
            'avg_days_remaining': days_remaining.mean()
        }
    
    @staticmethod
    def calculate_risk_assessment_metrics(
        actual_risk: pd.Series,
        predicted_risk: pd.Series
    ) -> Dict[str, float]:
        """Calculate risk assessment accuracy metrics.
        
        Args:
            actual_risk: Actual risk tolerance levels
            predicted_risk: Predicted risk tolerance levels
            
        Returns:
            Dictionary of risk assessment metrics
        """
        accuracy = accuracy_score(actual_risk, predicted_risk)
        precision = precision_score(actual_risk, predicted_risk, average='weighted')
        recall = recall_score(actual_risk, predicted_risk, average='weighted')
        f1 = f1_score(actual_risk, predicted_risk, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }


class Backtester:
    """Backtesting framework for financial planning strategies."""
    
    def __init__(self, initial_capital: float = 10000) -> None:
        """Initialize backtester.
        
        Args:
            initial_capital: Starting capital amount
        """
        self.initial_capital = initial_capital
        self.results = None
        
    def run_backtest(
        self,
        monthly_data: pd.DataFrame,
        strategy_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run backtest on financial planning strategy.
        
        Args:
            monthly_data: Monthly financial data
            strategy_config: Strategy configuration
            
        Returns:
            Backtest results
        """
        logger.info("Running financial planning backtest")
        
        # Initialize tracking variables
        capital = self.initial_capital
        portfolio_history = []
        savings_history = []
        
        for _, month_data in monthly_data.iterrows():
            # Calculate monthly savings
            monthly_savings = month_data['net_savings']
            
            # Apply strategy rules
            if strategy_config.get('emergency_fund_first', True):
                emergency_target = month_data['total_expenses'] * 6
                if capital < emergency_target:
                    capital += monthly_savings
                else:
                    # Invest excess savings
                    excess = monthly_savings
                    capital += excess * strategy_config.get('investment_rate', 0.7)
            else:
                capital += monthly_savings
            
            portfolio_history.append(capital)
            savings_history.append(monthly_savings)
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics(
            portfolio_history, savings_history, monthly_data
        )
        
        self.results = results
        return results
    
    def _calculate_performance_metrics(
        self,
        portfolio_history: List[float],
        savings_history: List[float],
        monthly_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        
        portfolio_series = pd.Series(portfolio_history)
        savings_series = pd.Series(savings_history)
        
        # Basic metrics
        total_return = (portfolio_series.iloc[-1] - self.initial_capital) / self.initial_capital
        annualized_return = (1 + total_return) ** (12 / len(portfolio_series)) - 1
        
        # Volatility
        monthly_returns = portfolio_series.pct_change().dropna()
        volatility = monthly_returns.std() * np.sqrt(12)
        
        # Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        peak = portfolio_series.expanding().max()
        drawdown = (portfolio_series - peak) / peak
        max_drawdown = drawdown.min()
        
        # Savings consistency
        savings_consistency = savings_series.std() / savings_series.mean()
        
        # Goal achievement
        target_savings_rate = 0.2
        achieved_months = (monthly_data['savings_rate'] >= target_savings_rate).sum()
        goal_achievement_rate = achieved_months / len(monthly_data)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'savings_consistency': savings_consistency,
            'goal_achievement_rate': goal_achievement_rate,
            'final_portfolio_value': portfolio_series.iloc[-1],
            'portfolio_history': portfolio_history,
            'savings_history': savings_history
        }


class ModelEvaluator:
    """Comprehensive model evaluation framework."""
    
    def __init__(self) -> None:
        """Initialize model evaluator."""
        self.metrics_calculator = FinancialMetrics()
        self.backtester = Backtester()
        
    def evaluate_budget_optimizer(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """Evaluate budget optimizer model.
        
        Args:
            model: Trained budget optimizer model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Evaluation results
        """
        logger.info("Evaluating budget optimizer")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate standard metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate financial-specific metrics
        financial_metrics = self.metrics_calculator.calculate_savings_rate_accuracy(
            y_test, y_pred
        )
        
        # Feature importance
        feature_importance = None
        if hasattr(model, 'feature_importance_') and model.feature_importance_ is not None:
            feature_importance = model.feature_importance_.to_dict()
        
        return {
            'model_name': model.model_name,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'financial_metrics': financial_metrics,
            'feature_importance': feature_importance,
            'predictions': y_pred,
            'actual': y_test
        }
    
    def evaluate_goal_tracker(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        goal_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Evaluate goal tracker model.
        
        Args:
            model: Trained goal tracker model
            X_test: Test features
            y_test: Test targets
            goal_data: Goal data for context
            
        Returns:
            Evaluation results
        """
        logger.info("Evaluating goal tracker")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate standard metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate goal-specific metrics
        goal_metrics = self.metrics_calculator.calculate_goal_progress_metrics(
            y_test, y_pred, goal_data['target_date']
        )
        
        return {
            'model_name': model.model_name,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'goal_metrics': goal_metrics,
            'predictions': y_pred,
            'actual': y_test
        }
    
    def evaluate_risk_assessor(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """Evaluate risk assessor model.
        
        Args:
            model: Trained risk assessor model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Evaluation results
        """
        logger.info("Evaluating risk assessor")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Calculate risk-specific metrics
        risk_metrics = self.metrics_calculator.calculate_risk_assessment_metrics(
            y_test, y_pred
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Feature importance
        feature_importance = None
        if hasattr(model, 'feature_importance_') and model.feature_importance_ is not None:
            feature_importance = model.feature_importance_.to_dict()
        
        return {
            'model_name': model.model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'risk_metrics': risk_metrics,
            'confusion_matrix': cm,
            'feature_importance': feature_importance,
            'predictions': y_pred,
            'actual': y_test
        }
    
    def run_comprehensive_evaluation(
        self,
        models: Dict[str, Any],
        test_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation of all models.
        
        Args:
            models: Dictionary of trained models
            test_data: Dictionary of test datasets
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info("Running comprehensive model evaluation")
        
        results = {}
        
        # Evaluate each model type
        if 'budget_optimizer' in models and 'budget_data' in test_data:
            results['budget_optimizer'] = self.evaluate_budget_optimizer(
                models['budget_optimizer'],
                test_data['budget_data']['X_test'],
                test_data['budget_data']['y_test']
            )
        
        if 'goal_tracker' in models and 'goal_data' in test_data:
            results['goal_tracker'] = self.evaluate_goal_tracker(
                models['goal_tracker'],
                test_data['goal_data']['X_test'],
                test_data['goal_data']['y_test'],
                test_data['goal_data']['goal_info']
            )
        
        if 'risk_assessor' in models and 'risk_data' in test_data:
            results['risk_assessor'] = self.evaluate_risk_assessor(
                models['risk_assessor'],
                test_data['risk_data']['X_test'],
                test_data['risk_data']['y_test']
            )
        
        # Generate summary report
        results['summary'] = self._generate_summary_report(results)
        
        return results
    
    def _generate_summary_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary report of all evaluations."""
        summary = {
            'total_models_evaluated': len(results) - 1,  # Exclude summary itself
            'model_performance': {}
        }
        
        for model_name, model_results in results.items():
            if model_name == 'summary':
                continue
                
            # Extract key metrics based on model type
            if 'r2' in model_results:
                summary['model_performance'][model_name] = {
                    'r2_score': model_results['r2'],
                    'mae': model_results['mae']
                }
            elif 'accuracy' in model_results:
                summary['model_performance'][model_name] = {
                    'accuracy': model_results['accuracy'],
                    'f1_score': model_results['f1_score']
                }
        
        return summary
    
    def save_evaluation_results(
        self,
        results: Dict[str, Any],
        output_dir: str
    ) -> None:
        """Save evaluation results to files.
        
        Args:
            results: Evaluation results
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save summary report
        summary_df = pd.DataFrame(results['summary']['model_performance']).T
        summary_df.to_csv(output_path / "evaluation_summary.csv")
        
        # Save detailed results for each model
        for model_name, model_results in results.items():
            if model_name == 'summary':
                continue
                
            # Save predictions
            if 'predictions' in model_results:
                pred_df = pd.DataFrame({
                    'actual': model_results['actual'],
                    'predicted': model_results['predictions']
                })
                pred_df.to_csv(output_path / f"{model_name}_predictions.csv", index=False)
        
        logger.info(f"Evaluation results saved to {output_dir}")
    
    def create_evaluation_plots(
        self,
        results: Dict[str, Any],
        output_dir: str
    ) -> None:
        """Create evaluation visualization plots.
        
        Args:
            results: Evaluation results
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Create plots for each model
        for model_name, model_results in results.items():
            if model_name == 'summary' or 'predictions' not in model_results:
                continue
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'{model_name.replace("_", " ").title()} Evaluation', fontsize=16)
            
            actual = model_results['actual']
            predicted = model_results['predictions']
            
            # Scatter plot
            axes[0, 0].scatter(actual, predicted, alpha=0.6)
            axes[0, 0].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
            axes[0, 0].set_xlabel('Actual')
            axes[0, 0].set_ylabel('Predicted')
            axes[0, 0].set_title('Actual vs Predicted')
            
            # Residuals plot
            residuals = actual - predicted
            axes[0, 1].scatter(predicted, residuals, alpha=0.6)
            axes[0, 1].axhline(y=0, color='r', linestyle='--')
            axes[0, 1].set_xlabel('Predicted')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].set_title('Residuals Plot')
            
            # Distribution comparison
            axes[1, 0].hist(actual, alpha=0.5, label='Actual', bins=20)
            axes[1, 0].hist(predicted, alpha=0.5, label='Predicted', bins=20)
            axes[1, 0].set_xlabel('Value')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Distribution Comparison')
            axes[1, 0].legend()
            
            # Feature importance (if available)
            if 'feature_importance' in model_results and model_results['feature_importance']:
                importance = model_results['feature_importance']
                top_features = dict(list(importance.items())[:10])
                
                features = list(top_features.keys())
                importances = list(top_features.values())
                
                axes[1, 1].barh(features, importances)
                axes[1, 1].set_xlabel('Importance')
                axes[1, 1].set_title('Top 10 Feature Importance')
            else:
                axes[1, 1].text(0.5, 0.5, 'No feature importance available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Feature Importance')
            
            plt.tight_layout()
            plt.savefig(output_path / f"{model_name}_evaluation.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Evaluation plots saved to {output_dir}")
