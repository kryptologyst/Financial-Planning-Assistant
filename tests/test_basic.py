"""Basic tests for financial planning assistant."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.data.processor import DataProcessor, FinancialData
from src.features.engineering import FeatureEngineer
from src.models.planners import BudgetOptimizer, GoalTracker, RiskAssessor
from src.utils.helpers import (
    calculate_financial_ratios,
    format_currency,
    calculate_time_to_goal,
    validate_financial_data
)


class TestDataProcessor:
    """Test data processor functionality."""
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        processor = DataProcessor()
        data = processor.generate_synthetic_data(n_months=6, n_transactions=100)
        
        assert isinstance(data, FinancialData)
        assert len(data.expenses) > 0
        assert len(data.income) > 0
        assert len(data.goals) > 0
        assert len(data.transactions) > 0
    
    def test_monthly_summary_calculation(self):
        """Test monthly summary calculation."""
        processor = DataProcessor()
        data = processor.generate_synthetic_data(n_months=3, n_transactions=50)
        summary = processor.calculate_monthly_summary(data)
        
        assert len(summary) == 3
        assert 'total_expenses' in summary.columns
        assert 'total_income' in summary.columns
        assert 'savings_rate' in summary.columns


class TestFeatureEngineer:
    """Test feature engineering functionality."""
    
    def test_budget_features(self):
        """Test budget feature creation."""
        engineer = FeatureEngineer()
        
        # Create sample data
        data = pd.DataFrame({
            'total_expenses': [1000, 1100, 1200],
            'total_income': [2000, 2100, 2200],
            'savings_rate': [0.5, 0.48, 0.45],
            'month': ['2023-01', '2023-02', '2023-03']
        })
        
        features = engineer.create_budget_features(data)
        
        assert len(features) == 3
        assert 'expense_to_income_ratio' in features.columns
        assert 'savings_rate' in features.columns


class TestModels:
    """Test model functionality."""
    
    def test_budget_optimizer(self):
        """Test budget optimizer model."""
        model = BudgetOptimizer(algorithm="random_forest")
        
        # Create sample data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        y = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == 5
        assert model.is_fitted is True
    
    def test_goal_tracker(self):
        """Test goal tracker model."""
        model = GoalTracker(algorithm="linear_regression")
        
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        y = pd.Series([0.2, 0.4, 0.6, 0.8, 1.0])
        
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == 5
        assert model.is_fitted is True
    
    def test_risk_assessor(self):
        """Test risk assessor model."""
        model = RiskAssessor(algorithm="random_forest")
        
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        y = pd.Series([0, 1, 2, 0, 1])
        
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == 5
        assert model.is_fitted is True


class TestUtils:
    """Test utility functions."""
    
    def test_calculate_financial_ratios(self):
        """Test financial ratio calculations."""
        ratios = calculate_financial_ratios(
            income=5000,
            expenses=3000,
            savings=2000,
            debt=10000
        )
        
        assert ratios['savings_rate'] == 0.4
        assert ratios['expense_ratio'] == 0.6
        assert ratios['debt_to_income'] == 2.0
    
    def test_format_currency(self):
        """Test currency formatting."""
        formatted = format_currency(1234.56)
        assert formatted == "$1,234.56"
    
    def test_calculate_time_to_goal(self):
        """Test time to goal calculation."""
        result = calculate_time_to_goal(
            current_amount=1000,
            target_amount=10000,
            monthly_contribution=500,
            expected_return=0.05
        )
        
        assert result['achievable'] is True
        assert result['months'] > 0
        assert result['years'] > 0
    
    def test_validate_financial_data(self):
        """Test financial data validation."""
        data = pd.DataFrame({
            'amount': [100, 200, 300],
            'category': ['food', 'transport', 'entertainment'],
            'date': [datetime.now(), datetime.now(), datetime.now()]
        })
        
        required_columns = ['amount', 'category', 'date']
        is_valid = validate_financial_data(data, required_columns)
        
        assert is_valid is True


if __name__ == "__main__":
    pytest.main([__file__])
