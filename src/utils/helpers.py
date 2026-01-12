"""Utility functions for financial planning assistant."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass


def calculate_financial_ratios(
    income: float,
    expenses: float,
    savings: float,
    debt: float = 0
) -> Dict[str, float]:
    """Calculate key financial ratios.
    
    Args:
        income: Monthly income
        expenses: Monthly expenses
        savings: Monthly savings
        debt: Total debt amount
        
        Returns:
            Dictionary of financial ratios
    """
    ratios = {}
    
    # Savings rate
    ratios['savings_rate'] = savings / income if income > 0 else 0
    
    # Expense ratio
    ratios['expense_ratio'] = expenses / income if income > 0 else 0
    
    # Debt-to-income ratio
    ratios['debt_to_income'] = debt / income if income > 0 else 0
    
    # Emergency fund ratio (assuming 6 months expenses)
    emergency_target = expenses * 6
    ratios['emergency_fund_ratio'] = savings / emergency_target if emergency_target > 0 else 0
    
    return ratios


def format_currency(amount: float, currency: str = "USD") -> str:
    """Format currency amount for display.
    
    Args:
        amount: Amount to format
        currency: Currency code
        
        Returns:
            Formatted currency string
    """
    if currency == "USD":
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"


def calculate_time_to_goal(
    current_amount: float,
    target_amount: float,
    monthly_contribution: float,
    expected_return: float = 0.05
) -> Dict[str, Any]:
    """Calculate time to reach financial goal.
    
    Args:
        current_amount: Current amount saved
        target_amount: Target amount
        monthly_contribution: Monthly contribution
        expected_return: Expected annual return
        
        Returns:
            Dictionary with time calculations
    """
    if monthly_contribution <= 0:
        return {
            'months': float('inf'),
            'years': float('inf'),
            'achievable': False,
            'message': 'Monthly contribution must be positive'
        }
    
    # Calculate monthly return rate
    monthly_rate = expected_return / 12
    
    # Use future value of annuity formula
    remaining_amount = target_amount - current_amount
    
    if remaining_amount <= 0:
        return {
            'months': 0,
            'years': 0,
            'achievable': True,
            'message': 'Goal already achieved'
        }
    
    # Calculate months needed
    if monthly_rate == 0:
        months = remaining_amount / monthly_contribution
    else:
        months = np.log(1 + (remaining_amount * monthly_rate) / monthly_contribution) / np.log(1 + monthly_rate)
    
    years = months / 12
    
    return {
        'months': months,
        'years': years,
        'achievable': months < 1000,  # Reasonable upper limit
        'message': f'Goal achievable in {years:.1f} years' if months < 1000 else 'Goal may not be achievable'
    }


def calculate_compound_interest(
    principal: float,
    monthly_contribution: float,
    annual_rate: float,
    years: int
) -> Dict[str, float]:
    """Calculate compound interest with monthly contributions.
    
    Args:
        principal: Initial amount
        monthly_contribution: Monthly contribution
        annual_rate: Annual interest rate
        years: Number of years
        
        Returns:
            Dictionary with calculation results
    """
    monthly_rate = annual_rate / 12
    months = years * 12
    
    # Future value of principal
    fv_principal = principal * (1 + monthly_rate) ** months
    
    # Future value of monthly contributions (annuity)
    if monthly_rate == 0:
        fv_contributions = monthly_contribution * months
    else:
        fv_contributions = monthly_contribution * (((1 + monthly_rate) ** months - 1) / monthly_rate)
    
    total_future_value = fv_principal + fv_contributions
    total_contributions = principal + (monthly_contribution * months)
    total_interest = total_future_value - total_contributions
    
    return {
        'future_value': total_future_value,
        'total_contributions': total_contributions,
        'total_interest': total_interest,
        'principal': principal,
        'monthly_contributions': monthly_contribution * months
    }


def detect_spending_patterns(transactions: pd.DataFrame) -> Dict[str, Any]:
    """Detect spending patterns in transaction data.
    
    Args:
        transactions: DataFrame with transaction data
        
        Returns:
            Dictionary with pattern analysis
    """
    patterns = {}
    
    # Group by category
    category_spending = transactions.groupby('category')['amount'].sum().abs()
    patterns['top_categories'] = category_spending.nlargest(5).to_dict()
    
    # Monthly spending trends
    transactions['month'] = pd.to_datetime(transactions['date']).dt.to_period('M')
    monthly_spending = transactions.groupby('month')['amount'].sum().abs()
    patterns['monthly_trend'] = monthly_spending.to_dict()
    
    # Spending volatility
    patterns['spending_volatility'] = monthly_spending.std() / monthly_spending.mean()
    
    # Weekend vs weekday spending
    transactions['is_weekend'] = pd.to_datetime(transactions['date']).dt.dayofweek >= 5
    weekend_spending = transactions[transactions['is_weekend']]['amount'].sum()
    weekday_spending = transactions[~transactions['is_weekend']]['amount'].sum()
    patterns['weekend_vs_weekday'] = {
        'weekend': abs(weekend_spending),
        'weekday': abs(weekday_spending)
    }
    
    return patterns


def validate_financial_data(data: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate financial data format and content.
    
    Args:
        data: DataFrame to validate
        required_columns: List of required column names
        
        Returns:
            True if valid, False otherwise
    """
    # Check required columns
    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    # Check for negative values in amount columns
    if 'amount' in data.columns:
        negative_amounts = data[data['amount'] < 0].shape[0]
        if negative_amounts > 0:
            logger.warning(f"Found {negative_amounts} negative amounts")
    
    # Check for missing values
    missing_values = data.isnull().sum()
    if missing_values.any():
        logger.warning(f"Missing values found: {missing_values[missing_values > 0].to_dict()}")
    
    return True


def create_financial_summary(
    income_data: pd.DataFrame,
    expense_data: pd.DataFrame,
    goal_data: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """Create comprehensive financial summary.
    
    Args:
        income_data: Income data
        expense_data: Expense data
        goal_data: Optional goal data
        
        Returns:
            Dictionary with financial summary
    """
    summary = {}
    
    # Income summary
    summary['total_income'] = income_data['amount'].sum()
    summary['avg_monthly_income'] = income_data['amount'].mean()
    summary['income_sources'] = income_data['source'].nunique()
    
    # Expense summary
    summary['total_expenses'] = expense_data['amount'].sum()
    summary['avg_monthly_expenses'] = expense_data['amount'].mean()
    summary['expense_categories'] = expense_data['category'].nunique()
    
    # Savings summary
    summary['net_savings'] = summary['total_income'] - summary['total_expenses']
    summary['savings_rate'] = summary['net_savings'] / summary['total_income'] if summary['total_income'] > 0 else 0
    
    # Goal summary
    if goal_data is not None:
        summary['total_goals'] = len(goal_data)
        summary['total_goal_amount'] = goal_data['target_amount'].sum()
        summary['current_goal_progress'] = goal_data['current_amount'].sum()
        summary['goal_completion_rate'] = summary['current_goal_progress'] / summary['total_goal_amount'] if summary['total_goal_amount'] > 0 else 0
    
    return summary
