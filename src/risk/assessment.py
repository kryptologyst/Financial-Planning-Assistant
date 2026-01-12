"""Risk management and assessment module for financial planning assistant."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """Risk management and assessment for financial planning."""
    
    def __init__(self) -> None:
        """Initialize risk manager."""
        self.risk_tolerance_levels = ['Conservative', 'Moderate', 'Aggressive']
        self.risk_thresholds = {
            'Conservative': 0.05,
            'Moderate': 0.10,
            'Aggressive': 0.15
        }
    
    def assess_financial_stability(
        self, 
        monthly_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Assess overall financial stability.
        
        Args:
            monthly_data: Monthly financial data
            
        Returns:
            Dictionary with stability assessment
        """
        # Calculate stability metrics
        income_stability = self._calculate_income_stability(monthly_data)
        expense_predictability = self._calculate_expense_predictability(monthly_data)
        savings_consistency = self._calculate_savings_consistency(monthly_data)
        emergency_fund_adequacy = self._calculate_emergency_fund_adequacy(monthly_data)
        
        # Overall stability score
        stability_score = np.mean([
            income_stability,
            expense_predictability,
            savings_consistency,
            emergency_fund_adequacy
        ])
        
        # Risk level determination
        if stability_score >= 0.8:
            risk_level = 'Low'
        elif stability_score >= 0.6:
            risk_level = 'Medium'
        else:
            risk_level = 'High'
        
        return {
            'stability_score': stability_score,
            'risk_level': risk_level,
            'income_stability': income_stability,
            'expense_predictability': expense_predictability,
            'savings_consistency': savings_consistency,
            'emergency_fund_adequacy': emergency_fund_adequacy
        }
    
    def _calculate_income_stability(self, monthly_data: pd.DataFrame) -> float:
        """Calculate income stability score."""
        income_cv = monthly_data['total_income'].std() / monthly_data['total_income'].mean()
        return max(0, 1 - income_cv)
    
    def _calculate_expense_predictability(self, monthly_data: pd.DataFrame) -> float:
        """Calculate expense predictability score."""
        expense_cv = monthly_data['total_expenses'].std() / monthly_data['total_expenses'].mean()
        return max(0, 1 - expense_cv)
    
    def _calculate_savings_consistency(self, monthly_data: pd.DataFrame) -> float:
        """Calculate savings consistency score."""
        savings_rate_cv = monthly_data['savings_rate'].std() / monthly_data['savings_rate'].mean()
        return max(0, 1 - savings_rate_cv)
    
    def _calculate_emergency_fund_adequacy(self, monthly_data: pd.DataFrame) -> float:
        """Calculate emergency fund adequacy score."""
        avg_expenses = monthly_data['total_expenses'].mean()
        total_savings = monthly_data['net_savings'].sum()
        emergency_months = total_savings / avg_expenses
        
        # Target: 6 months of expenses
        target_months = 6
        adequacy = min(1.0, emergency_months / target_months)
        
        return adequacy
    
    def calculate_var(
        self, 
        returns: pd.Series, 
        confidence_level: float = 0.05
    ) -> float:
        """Calculate Value at Risk (VaR).
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
            
        Returns:
            VaR value
        """
        return returns.quantile(confidence_level)
    
    def calculate_expected_shortfall(
        self, 
        returns: pd.Series, 
        confidence_level: float = 0.05
    ) -> float:
        """Calculate Expected Shortfall (Conditional VaR).
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level
            
        Returns:
            Expected shortfall value
        """
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def stress_test_scenarios(
        self, 
        monthly_data: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Run stress test scenarios.
        
        Args:
            monthly_data: Monthly financial data
            
        Returns:
            Dictionary with stress test results
        """
        scenarios = {}
        
        # Base case
        base_income = monthly_data['total_income'].mean()
        base_expenses = monthly_data['total_expenses'].mean()
        base_savings = base_income - base_expenses
        
        scenarios['base_case'] = {
            'income': base_income,
            'expenses': base_expenses,
            'savings': base_savings,
            'savings_rate': base_savings / base_income
        }
        
        # Income reduction scenarios
        for reduction in [0.1, 0.2, 0.3]:
            scenario_name = f'income_reduction_{int(reduction*100)}%'
            reduced_income = base_income * (1 - reduction)
            scenario_savings = reduced_income - base_expenses
            
            scenarios[scenario_name] = {
                'income': reduced_income,
                'expenses': base_expenses,
                'savings': scenario_savings,
                'savings_rate': scenario_savings / reduced_income if reduced_income > 0 else 0
            }
        
        # Expense increase scenarios
        for increase in [0.1, 0.2, 0.3]:
            scenario_name = f'expense_increase_{int(increase*100)}%'
            increased_expenses = base_expenses * (1 + increase)
            scenario_savings = base_income - increased_expenses
            
            scenarios[scenario_name] = {
                'income': base_income,
                'expenses': increased_expenses,
                'savings': scenario_savings,
                'savings_rate': scenario_savings / base_income
            }
        
        return scenarios
    
    def generate_risk_recommendations(
        self, 
        stability_assessment: Dict[str, Any]
    ) -> List[str]:
        """Generate risk management recommendations.
        
        Args:
            stability_assessment: Results from assess_financial_stability
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if stability_assessment['income_stability'] < 0.7:
            recommendations.append(
                "Consider diversifying income sources to improve stability"
            )
        
        if stability_assessment['expense_predictability'] < 0.7:
            recommendations.append(
                "Create detailed budgets and track expenses more closely"
            )
        
        if stability_assessment['savings_consistency'] < 0.7:
            recommendations.append(
                "Set up automatic savings transfers for consistency"
            )
        
        if stability_assessment['emergency_fund_adequacy'] < 0.8:
            recommendations.append(
                "Build emergency fund to cover 6 months of expenses"
            )
        
        if stability_assessment['risk_level'] == 'High':
            recommendations.append(
                "Consider reducing discretionary spending and increasing savings"
            )
        
        return recommendations


class PortfolioRiskAnalyzer:
    """Portfolio risk analysis for investment recommendations."""
    
    def __init__(self) -> None:
        """Initialize portfolio risk analyzer."""
        self.asset_classes = {
            'stocks': {'expected_return': 0.08, 'volatility': 0.15},
            'bonds': {'expected_return': 0.03, 'volatility': 0.05},
            'cash': {'expected_return': 0.02, 'volatility': 0.01},
            'real_estate': {'expected_return': 0.06, 'volatility': 0.12}
        }
    
    def calculate_portfolio_risk(
        self, 
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate portfolio risk metrics.
        
        Args:
            weights: Dictionary of asset class weights
            
        Returns:
            Dictionary with risk metrics
        """
        # Calculate expected return
        expected_return = sum(
            weights.get(asset, 0) * self.asset_classes[asset]['expected_return']
            for asset in self.asset_classes
        )
        
        # Calculate portfolio volatility (simplified)
        portfolio_variance = sum(
            weights.get(asset, 0) ** 2 * self.asset_classes[asset]['volatility'] ** 2
            for asset in self.asset_classes
        )
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Calculate Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        sharpe_ratio = (expected_return - risk_free_rate) / portfolio_volatility
        
        return {
            'expected_return': expected_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
        }
    
    def recommend_portfolio_allocation(
        self, 
        risk_tolerance: str,
        age: int = 30,
        time_horizon: int = 30
    ) -> Dict[str, float]:
        """Recommend portfolio allocation based on risk tolerance.
        
        Args:
            risk_tolerance: Risk tolerance level
            age: Current age
            time_horizon: Investment time horizon in years
            
        Returns:
            Recommended asset allocation
        """
        # Base allocation by risk tolerance
        base_allocations = {
            'Conservative': {'stocks': 0.3, 'bonds': 0.5, 'cash': 0.2, 'real_estate': 0.0},
            'Moderate': {'stocks': 0.6, 'bonds': 0.3, 'cash': 0.1, 'real_estate': 0.0},
            'Aggressive': {'stocks': 0.8, 'bonds': 0.1, 'cash': 0.1, 'real_estate': 0.0}
        }
        
        allocation = base_allocations.get(risk_tolerance, base_allocations['Moderate'])
        
        # Adjust for age (older = more conservative)
        age_factor = max(0.5, 1 - (age - 25) / 100)
        
        # Adjust for time horizon (longer = more aggressive)
        horizon_factor = min(1.2, 1 + (time_horizon - 10) / 100)
        
        # Apply adjustments
        for asset in allocation:
            if asset == 'stocks':
                allocation[asset] *= age_factor * horizon_factor
            elif asset == 'bonds':
                allocation[asset] *= (2 - age_factor) * (2 - horizon_factor)
        
        # Normalize weights
        total_weight = sum(allocation.values())
        allocation = {k: v / total_weight for k, v in allocation.items()}
        
        return allocation
