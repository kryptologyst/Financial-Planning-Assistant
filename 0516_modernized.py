#!/usr/bin/env python3
"""Modernized Financial Planning Assistant - Research and Educational Tool

DISCLAIMER: This is for educational and research purposes only.
This software is NOT intended for investment advice or financial planning advice.
"""

import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.processor import DataProcessor
from src.features.engineering import FeatureEngineer
from src.models.planners import BudgetOptimizer, GoalTracker, RiskAssessor
from src.evaluation.metrics import ModelEvaluator, Backtester
from src.utils.helpers import set_random_seeds, create_financial_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function demonstrating the financial planning assistant."""
    
    print("="*60)
    print("FINANCIAL PLANNING ASSISTANT - RESEARCH DEMO")
    print("="*60)
    print()
    print("DISCLAIMER: This is for educational and research purposes only.")
    print("This software is NOT intended for investment advice.")
    print()
    
    # Set random seeds for reproducibility
    set_random_seeds(42)
    
    # Initialize components
    processor = DataProcessor()
    feature_engineer = FeatureEngineer()
    
    print("1. Generating synthetic financial data...")
    
    # Generate synthetic data
    financial_data = processor.generate_synthetic_data(n_months=12, n_transactions=1000)
    monthly_summary = processor.calculate_monthly_summary(financial_data)
    
    print(f"   Generated {len(monthly_summary)} months of data")
    print(f"   Generated {len(financial_data.transactions)} transactions")
    
    # Create financial summary
    summary = create_financial_summary(
        financial_data.income,
        financial_data.expenses,
        financial_data.goals
    )
    
    print("\n2. Financial Summary:")
    print(f"   Average Monthly Income: ${summary['avg_monthly_income']:,.0f}")
    print(f"   Average Monthly Expenses: ${summary['avg_monthly_expenses']:,.0f}")
    print(f"   Net Monthly Savings: ${summary['net_savings']:,.0f}")
    print(f"   Savings Rate: {summary['savings_rate']:.1%}")
    
    print("\n3. Creating features and training models...")
    
    # Create features
    features = feature_engineer.create_all_features(financial_data, monthly_summary)
    
    # Prepare targets - ensure they match feature length
    budget_target = monthly_summary['savings_rate']
    
    # For goal tracker, we need to match the number of goals
    goal_target = financial_data.goals['current_amount'] / financial_data.goals['target_amount']
    
    # For risk assessor, use monthly data
    risk_target = pd.Series(np.random.choice([0, 1, 2], size=len(monthly_summary)))
    
    # Ensure features and targets have compatible lengths
    min_length = min(len(features), len(budget_target))
    features = features.iloc[:min_length]
    budget_target = budget_target.iloc[:min_length]
    risk_target = risk_target.iloc[:min_length]
    
    # Initialize and train models
    budget_optimizer = BudgetOptimizer(algorithm="xgboost")
    goal_tracker = GoalTracker(algorithm="linear_regression")
    risk_assessor = RiskAssessor(algorithm="random_forest")
    
    budget_optimizer.fit(features, budget_target)
    
    # For goal tracker, use a subset of features that matches goal data length
    goal_features = features.iloc[:len(goal_target)]
    goal_tracker.fit(goal_features, goal_target)
    
    risk_assessor.fit(features, risk_target)
    
    print("   Models trained successfully")
    
    print("\n4. Making predictions and generating recommendations...")
    
    # Make predictions
    budget_predictions = budget_optimizer.predict(features)
    goal_predictions = goal_tracker.predict(features)
    risk_predictions = risk_assessor.predict(features)
    
    # Generate recommendations
    budget_recommendations = budget_optimizer.predict_expense_recommendations(features)
    risk_profiles = risk_assessor.predict_risk_profile(features)
    
    print("\n5. Analysis Results:")
    
    # Budget analysis
    current_savings_rate = monthly_summary['savings_rate'].iloc[-1]
    predicted_savings_rate = budget_predictions[-1]
    
    print(f"   Current Savings Rate: {current_savings_rate:.1%}")
    print(f"   Predicted Savings Rate: {predicted_savings_rate:.1%}")
    
    latest_recommendation = budget_recommendations.iloc[-1]['recommendation']
    print(f"   Budget Recommendation: {latest_recommendation}")
    
    # Goal analysis
    print(f"\n   Financial Goals Progress:")
    for i, goal in financial_data.goals.iterrows():
        progress = goal['current_amount'] / goal['target_amount']
        print(f"   - {goal['goal_name']}: {progress:.1%} complete")
    
    # Risk assessment
    latest_risk = risk_profiles.iloc[-1]
    print(f"\n   Risk Assessment:")
    print(f"   - Risk Tolerance: {latest_risk['risk_tolerance']}")
    print(f"   - Confidence: {latest_risk['confidence']:.1%}")
    print(f"   - Recommendation: {latest_risk['recommendation']}")
    
    print("\n6. Running backtest...")
    
    # Run backtest
    backtester = Backtester(initial_capital=10000)
    backtest_results = backtester.run_backtest(
        monthly_summary,
        {
            'emergency_fund_first': True,
            'investment_rate': 0.7
        }
    )
    
    print(f"   Total Return: {backtest_results['total_return']:.1%}")
    print(f"   Annualized Return: {backtest_results['annualized_return']:.1%}")
    print(f"   Sharpe Ratio: {backtest_results['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {backtest_results['max_drawdown']:.1%}")
    
    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("="*60)
    print()
    print("To run the interactive demo:")
    print("  streamlit run demo/app.py")
    print()
    print("To train models with custom data:")
    print("  python scripts/train_models.py")
    print()
    print("To evaluate model performance:")
    print("  python scripts/evaluate.py")
    print()
    print("Remember: This is for educational purposes only, not financial advice!")


if __name__ == "__main__":
    main()
