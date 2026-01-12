#!/usr/bin/env python3
"""Script to evaluate financial planning models."""

import sys
from pathlib import Path
import logging
import argparse
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.processor import DataProcessor
from src.features.engineering import FeatureEngineer
from src.models.planners import BudgetOptimizer, GoalTracker, RiskAssessor, ModelEnsemble
from src.evaluation.metrics import ModelEvaluator, Backtester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to evaluate models."""
    parser = argparse.ArgumentParser(description="Evaluate financial planning models")
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data", 
        help="Directory containing test data"
    )
    parser.add_argument(
        "--model-dir", 
        type=str, 
        default="models", 
        help="Directory containing trained models"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="assets/results", 
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--test-size", 
        type=float, 
        default=0.2, 
        help="Test set size for evaluation"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting model evaluation")
    
    # Load data
    processor = DataProcessor()
    
    try:
        financial_data = processor.load_data(args.data_dir)
        logger.info("Data loaded successfully")
    except FileNotFoundError:
        logger.error(f"No data found in {args.data_dir}")
        logger.info("Please run generate_data.py first")
        return
    
    # Calculate monthly summary
    monthly_summary = processor.calculate_monthly_summary(financial_data)
    
    # Create features
    feature_engineer = FeatureEngineer()
    features = feature_engineer.create_all_features(financial_data, monthly_summary)
    
    # Prepare targets
    budget_target = monthly_summary['savings_rate']
    
    if 'progress_percentage' in financial_data.goals.columns:
        goal_target = financial_data.goals['progress_percentage']
    else:
        goal_target = financial_data.goals['current_amount'] / financial_data.goals['target_amount']
    
    # Generate synthetic risk labels for evaluation
    np.random.seed(42)
    risk_target = pd.Series(np.random.choice([0, 1, 2], size=len(monthly_summary)))
    
    # Split data for evaluation
    from sklearn.model_selection import train_test_split
    
    # For budget optimizer and risk assessor (monthly data)
    X_train, X_test, y_budget_train, y_budget_test = train_test_split(
        features, budget_target, test_size=args.test_size, random_state=42
    )
    
    _, _, y_risk_train, y_risk_test = train_test_split(
        features, risk_target, test_size=args.test_size, random_state=42
    )
    
    # For goal tracker (goal data)
    goal_features = features.iloc[:len(goal_target)]  # Match goal data size
    X_goal_train, X_goal_test, y_goal_train, y_goal_test = train_test_split(
        goal_features, goal_target, test_size=args.test_size, random_state=42
    )
    
    # Load trained models
    ensemble = ModelEnsemble()
    ensemble.add_model("budget_optimizer", BudgetOptimizer())
    ensemble.add_model("goal_tracker", GoalTracker())
    ensemble.add_model("risk_assessor", RiskAssessor())
    
    try:
        ensemble.load_ensemble(args.model_dir)
        logger.info("Models loaded successfully")
    except FileNotFoundError:
        logger.error(f"No models found in {args.model_dir}")
        logger.info("Please run train_models.py first")
        return
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Prepare test data dictionary
    test_data = {
        'budget_data': {
            'X_test': X_test,
            'y_test': y_budget_test
        },
        'goal_data': {
            'X_test': X_goal_test,
            'y_test': y_goal_test,
            'goal_info': financial_data.goals
        },
        'risk_data': {
            'X_test': X_test,
            'y_test': y_risk_test
        }
    }
    
    # Run comprehensive evaluation
    logger.info("Running comprehensive evaluation...")
    results = evaluator.run_comprehensive_evaluation(ensemble.models, test_data)
    
    # Save results
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    evaluator.save_evaluation_results(results, args.output_dir)
    evaluator.create_evaluation_plots(results, args.output_dir)
    
    # Run backtest
    logger.info("Running backtest...")
    backtester = Backtester()
    backtest_results = backtester.run_backtest(
        monthly_summary,
        {
            'emergency_fund_first': True,
            'investment_rate': 0.7
        }
    )
    
    # Save backtest results
    backtest_df = pd.DataFrame([backtest_results])
    backtest_df.to_csv(Path(args.output_dir) / "backtest_results.csv", index=False)
    
    # Display results
    logger.info("Evaluation completed successfully")
    logger.info(f"Results saved to {args.output_dir}")
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    for model_name, model_results in results.items():
        if model_name == 'summary':
            continue
            
        print(f"\n{model_name.upper().replace('_', ' ')}:")
        
        if 'r2' in model_results:
            print(f"  R² Score: {model_results['r2']:.3f}")
            print(f"  MAE: {model_results['mae']:.3f}")
        elif 'accuracy' in model_results:
            print(f"  Accuracy: {model_results['accuracy']:.3f}")
            print(f"  F1 Score: {model_results['f1_score']:.3f}")
    
    print(f"\nBACKTEST RESULTS:")
    print(f"  Total Return: {backtest_results['total_return']:.1%}")
    print(f"  Annualized Return: {backtest_results['annualized_return']:.1%}")
    print(f"  Sharpe Ratio: {backtest_results['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {backtest_results['max_drawdown']:.1%}")
    print(f"  Goal Achievement Rate: {backtest_results['goal_achievement_rate']:.1%}")


if __name__ == "__main__":
    main()
