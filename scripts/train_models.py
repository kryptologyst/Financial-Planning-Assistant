#!/usr/bin/env python3
"""Script to train financial planning models."""

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to train models."""
    parser = argparse.ArgumentParser(description="Train financial planning models")
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data", 
        help="Directory containing training data"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="models", 
        help="Output directory for trained models"
    )
    parser.add_argument(
        "--budget-algorithm", 
        type=str, 
        default="xgboost", 
        choices=["xgboost", "lightgbm", "random_forest"],
        help="Algorithm for budget optimizer"
    )
    parser.add_argument(
        "--goal-algorithm", 
        type=str, 
        default="linear_regression", 
        choices=["linear_regression", "ridge", "random_forest"],
        help="Algorithm for goal tracker"
    )
    parser.add_argument(
        "--risk-algorithm", 
        type=str, 
        default="random_forest", 
        choices=["random_forest", "logistic_regression", "xgboost"],
        help="Algorithm for risk assessor"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting model training")
    
    # Load data
    processor = DataProcessor()
    
    try:
        financial_data = processor.load_data(args.data_dir)
        logger.info("Data loaded successfully")
    except FileNotFoundError:
        logger.info("No existing data found, generating synthetic data")
        financial_data = processor.generate_synthetic_data()
        processor.save_data(financial_data, args.data_dir)
    
    # Calculate monthly summary
    monthly_summary = processor.calculate_monthly_summary(financial_data)
    
    # Create features
    feature_engineer = FeatureEngineer()
    features = feature_engineer.create_all_features(financial_data, monthly_summary)
    
    logger.info(f"Created {len(features.columns)} features")
    
    # Prepare targets
    # Budget optimizer target: savings rate
    budget_target = monthly_summary['savings_rate']
    
    # Goal tracker target: progress percentage
    if 'progress_percentage' in financial_data.goals.columns:
        goal_target = financial_data.goals['progress_percentage']
    else:
        # Calculate progress percentage
        goal_target = financial_data.goals['current_amount'] / financial_data.goals['target_amount']
    
    # Risk assessor target: synthetic risk tolerance labels
    np.random.seed(42)
    risk_target = pd.Series(np.random.choice([0, 1, 2], size=len(monthly_summary)))
    
    # Initialize models
    budget_optimizer = BudgetOptimizer(algorithm=args.budget_algorithm)
    goal_tracker = GoalTracker(algorithm=args.goal_algorithm)
    risk_assessor = RiskAssessor(algorithm=args.risk_algorithm)
    
    # Create model ensemble
    ensemble = ModelEnsemble()
    ensemble.add_model("budget_optimizer", budget_optimizer)
    ensemble.add_model("goal_tracker", goal_tracker)
    ensemble.add_model("risk_assessor", risk_assessor)
    
    # Prepare targets dictionary
    targets = {
        "budget_optimizer": budget_target,
        "goal_tracker": goal_target,
        "risk_assessor": risk_target
    }
    
    # Train models
    logger.info("Training models...")
    ensemble.fit_all(features, targets)
    
    # Save models
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ensemble.save_ensemble(args.output_dir)
    
    logger.info("Model training completed successfully")
    logger.info(f"Models saved to {args.output_dir}")
    
    # Display model information
    for name, model in ensemble.models.items():
        logger.info(f"{name}: {model.model_name}")
        if hasattr(model, 'feature_importance_') and model.feature_importance_ is not None:
            top_features = model.feature_importance_.head(5)
            logger.info(f"Top features: {list(top_features.index)}")


if __name__ == "__main__":
    main()
