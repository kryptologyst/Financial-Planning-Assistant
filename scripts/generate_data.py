#!/usr/bin/env python3
"""Script to generate synthetic financial data for the planning assistant."""

import sys
from pathlib import Path
import logging
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.processor import DataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to generate synthetic data."""
    parser = argparse.ArgumentParser(description="Generate synthetic financial data")
    parser.add_argument(
        "--months", 
        type=int, 
        default=12, 
        help="Number of months of data to generate"
    )
    parser.add_argument(
        "--transactions", 
        type=int, 
        default=1000, 
        help="Number of transactions to generate"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="data", 
        help="Output directory for generated data"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting synthetic data generation")
    
    # Initialize data processor
    processor = DataProcessor()
    
    # Generate synthetic data
    financial_data = processor.generate_synthetic_data(
        n_months=args.months,
        n_transactions=args.transactions
    )
    
    # Save data
    processor.save_data(financial_data, args.output_dir)
    
    # Calculate and display summary
    monthly_summary = processor.calculate_monthly_summary(financial_data)
    
    logger.info("Data generation completed successfully")
    logger.info(f"Generated {len(monthly_summary)} months of data")
    logger.info(f"Generated {len(financial_data.transactions)} transactions")
    logger.info(f"Average monthly income: ${monthly_summary['total_income'].mean():,.0f}")
    logger.info(f"Average monthly expenses: ${monthly_summary['total_expenses'].mean():,.0f}")
    logger.info(f"Average savings rate: {monthly_summary['savings_rate'].mean():.1%}")


if __name__ == "__main__":
    main()
