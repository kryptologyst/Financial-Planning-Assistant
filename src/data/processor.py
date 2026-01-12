"""Data processing module for financial planning assistant.

This module handles data loading, preprocessing, and synthetic data generation
for the financial planning assistant.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import yaml
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class FinancialData:
    """Container for financial data."""
    
    expenses: pd.DataFrame
    income: pd.DataFrame
    goals: pd.DataFrame
    transactions: pd.DataFrame
    
    def __post_init__(self) -> None:
        """Validate data after initialization."""
        self._validate_data()
    
    def _validate_data(self) -> None:
        """Validate that all dataframes have required columns."""
        required_columns = {
            'expenses': ['category', 'amount', 'date'],
            'income': ['source', 'amount', 'date'],
            'goals': ['goal_name', 'target_amount', 'target_date', 'current_amount'],
            'transactions': ['description', 'amount', 'category', 'date']
        }
        
        for df_name, required_cols in required_columns.items():
            df = getattr(self, df_name)
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                raise ValueError(f"{df_name} missing required columns: {missing_cols}")


class DataProcessor:
    """Handles data processing and synthetic data generation."""
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize data processor with configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config = self._load_config(config_path)
        self.seed = self.config.get('data', {}).get('seed', 42)
        np.random.seed(self.seed)
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Return default config if file not found
            return {
                'data': {
                    'seed': 42,
                    'categories': ['Rent', 'Groceries', 'Utilities', 'Entertainment', 'Transportation', 'Insurance', 'Miscellaneous']
                },
                'budget': {
                    'default_income': 4000,
                    'savings_rate_target': 0.20
                }
            }
    
    def generate_synthetic_data(
        self, 
        n_months: int = 12,
        n_transactions: int = 1000
    ) -> FinancialData:
        """Generate synthetic financial data for demonstration purposes.
        
        Args:
            n_months: Number of months of data to generate
            n_transactions: Number of transactions to generate
            
        Returns:
            FinancialData object containing synthetic data
        """
        logger.info(f"Generating synthetic data for {n_months} months")
        
        # Generate expenses data
        expenses = self._generate_expenses_data(n_months)
        
        # Generate income data
        income = self._generate_income_data(n_months)
        
        # Generate goals data
        goals = self._generate_goals_data()
        
        # Generate transactions data
        transactions = self._generate_transactions_data(n_transactions, n_months)
        
        return FinancialData(
            expenses=expenses,
            income=income,
            goals=goals,
            transactions=transactions
        )
    
    def _generate_expenses_data(self, n_months: int) -> pd.DataFrame:
        """Generate synthetic monthly expenses data."""
        categories = self.config['data']['categories']
        base_amounts = {
            'Rent': 1200,
            'Groceries': 400,
            'Utilities': 150,
            'Entertainment': 200,
            'Transportation': 300,
            'Insurance': 100,
            'Miscellaneous': 150,
            'Healthcare': 200,
            'Education': 300,
            'Savings': 0  # Will be calculated separately
        }
        
        data = []
        start_date = datetime.now() - timedelta(days=30 * n_months)
        
        for month in range(n_months):
            month_date = start_date + timedelta(days=30 * month)
            
            for category in categories:
                if category == 'Savings':
                    continue
                    
                base_amount = base_amounts.get(category, 100)
                # Add some randomness (±20%)
                variance = np.random.normal(0, 0.1)
                amount = max(0, base_amount * (1 + variance))
                
                data.append({
                    'category': category,
                    'amount': round(amount, 2),
                    'date': month_date,
                    'month': month_date.strftime('%Y-%m')
                })
        
        return pd.DataFrame(data)
    
    def _generate_income_data(self, n_months: int) -> pd.DataFrame:
        """Generate synthetic income data."""
        base_income = self.config['budget']['default_income']
        
        data = []
        start_date = datetime.now() - timedelta(days=30 * n_months)
        
        for month in range(n_months):
            month_date = start_date + timedelta(days=30 * month)
            
            # Add some income variance (±5%)
            variance = np.random.normal(0, 0.02)
            amount = base_income * (1 + variance)
            
            data.append({
                'source': 'Salary',
                'amount': round(amount, 2),
                'date': month_date,
                'month': month_date.strftime('%Y-%m')
            })
        
        return pd.DataFrame(data)
    
    def _generate_goals_data(self) -> pd.DataFrame:
        """Generate synthetic financial goals data."""
        goals_data = [
            {
                'goal_name': 'Emergency Fund',
                'target_amount': 10000,
                'target_date': datetime.now() + timedelta(days=365),
                'current_amount': 2500,
                'priority': 'High'
            },
            {
                'goal_name': 'Vacation Fund',
                'target_amount': 5000,
                'target_date': datetime.now() + timedelta(days=180),
                'current_amount': 1200,
                'priority': 'Medium'
            },
            {
                'goal_name': 'Home Down Payment',
                'target_amount': 50000,
                'target_date': datetime.now() + timedelta(days=1095),
                'current_amount': 8000,
                'priority': 'High'
            }
        ]
        
        return pd.DataFrame(goals_data)
    
    def _generate_transactions_data(self, n_transactions: int, n_months: int) -> pd.DataFrame:
        """Generate synthetic transaction data."""
        categories = self.config['data']['categories']
        descriptions = {
            'Groceries': ['Whole Foods', 'Safeway', 'Trader Joes', 'Local Market'],
            'Entertainment': ['Netflix', 'Spotify', 'Movie Theater', 'Restaurant'],
            'Transportation': ['Gas Station', 'Uber', 'Public Transit', 'Car Maintenance'],
            'Utilities': ['Electric Bill', 'Water Bill', 'Internet', 'Phone Bill'],
            'Healthcare': ['Doctor Visit', 'Pharmacy', 'Dental', 'Vision'],
            'Miscellaneous': ['Coffee Shop', 'ATM Withdrawal', 'Online Purchase', 'Cash']
        }
        
        data = []
        start_date = datetime.now() - timedelta(days=30 * n_months)
        
        for _ in range(n_transactions):
            # Random date within the period
            days_offset = np.random.randint(0, 30 * n_months)
            transaction_date = start_date + timedelta(days=days_offset)
            
            # Random category
            category = np.random.choice(categories)
            
            # Random description
            category_descriptions = descriptions.get(category, ['Purchase'])
            description = np.random.choice(category_descriptions)
            
            # Random amount (negative for expenses)
            if category == 'Savings':
                amount = np.random.uniform(100, 500)
            else:
                amount = -np.random.uniform(10, 200)
            
            data.append({
                'description': description,
                'amount': round(amount, 2),
                'category': category,
                'date': transaction_date
            })
        
        return pd.DataFrame(data)
    
    def load_data(self, data_dir: str) -> FinancialData:
        """Load financial data from CSV files.
        
        Args:
            data_dir: Directory containing CSV files
            
        Returns:
            FinancialData object containing loaded data
        """
        data_path = Path(data_dir)
        
        try:
            expenses = pd.read_csv(data_path / "expenses.csv")
            income = pd.read_csv(data_path / "income.csv")
            goals = pd.read_csv(data_path / "goals.csv")
            transactions = pd.read_csv(data_path / "transactions.csv")
            
            # Convert date columns
            for df in [expenses, income, transactions]:
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
            
            if 'target_date' in goals.columns:
                goals['target_date'] = pd.to_datetime(goals['target_date'])
            
            logger.info("Successfully loaded financial data")
            return FinancialData(
                expenses=expenses,
                income=income,
                goals=goals,
                transactions=transactions
            )
            
        except FileNotFoundError as e:
            logger.error(f"Data files not found in {data_dir}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def save_data(self, data: FinancialData, output_dir: str) -> None:
        """Save financial data to CSV files.
        
        Args:
            data: FinancialData object to save
            output_dir: Directory to save CSV files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        data.expenses.to_csv(output_path / "expenses.csv", index=False)
        data.income.to_csv(output_path / "income.csv", index=False)
        data.goals.to_csv(output_path / "goals.csv", index=False)
        data.transactions.to_csv(output_path / "transactions.csv", index=False)
        
        logger.info(f"Data saved to {output_dir}")
    
    def calculate_monthly_summary(self, data: FinancialData) -> pd.DataFrame:
        """Calculate monthly financial summary.
        
        Args:
            data: FinancialData object
            
        Returns:
            DataFrame with monthly summary statistics
        """
        # Group expenses by month
        monthly_expenses = (
            data.expenses.groupby('month')['amount']
            .sum()
            .reset_index()
            .rename(columns={'amount': 'total_expenses'})
        )
        
        # Group income by month
        monthly_income = (
            data.income.groupby('month')['amount']
            .sum()
            .reset_index()
            .rename(columns={'amount': 'total_income'})
        )
        
        # Merge and calculate savings
        summary = monthly_expenses.merge(monthly_income, on='month', how='outer')
        summary['net_savings'] = summary['total_income'] - summary['total_expenses']
        summary['savings_rate'] = summary['net_savings'] / summary['total_income']
        
        return summary.fillna(0)
