"""Feature engineering module for financial planning assistant."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature engineering for financial planning models."""
    
    def __init__(self) -> None:
        """Initialize feature engineer."""
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: List[str] = []
        
    def create_budget_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for budget optimization.
        
        Args:
            data: Financial data containing expenses and income
            
        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame()
        
        # Basic financial ratios
        features['expense_to_income_ratio'] = (
            data['total_expenses'] / data['total_income']
        )
        features['savings_rate'] = data['savings_rate']
        features['expense_variance'] = data['total_expenses'].rolling(3).std()
        
        # Trend features
        features['expense_trend'] = data['total_expenses'].diff()
        features['income_trend'] = data['total_income'].diff()
        features['savings_trend'] = data['net_savings'].diff()
        
        # Rolling statistics
        features['expense_ma_3'] = data['total_expenses'].rolling(3).mean()
        features['expense_ma_6'] = data['total_expenses'].rolling(6).mean()
        features['income_ma_3'] = data['total_income'].rolling(3).mean()
        
        # Volatility measures
        features['expense_volatility'] = (
            data['total_expenses'].rolling(6).std() / 
            data['total_expenses'].rolling(6).mean()
        )
        
        # Seasonal features
        features['month'] = pd.to_datetime(data['month']).dt.month
        features['quarter'] = features['month'].apply(lambda x: (x-1)//3 + 1)
        
        # Binary features
        features['above_savings_target'] = (
            features['savings_rate'] > 0.2
        ).astype(int)
        
        features['expense_increasing'] = (
            features['expense_trend'] > 0
        ).astype(int)
        
        return features.fillna(0)
    
    def create_goal_features(self, goals_data: pd.DataFrame) -> pd.DataFrame:
        """Create features for goal tracking.
        
        Args:
            goals_data: Goals data
            
        Returns:
            DataFrame with goal-related features
        """
        features = pd.DataFrame()
        
        # Time-based features
        features['days_to_target'] = (
            goals_data['target_date'] - pd.Timestamp.now()
        ).dt.days
        
        features['months_to_target'] = features['days_to_target'] / 30
        
        # Progress features
        features['progress_percentage'] = (
            goals_data['current_amount'] / goals_data['target_amount']
        )
        
        features['remaining_amount'] = (
            goals_data['target_amount'] - goals_data['current_amount']
        )
        
        # Required monthly contribution
        features['required_monthly_contribution'] = (
            features['remaining_amount'] / features['months_to_target']
        )
        
        # Priority encoding
        priority_map = {'High': 3, 'Medium': 2, 'Low': 1}
        features['priority_score'] = goals_data['priority'].map(priority_map)
        
        # Goal size categories (encoded as numeric)
        goal_size_categories = pd.cut(
            goals_data['target_amount'],
            bins=[0, 5000, 20000, 100000, float('inf')],
            labels=[0, 1, 2, 3]  # Use numeric labels instead of strings
        )
        features['goal_size_category'] = goal_size_categories.astype(float)
        
        return features
    
    def create_risk_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for risk assessment.
        
        Args:
            data: Financial data
            
        Returns:
            DataFrame with risk-related features
        """
        features = pd.DataFrame()
        
        # Income stability
        features['income_stability'] = (
            1 - data['total_income'].rolling(6).std() / 
            data['total_income'].rolling(6).mean()
        )
        
        # Expense predictability
        features['expense_predictability'] = (
            1 - data['total_expenses'].rolling(6).std() / 
            data['total_expenses'].rolling(6).mean()
        )
        
        # Emergency fund adequacy
        features['emergency_fund_months'] = (
            data['net_savings'].rolling(12).sum() / 
            data['total_expenses'].rolling(3).mean()
        )
        
        # Debt-to-income ratio (simulated)
        features['debt_to_income'] = np.random.uniform(0.1, 0.4, len(data))
        
        # Savings consistency
        features['savings_consistency'] = (
            data['savings_rate'].rolling(6).std()
        )
        
        # Financial stress indicators
        features['expense_growth_rate'] = (
            data['total_expenses'].pct_change().rolling(3).mean()
        )
        
        features['income_growth_rate'] = (
            data['total_income'].pct_change().rolling(3).mean()
        )
        
        return features.fillna(0)
    
    def create_transaction_features(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Create features from transaction data.
        
        Args:
            transactions: Transaction data
            
        Returns:
            DataFrame with transaction-based features
        """
        # Add month column if not present
        if 'month' not in transactions.columns and 'date' in transactions.columns:
            transactions = transactions.copy()
            transactions['month'] = pd.to_datetime(transactions['date']).dt.to_period('M')
        
        # Group by month and category
        monthly_transactions = (
            transactions.groupby(['month', 'category'])['amount']
            .sum()
            .unstack(fill_value=0)
        )
        
        features = pd.DataFrame()
        
        # Category-wise spending patterns
        for category in monthly_transactions.columns:
            features[f'{category}_spending'] = monthly_transactions[category]
            features[f'{category}_spending_ma'] = (
                monthly_transactions[category].rolling(3).mean()
            )
        
        # Spending diversity
        features['spending_diversity'] = (
            monthly_transactions.abs().sum(axis=1) / 
            monthly_transactions.abs().count(axis=1)
        )
        
        # Transaction frequency
        transaction_counts = (
            transactions.groupby('month').size()
        )
        features['transaction_frequency'] = transaction_counts
        
        return features.fillna(0)
    
    def scale_features(
        self, 
        features: pd.DataFrame, 
        feature_type: str,
        fit: bool = True
    ) -> pd.DataFrame:
        """Scale features using StandardScaler.
        
        Args:
            features: Features to scale
            feature_type: Type of features (for storing scaler)
            fit: Whether to fit the scaler
            
        Returns:
            Scaled features
        """
        if fit:
            self.scalers[feature_type] = StandardScaler()
            scaled_features = self.scalers[feature_type].fit_transform(features)
        else:
            if feature_type not in self.scalers:
                raise ValueError(f"Scaler for {feature_type} not found")
            scaled_features = self.scalers[feature_type].transform(features)
        
        return pd.DataFrame(
            scaled_features, 
            columns=features.columns, 
            index=features.index
        )
    
    def select_features(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        k: int = 10
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Select top k features using statistical tests.
        
        Args:
            X: Feature matrix
            y: Target variable
            k: Number of features to select
            
        Returns:
            Tuple of (selected_features, feature_names)
        """
        selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = selector.fit_transform(X, y)
        
        selected_features = pd.DataFrame(
            X_selected,
            columns=X.columns[selector.get_support()],
            index=X.index
        )
        
        self.feature_names = list(selected_features.columns)
        
        return selected_features, self.feature_names
    
    def create_all_features(
        self, 
        financial_data: Any,  # FinancialData type
        monthly_summary: pd.DataFrame
    ) -> pd.DataFrame:
        """Create all features for the financial planning models.
        
        Args:
            financial_data: FinancialData object
            monthly_summary: Monthly summary data
            
        Returns:
            Combined feature matrix
        """
        logger.info("Creating comprehensive feature set")
        
        # Create different feature sets
        budget_features = self.create_budget_features(monthly_summary)
        goal_features = self.create_goal_features(financial_data.goals)
        risk_features = self.create_risk_features(monthly_summary)
        transaction_features = self.create_transaction_features(financial_data.transactions)
        
        # Combine features
        all_features = pd.concat([
            budget_features,
            risk_features,
            transaction_features
        ], axis=1)
        
        # Handle goal features separately (different index)
        if not goal_features.empty:
            # For goals, we'll create one row per goal
            goal_features_expanded = pd.DataFrame()
            for idx, goal_row in goal_features.iterrows():
                goal_features_expanded = pd.concat([
                    goal_features_expanded,
                    pd.DataFrame([goal_row] * len(all_features))
                ], ignore_index=True)
            
            # Repeat goal features for each month
            all_features = pd.concat([
                all_features.reset_index(drop=True),
                goal_features_expanded.reset_index(drop=True)
            ], axis=1)
        
        logger.info(f"Created {len(all_features.columns)} features")
        
        # Ensure all features are numeric
        for col in all_features.columns:
            if all_features[col].dtype == 'object':
                # Convert object columns to numeric, replacing non-numeric values with 0
                all_features[col] = pd.to_numeric(all_features[col], errors='coerce').fillna(0)
        
        return all_features.fillna(0)
