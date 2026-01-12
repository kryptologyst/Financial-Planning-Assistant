"""Streamlit demo application for Financial Planning Assistant.

DISCLAIMER: This is for educational and research purposes only.
This software is NOT intended for investment advice or financial planning advice.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data.processor import DataProcessor, FinancialData
from src.features.engineering import FeatureEngineer
from src.models.planners import BudgetOptimizer, GoalTracker, RiskAssessor, ModelEnsemble
from src.evaluation.metrics import ModelEvaluator, Backtester

# Configure page
st.set_page_config(
    page_title="Financial Planning Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add disclaimer banner
st.error("""
**DISCLAIMER: This is a research and educational demonstration project only. 
This software is NOT intended for investment advice, financial planning advice, 
or any form of financial guidance. All outputs, recommendations, and analyses 
are for educational purposes only and may be inaccurate. Users should consult 
qualified financial professionals for actual financial planning needs.**
""")

def main():
    """Main application function."""
    
    st.title("💰 Financial Planning Assistant")
    st.markdown("**Research and Educational Tool for Personal Finance Analysis**")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Data generation options
    st.sidebar.subheader("Data Settings")
    n_months = st.sidebar.slider("Number of months", 6, 24, 12)
    n_transactions = st.sidebar.slider("Number of transactions", 500, 2000, 1000)
    
    # Model options
    st.sidebar.subheader("Model Settings")
    budget_algorithm = st.sidebar.selectbox(
        "Budget Optimizer Algorithm",
        ["xgboost", "lightgbm", "random_forest"]
    )
    goal_algorithm = st.sidebar.selectbox(
        "Goal Tracker Algorithm", 
        ["linear_regression", "ridge", "random_forest"]
    )
    risk_algorithm = st.sidebar.selectbox(
        "Risk Assessor Algorithm",
        ["random_forest", "logistic_regression", "xgboost"]
    )
    
    # Initialize session state
    if 'data_generated' not in st.session_state:
        st.session_state.data_generated = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "🤖 Model Training", 
        "📈 Analysis", 
        "🎯 Goals", 
        "⚠️ Risk Assessment"
    ])
    
    with tab1:
        show_data_overview(n_months, n_transactions)
    
    with tab2:
        show_model_training(budget_algorithm, goal_algorithm, risk_algorithm)
    
    with tab3:
        show_analysis()
    
    with tab4:
        show_goals_analysis()
    
    with tab5:
        show_risk_assessment()


def show_data_overview(n_months: int, n_transactions: int):
    """Show data overview tab."""
    st.header("📊 Financial Data Overview")
    
    if st.button("Generate Synthetic Data", key="generate_data"):
        with st.spinner("Generating synthetic financial data..."):
            # Initialize data processor
            processor = DataProcessor()
            
            # Generate synthetic data
            financial_data = processor.generate_synthetic_data(n_months, n_transactions)
            
            # Calculate monthly summary
            monthly_summary = processor.calculate_monthly_summary(financial_data)
            
            # Store in session state
            st.session_state.financial_data = financial_data
            st.session_state.monthly_summary = monthly_summary
            st.session_state.data_generated = True
            
            st.success("Data generated successfully!")
    
    if st.session_state.data_generated:
        financial_data = st.session_state.financial_data
        monthly_summary = st.session_state.monthly_summary
        
        # Display data summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Months", 
                len(monthly_summary),
                delta=None
            )
        
        with col2:
            avg_income = monthly_summary['total_income'].mean()
            st.metric(
                "Avg Monthly Income", 
                f"${avg_income:,.0f}",
                delta=None
            )
        
        with col3:
            avg_expenses = monthly_summary['total_expenses'].mean()
            st.metric(
                "Avg Monthly Expenses", 
                f"${avg_expenses:,.0f}",
                delta=None
            )
        
        with col4:
            avg_savings_rate = monthly_summary['savings_rate'].mean()
            st.metric(
                "Avg Savings Rate", 
                f"{avg_savings_rate:.1%}",
                delta=None
            )
        
        # Visualizations
        st.subheader("Financial Trends")
        
        # Income vs Expenses over time
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Income vs Expenses', 'Savings Rate Trend', 
                          'Expense Categories', 'Monthly Net Savings'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Income vs Expenses
        fig.add_trace(
            go.Scatter(x=monthly_summary['month'], y=monthly_summary['total_income'],
                      name='Income', line=dict(color='green')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=monthly_summary['month'], y=monthly_summary['total_expenses'],
                      name='Expenses', line=dict(color='red')),
            row=1, col=1
        )
        
        # Savings Rate
        fig.add_trace(
            go.Scatter(x=monthly_summary['month'], y=monthly_summary['savings_rate'],
                      name='Savings Rate', line=dict(color='blue')),
            row=1, col=2
        )
        
        # Expense Categories
        expense_by_category = financial_data.expenses.groupby('category')['amount'].sum()
        fig.add_trace(
            go.Bar(x=expense_by_category.index, y=expense_by_category.values,
                  name='Expenses by Category'),
            row=2, col=1
        )
        
        # Net Savings
        fig.add_trace(
            go.Bar(x=monthly_summary['month'], y=monthly_summary['net_savings'],
                  name='Net Savings'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Data tables
        st.subheader("Raw Data")
        
        data_tab1, data_tab2, data_tab3, data_tab4 = st.tabs([
            "Expenses", "Income", "Goals", "Transactions"
        ])
        
        with data_tab1:
            st.dataframe(financial_data.expenses.head(20))
        
        with data_tab2:
            st.dataframe(financial_data.income.head(20))
        
        with data_tab3:
            st.dataframe(financial_data.goals)
        
        with data_tab4:
            st.dataframe(financial_data.transactions.head(20))


def show_model_training(budget_algorithm: str, goal_algorithm: str, risk_algorithm: str):
    """Show model training tab."""
    st.header("🤖 Model Training")
    
    if not st.session_state.data_generated:
        st.warning("Please generate data first in the Data Overview tab.")
        return
    
    if st.button("Train Models", key="train_models"):
        with st.spinner("Training financial planning models..."):
            # Get data from session state
            financial_data = st.session_state.financial_data
            monthly_summary = st.session_state.monthly_summary
            
            # Initialize feature engineer
            feature_engineer = FeatureEngineer()
            
            # Create features
            features = feature_engineer.create_all_features(financial_data, monthly_summary)
            
            # Prepare targets
            budget_target = monthly_summary['savings_rate']
            goal_target = financial_data.goals['progress_percentage'] if 'progress_percentage' in financial_data.goals.columns else pd.Series([0.5] * len(financial_data.goals))
            risk_target = pd.Series(np.random.choice([0, 1, 2], size=len(monthly_summary)))  # Synthetic risk labels
            
            # Initialize models
            budget_optimizer = BudgetOptimizer(algorithm=budget_algorithm)
            goal_tracker = GoalTracker(algorithm=goal_algorithm)
            risk_assessor = RiskAssessor(algorithm=risk_algorithm)
            
            # Train models
            budget_optimizer.fit(features, budget_target)
            goal_tracker.fit(features, goal_target)
            risk_assessor.fit(features, risk_target)
            
            # Store in session state
            st.session_state.models = {
                'budget_optimizer': budget_optimizer,
                'goal_tracker': goal_tracker,
                'risk_assessor': risk_assessor
            }
            st.session_state.features = features
            st.session_state.models_trained = True
            
            st.success("Models trained successfully!")
    
    if st.session_state.models_trained:
        st.subheader("Model Performance")
        
        models = st.session_state.models
        
        # Display model information
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Budget Optimizer", models['budget_optimizer'].model_name)
        
        with col2:
            st.metric("Goal Tracker", models['goal_tracker'].model_name)
        
        with col3:
            st.metric("Risk Assessor", models['risk_assessor'].model_name)
        
        # Feature importance
        st.subheader("Feature Importance")
        
        if hasattr(models['budget_optimizer'], 'feature_importance_') and models['budget_optimizer'].feature_importance_ is not None:
            importance_df = models['budget_optimizer'].feature_importance_.head(10)
            
            fig = px.bar(
                x=importance_df.values,
                y=importance_df.index,
                orientation='h',
                title="Top 10 Most Important Features for Budget Optimization"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


def show_analysis():
    """Show analysis tab."""
    st.header("📈 Financial Analysis")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first in the Model Training tab.")
        return
    
    models = st.session_state.models
    features = st.session_state.features
    monthly_summary = st.session_state.monthly_summary
    
    # Budget Analysis
    st.subheader("Budget Analysis")
    
    # Make predictions
    budget_predictions = models['budget_optimizer'].predict(features)
    
    # Display current vs predicted savings rate
    col1, col2 = st.columns(2)
    
    with col1:
        current_savings_rate = monthly_summary['savings_rate'].iloc[-1]
        st.metric("Current Savings Rate", f"{current_savings_rate:.1%}")
    
    with col2:
        predicted_savings_rate = budget_predictions[-1]
        st.metric("Predicted Savings Rate", f"{predicted_savings_rate:.1%}")
    
    # Savings rate trend
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_summary['month'],
        y=monthly_summary['savings_rate'],
        name='Actual Savings Rate',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=monthly_summary['month'],
        y=budget_predictions[:len(monthly_summary)],
        name='Predicted Savings Rate',
        line=dict(color='red', dash='dash')
    ))
    fig.add_hline(y=0.2, line_dash="dot", line_color="green", 
                  annotation_text="20% Target")
    
    fig.update_layout(
        title="Savings Rate: Actual vs Predicted",
        xaxis_title="Month",
        yaxis_title="Savings Rate"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.subheader("Budget Recommendations")
    
    recommendations = models['budget_optimizer'].predict_expense_recommendations(features)
    latest_recommendation = recommendations.iloc[-1]['recommendation']
    
    st.info(f"**Recommendation:** {latest_recommendation}")
    
    # Expense breakdown
    expense_categories = st.session_state.financial_data.expenses.groupby('category')['amount'].sum()
    
    fig = px.pie(
        values=expense_categories.values,
        names=expense_categories.index,
        title="Expense Distribution by Category"
    )
    st.plotly_chart(fig, use_container_width=True)


def show_goals_analysis():
    """Show goals analysis tab."""
    st.header("🎯 Financial Goals Analysis")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first in the Model Training tab.")
        return
    
    models = st.session_state.models
    financial_data = st.session_state.financial_data
    
    # Goals overview
    st.subheader("Current Goals")
    
    goals_df = financial_data.goals.copy()
    
    # Calculate progress percentages
    goals_df['progress_percentage'] = goals_df['current_amount'] / goals_df['target_amount']
    goals_df['remaining_amount'] = goals_df['target_amount'] - goals_df['current_amount']
    
    # Display goals
    for _, goal in goals_df.iterrows():
        with st.expander(f"{goal['goal_name']} - {goal['priority']} Priority"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Target Amount", f"${goal['target_amount']:,.0f}")
            
            with col2:
                st.metric("Current Amount", f"${goal['current_amount']:,.0f}")
            
            with col3:
                st.metric("Progress", f"{goal['progress_percentage']:.1%}")
            
            # Progress bar
            st.progress(goal['progress_percentage'])
            
            # Goal timeline
            days_remaining = (goal['target_date'] - pd.Timestamp.now()).days
            st.write(f"**Days remaining:** {days_remaining}")
            
            if days_remaining > 0:
                required_monthly = goal['remaining_amount'] / (days_remaining / 30)
                st.write(f"**Required monthly contribution:** ${required_monthly:,.0f}")
    
    # Goals visualization
    st.subheader("Goals Progress Visualization")
    
    fig = px.bar(
        x=goals_df['goal_name'],
        y=goals_df['progress_percentage'],
        title="Progress Towards Financial Goals",
        color=goals_df['priority'],
        color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'}
    )
    fig.update_layout(yaxis_title="Progress Percentage")
    st.plotly_chart(fig, use_container_width=True)


def show_risk_assessment():
    """Show risk assessment tab."""
    st.header("⚠️ Risk Assessment")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first in the Model Training tab.")
        return
    
    models = st.session_state.models
    features = st.session_state.features
    monthly_summary = st.session_state.monthly_summary
    
    # Risk profile prediction
    st.subheader("Risk Profile Assessment")
    
    # Make risk predictions
    risk_predictions = models['risk_assessor'].predict(features)
    risk_profiles = models['risk_assessor'].predict_risk_profile(features)
    
    # Display current risk assessment
    latest_risk = risk_profiles.iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_level = latest_risk['risk_tolerance']
        color = {'Conservative': 'green', 'Moderate': 'orange', 'Aggressive': 'red'}
        st.metric("Risk Tolerance", risk_level)
    
    with col2:
        confidence = latest_risk['confidence']
        st.metric("Confidence", f"{confidence:.1%}")
    
    with col3:
        st.metric("Recommendation", latest_risk['recommendation'])
    
    # Risk factors analysis
    st.subheader("Risk Factors Analysis")
    
    # Calculate risk factors
    income_stability = 1 - monthly_summary['total_income'].rolling(3).std().iloc[-1] / monthly_summary['total_income'].rolling(3).mean().iloc[-1]
    expense_predictability = 1 - monthly_summary['total_expenses'].rolling(3).std().iloc[-1] / monthly_summary['total_expenses'].rolling(3).mean().iloc[-1]
    savings_consistency = monthly_summary['savings_rate'].std()
    
    risk_factors = pd.DataFrame({
        'Factor': ['Income Stability', 'Expense Predictability', 'Savings Consistency'],
        'Score': [income_stability, expense_predictability, 1 - savings_consistency],
        'Risk Level': ['Low' if x > 0.7 else 'Medium' if x > 0.4 else 'High' for x in [income_stability, expense_predictability, 1 - savings_consistency]]
    })
    
    fig = px.bar(
        risk_factors,
        x='Factor',
        y='Score',
        color='Risk Level',
        title="Risk Factors Assessment",
        color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
    )
    fig.update_layout(yaxis_title="Risk Score (Higher = Lower Risk)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk recommendations
    st.subheader("Risk Management Recommendations")
    
    recommendations = []
    
    if income_stability < 0.7:
        recommendations.append("Consider building multiple income streams for stability")
    
    if expense_predictability < 0.7:
        recommendations.append("Create a detailed budget to improve expense predictability")
    
    if savings_consistency > 0.1:
        recommendations.append("Set up automatic savings transfers for consistency")
    
    if latest_risk['risk_tolerance'] == 'Conservative':
        recommendations.append("Focus on emergency fund and low-risk investments")
    elif latest_risk['risk_tolerance'] == 'Aggressive':
        recommendations.append("Consider higher-risk investments but maintain emergency fund")
    
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")


if __name__ == "__main__":
    main()
