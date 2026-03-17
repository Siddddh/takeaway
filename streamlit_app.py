import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import asyncio
from src.models.model_trainer import ModelTrainer
from src.features.feature_engineering import FeatureEngineer
from src.utils.data_generator import generate_synthetic_data
import joblib

# Set page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

# Initialize components
@st.cache_resource
def initialize_components():
    model_trainer = ModelTrainer()
    feature_engineer = FeatureEngineer()
    return model_trainer, feature_engineer

model_trainer, feature_engineer = initialize_components()

# Title and description
st.title("🔍 Fraud Detection System")

# Load metrics directly from the model file
model_data = joblib.load('models/fraud_detector.joblib')
metrics = model_data.get('metrics', {})

# Model Performance Section
st.markdown("### 📊 Model Performance")
st.markdown("""
These metrics show how well the model performs on test data. They are calculated during model training and don't change when you generate demo data.
""")

# Display metrics in a more prominent way
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "F1 Score",
        f"{metrics.get('f1_score', 0.0):.3f}",
        help="Balanced measure of precision and recall"
    )
with col2:
    st.metric(
        "Precision",
        f"{metrics.get('precision', 0.0):.3f}",
        help="Proportion of true fraud cases among predicted fraud cases"
    )
with col3:
    st.metric(
        "Recall",
        f"{metrics.get('recall', 0.0):.3f}",
        help="Proportion of actual fraud cases correctly identified"
    )
with col4:
    st.metric(
        "AUC-ROC",
        f"{metrics.get('auc', 0.0):.3f}",
        help="Model's ability to distinguish between fraud and non-fraud"
    )

st.markdown("""
### Welcome to the Fraud Detection System!

This application helps you detect potentially fraudulent transactions in real-time. You can:

1. **Use Demo Mode**: Generate and analyze synthetic transaction data
2. **Upload Your Data**: Analyze your own transaction data in CSV format

The system uses advanced machine learning to identify suspicious patterns and calculate fraud probability scores.
""")

# Sidebar
st.sidebar.title("Settings")
st.sidebar.markdown("""
Choose how you want to use the system:
- **Demo Mode**: Test the system with generated data
- **Upload Mode**: Analyze your own transaction data in CSV format
""")

demo_mode = st.sidebar.checkbox("Demo Mode", value=True)

if demo_mode:
    st.sidebar.markdown("""
    ### Demo Mode Settings
    Generate synthetic transaction data to test the system. You can adjust the number of transactions to generate.
    """)
    
    # Generate demo data
    n_samples = st.sidebar.slider("Number of Demo Transactions", 10, 10000, 1000)
    if st.sidebar.button("Generate Demo Data"):
        with st.spinner("Generating demo data..."):
            demo_data = generate_synthetic_data(n_samples=n_samples, fraud_ratio=0.5)
            st.session_state.demo_data = demo_data
            st.session_state.processed = False

    if 'demo_data' in st.session_state and not st.session_state.processed:
        # Process demo data
        with st.spinner("Analyzing transactions..."):
            X = feature_engineer.prepare_features(st.session_state.demo_data)
            predictions = model_trainer.predict(X)
            scores = model_trainer.get_fraud_score(X)
            
            # Add predictions to data
            results = st.session_state.demo_data.copy()
            results['is_fraud'] = predictions
            results['fraud_score'] = scores
            
            st.session_state.results = results
            st.session_state.processed = True

    if 'results' in st.session_state:
        # Display results
        st.subheader("📊 Demo Analysis Results")
        st.markdown("""
        These metrics show the results of analyzing your generated demo transactions. They are different from the model performance metrics above.
        """)
        
        # Summary metrics
        st.markdown("### Transaction Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Transactions", len(st.session_state.results))
        with col2:
            st.metric("Fraudulent Transactions", 
                     st.session_state.results['is_fraud'].sum())
        with col3:
            st.metric("Average Fraud Score", 
                     f"{st.session_state.results['fraud_score'].mean():.3f}")
        
        # Display transactions
        st.markdown("### Transaction Details")
        st.markdown("Below is a detailed view of all transactions with their fraud predictions and scores.")
        st.dataframe(st.session_state.results)
        
        # Plot fraud scores distribution
        st.markdown("### Fraud Score Distribution")
        st.markdown("""
        This chart shows the distribution of fraud scores across all transactions. 
        - Higher scores (closer to 1.0) indicate higher probability of fraud
        - Lower scores (closer to 0.0) indicate lower probability of fraud
        """)
        
        # Create bins for better visualization
        bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1
        hist_values, bin_edges = np.histogram(st.session_state.results['fraud_score'], bins=bins)
        
        # Create a DataFrame for plotting
        plot_data = pd.DataFrame({
            'Fraud Score Range': [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)],
            'Number of Transactions': hist_values
        })
        
        # Plot using bar chart with better formatting
        st.bar_chart(
            plot_data.set_index('Fraud Score Range'),
            use_container_width=True
        )
        
        # Add a line showing the average fraud score
        avg_score = st.session_state.results['fraud_score'].mean()
        st.markdown(f"**Average Fraud Score: {avg_score:.3f}**")
        
        # Add explanation of the distribution
        st.markdown("""
        **Understanding the Distribution:**
        - The height of each bar shows how many transactions fall into that fraud score range
        - A higher concentration of transactions in the higher score ranges (right side) suggests more suspicious activity
        - A higher concentration in the lower score ranges (left side) suggests more normal transactions
        """)

        # Highlight Fraudulent Transactions
        st.markdown("### 🚨 Detected Fraudulent Transactions")
        fraudulent_txns = st.session_state.results[st.session_state.results['is_fraud'] == 1].copy()
        
        if len(fraudulent_txns) > 0:
            # Sort by fraud score in descending order
            fraudulent_txns = fraudulent_txns.sort_values('fraud_score', ascending=False)
            
            # Add probable reasons for fraud
            def get_fraud_reasons(row):
                reasons = []
                # Check amount
                if 'amount' in row and row['amount'] > fraudulent_txns['amount'].mean() * 2:
                    reasons.append("Unusually high amount")
                
                # Check time
                if 'hour' in row and row['hour'] in [0, 1, 2, 3, 4, 5]:
                    reasons.append("Transaction during unusual hours")
                
                # Check merchant category
                if 'merchant_category' in row and row['merchant_category'] in ['gambling', 'cryptocurrency']:
                    reasons.append("High-risk merchant category")
                
                # Check payment method
                if 'payment_method' in row and row['payment_method'] in ['cryptocurrency', 'wire_transfer']:
                    reasons.append("High-risk payment method")
                
                # If no specific reasons found, add general reason
                return ", ".join(reasons) if reasons else "Multiple suspicious patterns"

            fraudulent_txns['probable_reasons'] = fraudulent_txns.apply(get_fraud_reasons, axis=1)
            
            # Display summary
            st.markdown(f"""
            #### Summary of Detected Fraud
            - **Total Fraudulent Transactions**: {len(fraudulent_txns)}
            - **Highest Fraud Score**: {fraudulent_txns['fraud_score'].max():.3f}
            - **Average Fraud Score**: {fraudulent_txns['fraud_score'].mean():.3f}
            """)
            
            # Display detailed table
            st.markdown("#### Detailed Analysis of Fraudulent Transactions")
            display_cols = ['amount', 'merchant_category', 'payment_method', 'fraud_score', 'probable_reasons']
            # Filter display_cols to only include columns that exist
            display_cols = [col for col in display_cols if col in fraudulent_txns.columns]
            
            st.dataframe(
                fraudulent_txns[display_cols].rename(columns={
                    'amount': 'Amount',
                    'merchant_category': 'Merchant Category',
                    'payment_method': 'Payment Method',
                    'fraud_score': 'Fraud Score',
                    'probable_reasons': 'Probable Reasons'
                }),
                use_container_width=True
            )
            
            # Add explanation of fraud detection
            st.markdown("""
            #### How Fraud is Detected
            The system identifies fraudulent transactions based on several factors:
            1. **Transaction Amount**: Unusually high amounts compared to normal transactions
            2. **Timing**: Transactions during unusual hours (midnight to 5 AM)
            3. **Merchant Category**: Transactions with high-risk merchants
            4. **Payment Method**: Use of high-risk payment methods
            5. **Pattern Changes**: Significant deviations from normal spending patterns
            
            The fraud score (0-1) indicates the probability of fraud, with higher scores indicating higher risk.
            """)
        else:
            st.success("No fraudulent transactions detected in this dataset!")

else:
    st.markdown("""
    ### Upload Your Data
    Upload a CSV file containing your transaction data. The file should include the following columns:
    - amount
    - merchant_category
    - payment_method
    - timestamp
    - location (as JSON with lat/lon)
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload Transaction Data (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            with st.spinner("Processing your data..."):
                # Read data
                data = pd.read_csv(uploaded_file)
                
                # Process data
                X = feature_engineer.prepare_features(data)
                predictions = model_trainer.predict(X)
                scores = model_trainer.get_fraud_score(X)
                
                # Add predictions to data
                results = data.copy()
                results['is_fraud'] = predictions
                results['fraud_score'] = scores
                
                # Display results
                st.subheader("📊 Analysis Results")
                st.markdown("""
                These metrics show the results of analyzing your uploaded transactions. They are different from the model performance metrics above.
                """)
                
                # Summary metrics
                st.markdown("### Transaction Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Transactions", len(results))
                with col2:
                    st.metric("Fraudulent Transactions", results['is_fraud'].sum())
                with col3:
                    st.metric("Average Fraud Score", 
                             f"{results['fraud_score'].mean():.3f}")
                
                # Display transactions
                st.markdown("### Transaction Details")
                st.markdown("Below is a detailed view of all transactions with their fraud predictions and scores.")
                st.dataframe(results)
                
                # Plot fraud scores distribution
                st.markdown("### Fraud Score Distribution")
                st.markdown("This chart shows the distribution of fraud scores across all transactions. Higher scores indicate higher probability of fraud.")
                st.bar_chart(results['fraud_score'].value_counts().sort_index())
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.markdown("""
            ### Troubleshooting
            Please ensure your CSV file:
            1. Contains all required columns
            2. Has valid data formats
            3. Doesn't contain any missing values
            """)

# Model information
st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
if model_trainer.is_trained:
    st.sidebar.success("✅ Model is trained and ready")
    st.sidebar.markdown("""
    The model has been trained and is ready to analyze transactions.
    You can see the model's performance metrics in the main view.
    """)
else:
    st.sidebar.warning("⚠️ Model needs training")
    st.sidebar.markdown("""
    The model needs to be trained before it can analyze transactions.
    Please contact the system administrator.
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit") 