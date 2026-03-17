import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import logging
from src.features.feature_engineering import FeatureEngineer
from src.models.model_trainer import ModelTrainer
from src.utils.data_stream import DataStream
from src.utils.database import Database
from src.config.config import MODEL_CONFIG, FEATURE_CONFIG, DB_CONFIG
import os
from src.utils.data_generator import generate_synthetic_data
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Advanced Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

def generate_synthetic_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic transaction data for training."""
    try:
        # Generate timestamps
        timestamps = [
            datetime.now() - timedelta(minutes=np.random.randint(0, 1000))
            for _ in range(n_samples)
        ]
        
        # Generate user IDs
        user_ids = [f"user_{np.random.randint(1, 100)}" for _ in range(n_samples)]
        
        # Generate amounts (using lognormal distribution for realistic amounts)
        amounts = np.round(np.random.lognormal(4, 1, n_samples), 2)
        
        # Generate merchant data
        merchants = [f"merchant_{np.random.randint(1, 50)}" for _ in range(n_samples)]
        merchant_categories = np.random.choice(
            ['retail', 'food', 'travel', 'entertainment'],
            size=n_samples
        )
        
        # Generate location data
        cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami']
        locations = [
            {
                'city': np.random.choice(cities),
                'lat': np.random.uniform(25, 45),
                'lon': np.random.uniform(-125, -70)
            }
            for _ in range(n_samples)
        ]
        
        # Generate payment methods
        payment_methods = np.random.choice(
            ['credit_card', 'debit_card', 'bank_transfer'],
            size=n_samples
        )
        
        # Generate device and IP data
        device_ids = [f"device_{np.random.randint(1, 100)}" for _ in range(n_samples)]
        ip_addresses = [
            f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
            for _ in range(n_samples)
        ]
        
        # Generate fraud labels (10% fraud rate)
        is_fraud = np.random.random(n_samples) < 0.1
        
        # Create DataFrame
        data = pd.DataFrame({
            'timestamp': timestamps,
            'user_id': user_ids,
            'amount': amounts,
            'merchant': merchants,
            'merchant_category': merchant_categories,
            'location': locations,
            'payment_method': payment_methods,
            'device_id': device_ids,
            'ip_address': ip_addresses,
            'is_fraud': is_fraud
        })
        
        # Sort by timestamp
        data = data.sort_values('timestamp')
        
        return data
        
    except Exception as e:
        logger.error(f"Error generating synthetic data: {str(e)}")
        raise

# Title and description
st.title("🔍 Advanced Real-time Fraud Detection System")
st.write("""
This system implements advanced fraud detection techniques including:
- Transaction velocity analysis
- Geographic anomaly detection
- Behavioral pattern analysis
- Merchant risk scoring
- Time-based pattern analysis
- Amount pattern analysis
""")

# Initialize components
feature_engineer = FeatureEngineer()
model_trainer = ModelTrainer()
data_stream = DataStream()
database = Database()

# Try to load existing model
try:
    model_trainer.load_model('models/fraud_detector.joblib')
    st.session_state.model = model_trainer.best_model
    logger.info("Loaded existing model successfully")
except Exception as e:
    logger.warning(f"No existing model found or error loading model: {str(e)}")
    st.session_state.model = None

# Initialize session state
if 'transactions' not in st.session_state:
    st.session_state.transactions = pd.DataFrame()
if 'is_streaming' not in st.session_state:
    st.session_state.is_streaming = False
if 'training_progress' not in st.session_state:
    st.session_state.training_progress = {
        'status': 'Not Started',
        'current_step': '',
        'progress': 0.0,
        'metrics': {}
    }
if 'model' not in st.session_state:
    st.session_state.model = None

# Sidebar controls
st.sidebar.title("Controls")
simulation_speed = st.sidebar.slider("Simulation Speed (transactions/second)", 1, 10, 1)
threshold = st.sidebar.slider("Fraud Detection Threshold", 0.0, 1.0, 0.95)

# Model training section
st.sidebar.markdown("---")
st.sidebar.subheader("Model Training")
if st.sidebar.button("Train Model"):
    try:
        # Initialize progress tracking
        progress = st.sidebar.progress(0.0)
        status_container = st.sidebar.empty()
        metrics_container = st.sidebar.empty()
        error_container = st.sidebar.empty()
        
        # Update status
        status_container.text("Loading data...")
        progress.progress(0.1)
        
        # Get historical data
        try:
            historical_data = database.get_all_transactions()
            if historical_data.empty:
                st.warning("No historical data found. Generating synthetic data...")
                historical_data = generate_synthetic_data(1000)
                progress.progress(0.2)
        except Exception as e:
            st.warning("Database error. Generating synthetic data...")
            historical_data = generate_synthetic_data(1000)
            progress.progress(0.2)
        
        # Update status
        status_container.text("Engineering features...")
        progress.progress(0.3)

        # Engineer features
        try:
            processed_data = feature_engineer.engineer_features(historical_data)
            progress.progress(0.4)
        except Exception as e:
            error_container.error(f"Error engineering features: {str(e)}")
            st.stop()
        
        # Update status
        status_container.text("Training model...")
        progress.progress(0.5)
        
        # Train model
        try:
            results = model_trainer.train(processed_data)
            progress.progress(0.8)
            
            # Check if any model was successfully trained
            if not any(result.get('status') == 'success' for result in results.values()):
                raise ValueError("No model was successfully trained")
            
            # Verify that the best model exists
            if model_trainer.best_model is None:
                raise ValueError("No best model was selected during training")
            
            # Update session state with the trained model
            st.session_state.model = model_trainer
            
        except Exception as e:
            error_container.error(f"Error training model: {str(e)}")
            st.stop()
        
        # Update status
        status_container.text("Saving model...")
        progress.progress(0.9)
        
        # Save model
        try:
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            # Verify model exists before saving
            if model_trainer.best_model is None:
                raise ValueError("No trained model available to save")
            
            # Save the model
            model_trainer.save_model('models/fraud_detector.joblib')
            progress.progress(1.0)
            status_container.text("Training completed!")
            
            # Display success message
            st.success("Model trained and saved successfully!")
            
        except Exception as e:
            error_container.error(f"Error saving model: {str(e)}")
            st.stop()
        
        # Display metrics
        metrics_df = pd.DataFrame()
        for model_name, result in results.items():
            if result.get('status') == 'success':
                metrics = pd.DataFrame({
                    'Model': [model_name],
                    'Precision': [result['test_metrics']['precision']],
                    'Recall': [result['test_metrics']['recall']],
                    'F1 Score': [result['test_metrics']['f1']]
                })
                metrics_df = pd.concat([metrics_df, metrics], ignore_index=True)
        
        if not metrics_df.empty:
            metrics_container.dataframe(metrics_df)
        else:
            metrics_container.warning("No models were successfully trained")
        
    except Exception as e:
        error_container.error(f"An unexpected error occurred: {str(e)}")
        st.stop()

# Main dashboard
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Transaction Monitor")
    transaction_placeholder = st.empty()
    chart_placeholder = st.empty()

with col2:
    st.subheader("Fraud Alerts")
    alerts_placeholder = st.empty()
    
    st.subheader("Statistics")
    stats_placeholder = st.empty()

# Start/Stop simulation
if st.sidebar.button("Start/Stop Simulation"):
    st.session_state.is_streaming = not st.session_state.is_streaming

# Simulation loop
async def run_simulation():
    while st.session_state.is_streaming:
        try:
            # Generate and process transaction
            transaction = await data_stream.generate_synthetic_transaction()
            processed_transaction = await data_stream.process_transaction(transaction)
            
            # Engineer features
            df = pd.DataFrame([processed_transaction])
            df = feature_engineer.engineer_features(df)
            
            # Predict fraud
            if st.session_state.model:
                predictions, scores = model_trainer.predict(df)
                score = scores[0]
                is_fraud = predictions[0] == -1  # For Isolation Forest
            else:
                score = 0.5
                is_fraud = False
            
            # Update transaction data
            processed_transaction['is_fraud'] = is_fraud
            processed_transaction['fraud_score'] = score
        
            # Store in database
            database.add_transaction(processed_transaction)
            
            # Update session state
            st.session_state.transactions = pd.concat([
                st.session_state.transactions,
                pd.DataFrame([processed_transaction])
            ]).reset_index(drop=True)
        
            # Update displays
            with transaction_placeholder:
                st.write("Latest Transaction:")
                st.write(processed_transaction)
        
            with chart_placeholder:
                fig = px.scatter(
                    st.session_state.transactions,
                    x='timestamp',
                    y='amount',
                    color='is_fraud',
                    title='Transaction Amounts Over Time'
                )
                st.plotly_chart(fig)
        
            with alerts_placeholder:
                if is_fraud:
                    st.error(f"🚨 Fraud Alert! Score: {score:.2f}")
                    st.write(f"Amount: ${processed_transaction['amount']:.2f}")
                    st.write(f"Merchant: {processed_transaction['merchant']}")
                    st.write(f"Location: {processed_transaction['location']['city']}")
        
            with stats_placeholder:
                stats = database.get_transaction_stats()
                st.metric("Total Transactions", stats['total_transactions'])
                st.metric("Fraudulent Transactions", stats['fraudulent_transactions'])
                st.metric("Fraud Rate", f"{stats['fraud_rate']:.2f}%")
        
            # Wait for next transaction
            await asyncio.sleep(1/simulation_speed)

        except Exception as e:
            logger.error(f"Error in simulation: {str(e)}")
            await asyncio.sleep(1)

# Run the simulation
if st.session_state.is_streaming:
    asyncio.run(run_simulation())

# Add data export functionality
if st.sidebar.button("Export Data"):
    try:
        transactions = database.get_user_transactions(limit=1000)
        df = pd.DataFrame([t.to_dict() for t in transactions])
        df.to_csv('transaction_history.csv', index=False)
        st.sidebar.success("Data exported successfully!")
    except Exception as e:
        st.sidebar.error(f"Error exporting data: {str(e)}")

# Add merchant analysis
st.sidebar.title("Merchant Analysis")
if st.sidebar.button("Show Merchant Stats"):
    try:
        merchant_stats = database.get_merchant_stats()
        st.sidebar.write("Merchant Statistics:")
        st.sidebar.json(merchant_stats)
    except Exception as e:
        st.sidebar.error(f"Error getting merchant stats: {str(e)}")

class FraudDetectionApp:
    def __init__(self):
        """Initialize the fraud detection application."""
        try:
            # Create necessary directories
            os.makedirs('models', exist_ok=True)
            
            # Initialize components
            self.db = Database()
            self.data_stream = DataStream()
            self.feature_engineer = FeatureEngineer()
            self.model_trainer = ModelTrainer()
            
            # Train initial model if needed
            if not self.model_trainer.is_trained:
                logger.info("Training initial model...")
                initial_data = generate_synthetic_data(n_samples=1000)
                
                # Prepare features
                X = self.feature_engineer.prepare_features(initial_data)
                y = initial_data['is_fraud']
                
                # Train model
                self.model_trainer.train(X, y)
            
            logger.info("Application initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing application: {str(e)}")
            raise

    async def process_transaction(self, transaction: dict) -> dict:
        """Process a single transaction with fraud detection."""
        try:
            # Add processing timestamp
            transaction['processed_at'] = datetime.now()
            
            # Convert transaction to DataFrame for feature engineering
            transaction_df = self.feature_engineer.prepare_single_transaction(transaction)
            
            # Get fraud prediction
            prediction = self.model_trainer.predict(transaction_df)
            fraud_score = self.model_trainer.get_fraud_score(transaction_df)
            
            # Add prediction results to transaction
            transaction['is_fraud'] = bool(prediction[0])
            transaction['fraud_score'] = float(fraud_score[0])
            
            # Ensure location is JSON string
            if isinstance(transaction.get('location'), dict):
                transaction['location'] = json.dumps(transaction['location'])
            
            # Convert timestamps to datetime if they're strings
            if isinstance(transaction.get('timestamp'), str):
                transaction['timestamp'] = datetime.fromisoformat(transaction['timestamp'])
            if isinstance(transaction.get('processed_at'), str):
                transaction['processed_at'] = datetime.fromisoformat(transaction['processed_at'])
            
            # Store transaction in database
            transaction_id = self.db.add_transaction(transaction)
            transaction['id'] = transaction_id
            
            return transaction
            
        except Exception as e:
            logger.error(f"Error processing transaction: {str(e)}")
            # Return transaction with default values
            transaction['is_fraud'] = False
            transaction['fraud_score'] = 0.0
            transaction['processed_at'] = datetime.now()
            return transaction

    async def process_transactions(self):
        """Process transactions from the stream."""
        while True:
            try:
                # Get transaction from stream
                transaction = await self.data_stream.transaction_queue.get()
                
                # Process transaction
                processed_transaction = await self.process_transaction(transaction)
                
                # Log result
                if processed_transaction['is_fraud']:
                    logger.warning(
                        f"Fraud detected! Transaction ID: {processed_transaction['id']}, "
                        f"Score: {processed_transaction['fraud_score']:.2f}"
                    )
                else:
                    logger.info(
                        f"Transaction processed. ID: {processed_transaction['id']}, "
                        f"Score: {processed_transaction['fraud_score']:.2f}"
                    )
                
                # Mark task as done
                self.data_stream.transaction_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in transaction processing loop: {str(e)}")
                # Don't raise the exception to keep the loop running
                continue

    async def run(self):
        """Run the fraud detection application."""
        try:
            # Start transaction streaming
            stream_task = asyncio.create_task(
                self.data_stream.stream_transactions(interval=1.0)
            )
            
            # Start transaction processing
            process_task = asyncio.create_task(self.process_transactions())
            
            # Wait for both tasks
            await asyncio.gather(stream_task, process_task)
            
        except Exception as e:
            logger.error(f"Error running application: {str(e)}")
            raise
        finally:
            # Cleanup
            if hasattr(self, 'db'):
                del self.db

async def main():
    """Main entry point for the application."""
    try:
        app = FraudDetectionApp()
        await app.run()
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        raise 