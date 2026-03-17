from src.utils.data_generator import generate_synthetic_data
from src.models.model_trainer import ModelTrainer
from src.features.feature_engineering import FeatureEngineer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model():
    try:
        # Generate synthetic data with balanced fraud ratio
        logger.info('Generating synthetic data...')
        data = generate_synthetic_data(n_samples=5000, fraud_ratio=0.5)
        
        # Prepare features and target
        logger.info('Preparing features...')
        feature_engineer = FeatureEngineer()
        X = feature_engineer.prepare_features(data)
        y = data['is_fraud']
        
        # Train the model
        logger.info('Training the model...')
        trainer = ModelTrainer()
        trainer.train(X, y)
        
        # Print metrics
        logger.info('Training complete. Model metrics:')
        logger.info(f"F1 Score: {trainer.f1_score:.3f}")
        logger.info(f"Precision: {trainer.precision:.3f}")
        logger.info(f"Recall: {trainer.recall:.3f}")
        logger.info(f"AUC-ROC: {trainer.auc:.3f}")
        
        return trainer
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model() 