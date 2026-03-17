import joblib
from src.models.model_trainer import ModelTrainer
from src.utils.data_generator import generate_synthetic_data
from src.features.feature_engineering import FeatureEngineer

def update_model():
    # Generate synthetic data
    print("Generating synthetic data...")
    data = generate_synthetic_data(n_samples=5000, fraud_ratio=0.5)
    
    # Prepare features
    print("Preparing features...")
    feature_engineer = FeatureEngineer()
    X = feature_engineer.prepare_features(data)
    y = data['is_fraud']
    
    # Train model
    print("Training model...")
    trainer = ModelTrainer()
    trainer.train(X, y)
    
    # Save model with metrics
    print("Saving model and metrics...")
    model_data = {
        'model': trainer.model,
        'metrics': {
            'f1_score': trainer.f1_score,
            'precision': trainer.precision,
            'recall': trainer.recall,
            'auc': trainer.auc
        }
    }
    joblib.dump(model_data, 'models/fraud_detector.joblib')
    
    print("Model updated successfully!")
    print(f"F1 Score: {trainer.f1_score:.3f}")
    print(f"Precision: {trainer.precision:.3f}")
    print(f"Recall: {trainer.recall:.3f}")
    print(f"AUC-ROC: {trainer.auc:.3f}")

if __name__ == "__main__":
    update_model() 