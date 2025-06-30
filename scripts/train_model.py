#!/usr/bin/env python3
"""
XGBoost Model Training Script
This script trains an XGBoost model and logs it to MLflow
"""

import os
import sys
import logging
import argparse
from typing import Dict, Any

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import xgboost as xgb
import mlflow
import mlflow.xgboost
import boto3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XGBoostTrainer:
    """XGBoost model trainer with MLflow integration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.metrics = {}
        
        # Initialize MLflow
        self._setup_mlflow()
        
    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
        mlflow.set_experiment(self.config.get('experiment_name', 'xgboost-training'))
        
    def generate_sample_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Generate sample data for demonstration"""
        np.random.seed(42)
        
        # Generate features
        n_features = 10
        X = np.random.randn(n_samples, n_features)
        
        # Generate target (regression problem)
        # y = 2*X[:,0] + 1.5*X[:,1] - 0.8*X[:,2] + noise
        coefficients = [2.0, 1.5, -0.8] + [0.1] * (n_features - 3)
        y = np.dot(X, coefficients) + np.random.normal(0, 0.1, n_samples)
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        return df
    
    def prepare_data(self, data: pd.DataFrame) -> tuple:
        """Prepare data for training"""
        # Split features and target
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
        """Train XGBoost model"""
        logger.info("Starting model training...")
        
        # Initialize model
        model = xgb.XGBRegressor(
            n_estimators=self.config.get('n_estimators', 100),
            max_depth=self.config.get('max_depth', 6),
            learning_rate=self.config.get('learning_rate', 0.1),
            random_state=42,
            n_jobs=-1
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        logger.info("Model training completed")
        return model
    
    def evaluate_model(self, model: xgb.XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance"""
        logger.info("Evaluating model...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2
        }
        
        logger.info(f"Model Performance - MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        return metrics
    
    def save_model_to_s3(self, model: xgb.XGBRegressor, model_name: str):
        """Save model to S3 bucket"""
        try:
            s3_client = boto3.client('s3')
            bucket_name = os.getenv('S3_BUCKET_NAME')
            
            if not bucket_name:
                logger.warning("S3_BUCKET_NAME not set, skipping S3 upload")
                return
            
            # Save model locally first
            local_path = f"/tmp/{model_name}.json"
            model.save_model(local_path)
            
            # Upload to S3
            s3_key = f"models/{model_name}.json"
            s3_client.upload_file(local_path, bucket_name, s3_key)
            
            logger.info(f"Model saved to S3: s3://{bucket_name}/{s3_key}")
            
            # Clean up local file
            os.remove(local_path)
            
        except Exception as e:
            logger.error(f"Failed to save model to S3: {e}")
    
    def run_training_pipeline(self):
        """Run complete training pipeline"""
        with mlflow.start_run():
            logger.info("Starting XGBoost training pipeline")
            
            # Log parameters
            mlflow.log_params(self.config)
            
            # Generate or load data
            if self.config.get('use_sample_data', True):
                data = self.generate_sample_data(self.config.get('n_samples', 10000))
                logger.info(f"Generated sample data with {len(data)} samples")
            else:
                # Load from file
                data_path = self.config.get('data_path')
                if not data_path:
                    raise ValueError("data_path must be provided when use_sample_data is False")
                data = pd.read_csv(data_path)
                logger.info(f"Loaded data from {data_path}")
            
            # Prepare data
            X_train, X_test, y_train, y_test = self.prepare_data(data)
            
            # Train model
            self.model = self.train_model(X_train, y_train)
            
            # Evaluate model
            self.metrics = self.evaluate_model(self.model, X_test, y_test)
            
            # Log metrics
            mlflow.log_metrics(self.metrics)
            
            # Log feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            mlflow.log_artifact(feature_importance.to_csv(index=False), "feature_importance.csv")
            
            # Log model
            mlflow.xgboost.log_model(self.model, "model")
            
            # Save model to S3 if configured
            if os.getenv('S3_BUCKET_NAME'):
                self.save_model_to_s3(self.model, f"xgboost_model_{mlflow.active_run().info.run_id}")
            
            logger.info("Training pipeline completed successfully")
            
            return self.model, self.metrics

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train XGBoost model')
    parser.add_argument('--n-estimators', type=int, default=100, help='Number of estimators')
    parser.add_argument('--max-depth', type=int, default=6, help='Maximum depth')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--n-samples', type=int, default=10000, help='Number of samples for synthetic data')
    parser.add_argument('--experiment-name', type=str, default='xgboost-training', help='MLflow experiment name')
    parser.add_argument('--data-path', type=str, help='Path to data file (optional)')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'n_samples': args.n_samples,
        'experiment_name': args.experiment_name,
        'use_sample_data': args.data_path is None,
        'data_path': args.data_path
    }
    
    # Initialize trainer
    trainer = XGBoostTrainer(config)
    
    # Run training pipeline
    try:
        model, metrics = trainer.run_training_pipeline()
        logger.info("Training completed successfully!")
        logger.info(f"Final metrics: {metrics}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 