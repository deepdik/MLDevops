#!/usr/bin/env python3
"""
XGBoost Model Training Notebook
This script demonstrates the complete pipeline for training and deploying an XGBoost model
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import mlflow
import mlflow.xgboost
import boto3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("✅ All libraries imported successfully!")

# ============================================================================
# 1. Data Generation and Exploration
# ============================================================================

print("\n📊 Generating sample data...")

# Generate sample data
np.random.seed(42)
n_samples = 10000
n_features = 10

# Generate features
X = np.random.randn(n_samples, n_features)

# Generate target (regression problem)
coefficients = [2.0, 1.5, -0.8] + [0.1] * (n_features - 3)
y = np.dot(X, coefficients) + np.random.normal(0, 0.1, n_samples)

# Create DataFrame
feature_names = [f'feature_{i}' for i in range(n_features)]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print(f"📊 Dataset shape: {df.shape}")
print(f"🎯 Target variable: {df['target'].describe()}")
print("\nFirst 5 rows:")
print(df.head())

# ============================================================================
# 2. Data Preparation
# ============================================================================

print("\n🔧 Preparing data...")

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"📊 Training set: {X_train.shape}")
print(f"📊 Test set: {X_test.shape}")
print(f"🎯 Training target mean: {y_train.mean():.4f}")
print(f"🎯 Test target mean: {y_test.mean():.4f}")

# ============================================================================
# 3. XGBoost Model Training
# ============================================================================

print("\n🚀 Training XGBoost model...")

# Initialize MLflow
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
mlflow.set_experiment('xgboost-training')

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    }
    mlflow.log_params(params)
    
    # Initialize and train XGBoost model
    model = xgb.XGBRegressor(**params, n_jobs=-1)
    
    print("🚀 Training XGBoost model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    metrics = {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_rmse': np.sqrt(train_mse),
        'test_rmse': np.sqrt(test_mse),
        'train_r2': train_r2,
        'test_r2': test_r2
    }
    
    # Log metrics
    mlflow.log_metrics(metrics)
    
    # Log feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    mlflow.log_artifact(feature_importance.to_csv(index=False), "feature_importance.csv")
    
    # Log model
    mlflow.xgboost.log_model(model, "model")
    
    print("✅ Model training completed!")
    print(f"📊 Training R²: {train_r2:.4f}")
    print(f"📊 Test R²: {test_r2:.4f}")
    print(f"📊 Test RMSE: {np.sqrt(test_mse):.4f}")

# ============================================================================
# 4. Model Evaluation and Visualization
# ============================================================================

print("\n📈 Creating visualizations...")

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Actual vs Predicted (Training)
axes[0, 0].scatter(y_train, y_pred_train, alpha=0.6, color='blue')
axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Values')
axes[0, 0].set_ylabel('Predicted Values')
axes[0, 0].set_title(f'Training: Actual vs Predicted (R² = {train_r2:.4f})')

# Actual vs Predicted (Test)
axes[0, 1].scatter(y_test, y_pred_test, alpha=0.6, color='green')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual Values')
axes[0, 1].set_ylabel('Predicted Values')
axes[0, 1].set_title(f'Test: Actual vs Predicted (R² = {test_r2:.4f})')

# Residuals (Training)
residuals_train = y_train - y_pred_train
axes[1, 0].scatter(y_pred_train, residuals_train, alpha=0.6, color='blue')
axes[1, 0].axhline(y=0, color='r', linestyle='--')
axes[1, 0].set_xlabel('Predicted Values')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].set_title('Training: Residuals Plot')

# Residuals (Test)
residuals_test = y_test - y_pred_test
axes[1, 1].scatter(y_pred_test, residuals_test, alpha=0.6, color='green')
axes[1, 1].axhline(y=0, color='r', linestyle='--')
axes[1, 1].set_xlabel('Predicted Values')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Test: Residuals Plot')

plt.tight_layout()
plt.savefig('../models/model_evaluation.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature importance visualization
plt.figure(figsize=(10, 6))
feature_importance.plot(x='feature', y='importance', kind='barh', color='skyblue')
plt.title('XGBoost Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('../models/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("🔝 Top 5 most important features:")
print(feature_importance.head())

# ============================================================================
# 5. Model Deployment Preparation
# ============================================================================

print("\n💾 Saving model for deployment...")

# Save model for deployment
model_path = "../models/xgboost_model.json"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model.save_model(model_path)

print(f"💾 Model saved to: {model_path}")

# Save model to S3 if configured
s3_bucket = os.getenv('S3_BUCKET_NAME')
if s3_bucket:
    try:
        s3_client = boto3.client('s3')
        s3_key = f"models/xgboost_model_latest.json"
        s3_client.upload_file(model_path, s3_bucket, s3_key)
        print(f"☁️  Model uploaded to S3: s3://{s3_bucket}/{s3_key}")
    except Exception as e:
        print(f"⚠️  Failed to upload to S3: {e}")
else:
    print("⚠️  S3_BUCKET_NAME not set, skipping S3 upload")

# ============================================================================
# 6. Model Performance Summary
# ============================================================================

print("\n📊 XGBoost Model Performance Summary")
print("=" * 50)
print(f"Training R² Score: {train_r2:.4f}")
print(f"Test R² Score: {test_r2:.4f}")
print(f"Training RMSE: {np.sqrt(train_mse):.4f}")
print(f"Test RMSE: {np.sqrt(test_mse):.4f}")
print(f"Number of features: {X_train.shape[1]}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print("=" * 50)

print("\n🎯 Model is ready for deployment!")
print("Next steps:")
print("1. Deploy to Kubernetes using Seldon Core")
print("2. Set up monitoring and alerting")
print("3. Implement A/B testing")
print("4. Set up automated retraining pipeline") 