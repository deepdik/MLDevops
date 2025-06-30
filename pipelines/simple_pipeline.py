#!/usr/bin/env python3
"""
Simple Kubeflow Pipeline for XGBoost Model Training
"""

import kfp
from kfp import dsl

@dsl.component
def train_xgboost():
    """Simple XGBoost training component"""
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    import joblib
    import os
    
    print("🚀 Starting XGBoost training...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    coefficients = np.random.randn(n_features) * 0.5
    y = np.dot(X, coefficients) + np.random.normal(0, 0.1, n_samples)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    print(f"✅ Model trained successfully!")
    print(f"📊 R² Score: {r2:.4f}")
    
    # Save model
    os.makedirs('/tmp/model', exist_ok=True)
    model.save_model('/tmp/model/xgboost_model.json')
    
    return f"Model saved with R²: {r2:.4f}"

@dsl.component
def evaluate_model():
    """Model evaluation component"""
    import xgboost as xgb
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    print("📈 Evaluating model...")
    
    # Load model
    model = xgb.XGBRegressor()
    model.load_model('/tmp/model/xgboost_model.json')
    
    # Generate test data
    np.random.seed(123)
    X_test = np.random.randn(100, 5)
    y_test = np.dot(X_test, np.random.randn(5) * 0.5) + np.random.normal(0, 0.1, 100)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Create evaluation plot
    os.makedirs('/tmp/evaluation', exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('XGBoost Model: Actual vs Predicted')
    plt.tight_layout()
    plt.savefig('/tmp/evaluation/predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Model evaluation completed!")
    return "Evaluation plots saved"

@dsl.pipeline(
    name='Simple XGBoost Pipeline',
    description='A simple automated XGBoost training pipeline'
)
def simple_xgboost_pipeline():
    """Simple XGBoost training pipeline"""
    
    # Train the model
    train_task = train_xgboost()
    
    # Evaluate the model
    eval_task = evaluate_model()
    
    # Set dependency
    eval_task.after(train_task)

if __name__ == '__main__':
    # Compile the pipeline
    kfp.compiler.Compiler().compile(simple_xgboost_pipeline, 'simple_xgboost_pipeline.yaml')
    print("✅ Simple pipeline compiled successfully!") 