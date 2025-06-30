#!/usr/bin/env python3
"""
Kubeflow Pipeline for XGBoost Model Training and Deployment
This pipeline automates the entire ML workflow
"""

import kfp
from kfp import dsl
import os

# Define the data generation component
@dsl.component
def generate_data_op():
    """Generate synthetic data for training"""
    import numpy as np
    import pandas as pd
    import os
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 10000
    n_features = 10
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target with some non-linear relationships
    coefficients = np.random.randn(n_features) * 0.5
    y = np.dot(X, coefficients) + np.random.normal(0, 0.1, n_samples)
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Save data
    os.makedirs('/tmp/data', exist_ok=True)
    df.to_csv('/tmp/data/dataset.csv', index=False)
    
    print(f"Generated dataset with {df.shape[0]} samples and {df.shape[1]-1} features")
    return '/tmp/data/dataset.csv'

# Define the data preprocessing component
@dsl.component
def preprocess_data_op(data_path: str):
    """Preprocess the data for training"""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import joblib
    import os
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Save processed data
    os.makedirs('/tmp/processed', exist_ok=True)
    
    X_train.to_csv('/tmp/processed/X_train.csv', index=False)
    X_test.to_csv('/tmp/processed/X_test.csv', index=False)
    y_train.to_csv('/tmp/processed/y_train.csv', index=False)
    y_test.to_csv('/tmp/processed/y_test.csv', index=False)
    
    # Save preprocessing info
    preprocessing_info = {
        'feature_names': list(X_train.columns),
        'target_name': 'target',
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    joblib.dump(preprocessing_info, '/tmp/processed/preprocessing_info.pkl')
    
    print(f"Preprocessed data: {len(X_train)} training, {len(X_test)} test samples")
    return '/tmp/processed'

# Define the model training component
@dsl.component
def train_model_op(processed_data_path: str):
    """Train the XGBoost model"""
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib
    import os
    
    # Load processed data
    X_train = pd.read_csv(f'{processed_data_path}/X_train.csv')
    X_test = pd.read_csv(f'{processed_data_path}/X_test.csv')
    y_train = pd.read_csv(f'{processed_data_path}/y_train.csv')['target']
    y_test = pd.read_csv(f'{processed_data_path}/y_test.csv')['target']
    
    # Model parameters
    params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Train model
    model = xgb.XGBRegressor(**params)
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
        'train_rmse': train_mse ** 0.5,
        'test_rmse': test_mse ** 0.5,
        'train_r2': train_r2,
        'test_r2': test_r2
    }
    
    # Save model and metrics
    os.makedirs('/tmp/model', exist_ok=True)
    model.save_model('/tmp/model/xgboost_model.json')
    joblib.dump(model, '/tmp/model/xgboost_model.pkl')
    
    # Save metrics
    joblib.dump(metrics, '/tmp/model/metrics.pkl')
    
    print(f"Model trained successfully!")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test RMSE: {test_mse**0.5:.4f}")
    
    return '/tmp/model'

# Define the model evaluation component
@dsl.component
def evaluate_model_op(model_path: str, processed_data_path: str):
    """Evaluate the model and create visualizations"""
    import pandas as pd
    import xgboost as xgb
    import matplotlib.pyplot as plt
    import joblib
    import os
    
    # Load model and data
    model = xgb.XGBRegressor()
    model.load_model(f'{model_path}/xgboost_model.json')
    
    X_test = pd.read_csv(f'{processed_data_path}/X_test.csv')
    y_test = pd.read_csv(f'{processed_data_path}/y_test.csv')['target']
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Create evaluation plots
    os.makedirs('/tmp/evaluation', exist_ok=True)
    
    # Actual vs Predicted plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.tight_layout()
    plt.savefig('/tmp/evaluation/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature importance plot
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    feature_importance.plot(x='feature', y='importance', kind='barh')
    plt.title('XGBoost Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('/tmp/evaluation/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Model evaluation completed!")
    return '/tmp/evaluation'

# Define the model deployment component
@dsl.component
def deploy_model_op(model_path: str, processed_data_path: str):
    """Deploy the model to Kubernetes"""
    import subprocess
    import os
    import joblib
    
    # Load preprocessing info
    preprocessing_info = joblib.load(f'{processed_data_path}/preprocessing_info.pkl')
    
    # Create deployment manifest
    deployment_yaml = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: xgboost-pipeline-model
  namespace: kubeflow
  labels:
    app: xgboost-pipeline-model
spec:
  replicas: 1
  selector:
    matchLabels:
      app: xgboost-pipeline-model
  template:
    metadata:
      labels:
        app: xgboost-pipeline-model
    spec:
      containers:
      - name: xgboost-model
        image: python:3.9-slim
        ports:
        - containerPort: 8080
        env:
        - name: MODEL_PATH
          value: "/app/models/xgboost_model.pkl"
        volumeMounts:
        - name: model-volume
          mountPath: /app/models
        command: ["python", "-m", "http.server", "8080"]
      volumes:
      - name: model-volume
        configMap:
          name: xgboost-pipeline-model
---
apiVersion: v1
kind: Service
metadata:
  name: xgboost-pipeline-service
  namespace: kubeflow
spec:
  selector:
    app: xgboost-pipeline-model
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
"""
    
    # Save deployment manifest
    os.makedirs('/tmp/deployment', exist_ok=True)
    with open('/tmp/deployment/deployment.yaml', 'w') as f:
        f.write(deployment_yaml)
    
    print("Model deployment manifest created!")
    return '/tmp/deployment'

# Define the pipeline
@dsl.pipeline(
    name='XGBoost Training Pipeline',
    description='Automated XGBoost model training and deployment pipeline'
)
def xgboost_pipeline():
    """Main pipeline for XGBoost model training and deployment"""
    
    # Step 1: Generate data
    data_task = generate_data_op()
    
    # Step 2: Preprocess data
    preprocess_task = preprocess_data_op(data_path=data_task.outputs['output'])
    
    # Step 3: Train model
    train_task = train_model_op(processed_data_path=preprocess_task.outputs['output'])
    
    # Step 4: Evaluate model
    evaluate_task = evaluate_model_op(
        model_path=train_task.outputs['output'], 
        processed_data_path=preprocess_task.outputs['output']
    )
    
    # Step 5: Deploy model
    deploy_task = deploy_model_op(
        model_path=train_task.outputs['output'], 
        processed_data_path=preprocess_task.outputs['output']
    )
    
    # Set dependencies
    preprocess_task.after(data_task)
    train_task.after(preprocess_task)
    evaluate_task.after(train_task)
    deploy_task.after(train_task)

if __name__ == '__main__':
    # Compile the pipeline
    kfp.compiler.Compiler().compile(xgboost_pipeline, 'xgboost_pipeline.yaml')
    print("Pipeline compiled successfully!") 