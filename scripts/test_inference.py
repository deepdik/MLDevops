#!/usr/bin/env python3
"""
Test Inference Script for XGBoost Model
This script tests the XGBoost model either locally or via REST API
"""

import os
import sys
import json
import argparse
import requests
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from typing import List, Dict, Any
import time

def generate_test_data(n_samples: int = 10) -> List[List[float]]:
    """Generate test data for inference"""
    np.random.seed(42)
    
    # Generate features (same as training data)
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    
    return X.tolist()

def test_local_model():
    """Test the locally trained model"""
    print("🧪 Testing locally trained XGBoost model...")
    
    # Load the trained model
    model_path = "../models/xgboost_model.pkl"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Generate test data
    print("📊 Generating test data...")
    test_data = generate_test_data(10)
    X_test = np.array(test_data)
    
    # Make predictions
    print("🚀 Making predictions...")
    start_time = time.time()
    predictions = model.predict(X_test)
    end_time = time.time()
    
    print("✅ Local model test successful!")
    print(f"⏱️  Prediction time: {end_time - start_time:.4f} seconds")
    print(f"📈 Predictions: {predictions.tolist()}")
    
    # Basic statistics
    print(f"📊 Prediction statistics:")
    print(f"   Mean: {np.mean(predictions):.4f}")
    print(f"   Std: {np.std(predictions):.4f}")
    print(f"   Min: {np.min(predictions):.4f}")
    print(f"   Max: {np.max(predictions):.4f}")
    
    return True

def send_prediction_request(model_url: str, data: List[List[float]]) -> Dict[str, Any]:
    """Send prediction request to the deployed model"""
    
    # Prepare request payload
    payload = {
        "data": {
            "ndarray": data
        }
    }
    # Send request
    try:
        response = requests.post(
            f"{model_url}/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error sending prediction request: {e}")
        return None

def test_model_health(model_url: str) -> bool:
    """Test if the model is healthy"""
    try:
        response = requests.get(f"{model_url}/health/ready", timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def test_model_metadata(model_url: str) -> Dict[str, Any]:
    """Get model metadata"""
    try:
        response = requests.get(f"{model_url}/metadata", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting model metadata: {e}")
        return {}

def test_deployed_model(model_url: str, n_samples: int = 10, wait_time: int = 30):
    """Test deployed model via REST API"""
    print(f"🧪 Testing deployed XGBoost model at: {model_url}")
    
    # Wait for service to be ready
    print(f"⏳ Waiting {wait_time} seconds for service to be ready...")
    time.sleep(wait_time)
    
    # Test health endpoint
    print("🔍 Testing model health...")
    if test_model_health(model_url):
        print("✅ Model is healthy")
    else:
        print("❌ Model is not healthy")
        return False
    
    # Get model metadata
    print("📋 Getting model metadata...")
    metadata = test_model_metadata(model_url)
    if metadata:
        print(f"✅ Model metadata: {json.dumps(metadata, indent=2)}")
    
    # Generate test data
    print(f"📊 Generating {n_samples} test samples...")
    test_data = generate_test_data(n_samples)
    
    # Send prediction request
    print("🚀 Sending prediction request...")
    start_time = time.time()
    result = send_prediction_request(model_url, test_data)
    end_time = time.time()
    
    if result:
        print("✅ Prediction successful!")
        print(f"⏱️  Response time: {end_time - start_time:.2f} seconds")
        
        # Extract predictions
        predictions = result.get('data', {}).get('ndarray', [])
        print(f"📈 Predictions: {predictions}")
        
        # Basic statistics
        if predictions:
            pred_array = np.array(predictions)
            print(f"📊 Prediction statistics:")
            print(f"   Mean: {np.mean(pred_array):.4f}")
            print(f"   Std: {np.std(pred_array):.4f}")
            print(f"   Min: {np.min(pred_array):.4f}")
            print(f"   Max: {np.max(pred_array):.4f}")
        
        print("🎉 Deployed model inference test completed successfully!")
        return True
    else:
        print("❌ Prediction failed!")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Test XGBoost model inference')
    parser.add_argument('--model-url', type=str, help='Model service URL (optional for local testing)')
    parser.add_argument('--n-samples', type=int, default=10, help='Number of test samples')
    parser.add_argument('--wait-time', type=int, default=30, help='Wait time for service to be ready (seconds)')
    parser.add_argument('--local', action='store_true', help='Test local model only')
    
    args = parser.parse_args()
    
    # If no model URL provided or local flag is set, test locally
    if args.local or not args.model_url:
        success = test_local_model()
        if success:
            print("🎉 Local model test completed successfully!")
            sys.exit(0)
        else:
            print("❌ Local model test failed!")
            sys.exit(1)
    else:
        # Test deployed model
        success = test_deployed_model(args.model_url, args.n_samples, args.wait_time)
        if success:
            sys.exit(0)
        else:
            sys.exit(1)

if __name__ == "__main__":
    main() 