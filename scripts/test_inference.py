#!/usr/bin/env python3
"""
Test Inference Script for XGBoost Model
This script tests the deployed XGBoost model via REST API
"""

import os
import sys
import json
import argparse
import requests
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import time

def generate_test_data(n_samples: int = 10) -> List[List[float]]:
    """Generate test data for inference"""
    np.random.seed(42)
    
    # Generate features (same as training data)
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    
    return X.tolist()

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

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Test XGBoost model inference')
    parser.add_argument('--model-url', type=str, required=True, help='Model service URL')
    parser.add_argument('--n-samples', type=int, default=10, help='Number of test samples')
    parser.add_argument('--wait-time', type=int, default=30, help='Wait time for service to be ready (seconds)')
    
    args = parser.parse_args()
    
    print(f"🧪 Testing XGBoost model inference at: {args.model_url}")
    
    # Wait for service to be ready
    print(f"⏳ Waiting {args.wait_time} seconds for service to be ready...")
    time.sleep(args.wait_time)
    
    # Test health endpoint
    print("🔍 Testing model health...")
    if test_model_health(args.model_url):
        print("✅ Model is healthy")
    else:
        print("❌ Model is not healthy")
        sys.exit(1)
    
    # Get model metadata
    print("📋 Getting model metadata...")
    metadata = test_model_metadata(args.model_url)
    if metadata:
        print(f"✅ Model metadata: {json.dumps(metadata, indent=2)}")
    
    # Generate test data
    print(f"📊 Generating {args.n_samples} test samples...")
    test_data = generate_test_data(args.n_samples)
    
    # Send prediction request
    print("🚀 Sending prediction request...")
    start_time = time.time()
    result = send_prediction_request(args.model_url, test_data)
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
        
        print("🎉 Model inference test completed successfully!")
    else:
        print("❌ Prediction failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 