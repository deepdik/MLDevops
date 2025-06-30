#!/usr/bin/env python3
"""
Simple Flask-based XGBoost Model Server
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from flask import Flask, request, jsonify
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model and preprocessing info
model = None
preprocessing_info = None
feature_names = None

def load_model():
    """Load the XGBoost model and preprocessing info"""
    global model, preprocessing_info, feature_names
    
    try:
        # Load model
        model_path = os.getenv('MODEL_PATH', '/app/models/xgboost_model.pkl')
        preprocessing_path = os.getenv('PREPROCESSING_PATH', '/app/models/preprocessing_info.pkl')
        
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Loading preprocessing info from: {preprocessing_path}")
        
        if model_path.endswith('.pkl'):
            model = joblib.load(model_path)
        elif model_path.endswith('.json'):
            model = xgb.XGBRegressor()
            model.load_model(model_path)
        
        # Load preprocessing info
        if os.path.exists(preprocessing_path):
            preprocessing_info = joblib.load(preprocessing_path)
            feature_names = preprocessing_info.get('feature_names', [])
            logger.info(f"Loaded preprocessing info with {len(feature_names)} features")
        
        logger.info("Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def predict_single(features):
    """Make prediction for a single sample"""
    try:
        # Convert to numpy array
        if isinstance(features, dict):
            # Handle dict input
            if feature_names:
                # Use feature names to order the input
                ordered_features = []
                for feature_name in feature_names:
                    if feature_name in features:
                        ordered_features.append(features[feature_name])
                    else:
                        ordered_features.append(0.0)  # Default value
                X = np.array(ordered_features).reshape(1, -1)
            else:
                X = np.array(list(features.values())).reshape(1, -1)
        else:
            X = np.array(features).reshape(1, -1)
        
        # Convert to DataFrame
        if feature_names and len(feature_names) == X.shape[1]:
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = pd.DataFrame(X)
        
        # Make prediction
        prediction = model.predict(X_df)[0]
        return float(prediction)
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'feature_names': feature_names
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Handle different input formats
        if 'instances' in data:
            # Batch prediction
            instances = data['instances']
            predictions = []
            for instance in instances:
                pred = predict_single(instance)
                predictions.append(pred)
            return jsonify({'predictions': predictions})
        
        elif 'data' in data:
            # Single prediction with 'data' key
            prediction = predict_single(data['data'])
            return jsonify({'prediction': prediction})
        
        else:
            # Single prediction with direct features
            prediction = predict_single(data)
            return jsonify({'prediction': prediction})
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/metadata', methods=['GET'])
def metadata():
    """Model metadata endpoint"""
    if preprocessing_info:
        return jsonify({
            'name': 'xgboost-model',
            'version': '1.0.0',
            'model_type': preprocessing_info.get('model_type', 'xgboost_regressor'),
            'feature_names': feature_names,
            'training_date': preprocessing_info.get('training_date', 'unknown'),
            'metrics': preprocessing_info.get('metrics', {})
        })
    else:
        return jsonify({
            'name': 'xgboost-model',
            'version': '1.0.0',
            'model_type': 'xgboost_regressor'
        })

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'message': 'XGBoost Model Server',
        'endpoints': {
            'health': '/health',
            'predict': '/predict',
            'metadata': '/metadata'
        },
        'usage': {
            'single_prediction': {
                'method': 'POST',
                'url': '/predict',
                'body': {'feature_0': 0.1, 'feature_1': 0.2, ...}
            },
            'batch_prediction': {
                'method': 'POST',
                'url': '/predict',
                'body': {'instances': [{'feature_0': 0.1, ...}, ...]}
            }
        }
    })

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        logger.info("Starting XGBoost Model Server...")
        port = int(os.environ.get('PORT', 8080))
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        logger.error("Failed to load model. Exiting.")
        exit(1) 