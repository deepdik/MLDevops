#!/usr/bin/env python3
"""
Custom XGBoost Model Server for Seldon Core
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from typing import Dict, List, Union, Any
from seldon_core.user_model import SeldonComponent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XGBoostModel(SeldonComponent):
    """
    Custom XGBoost model server for Seldon Core
    """
    
    def __init__(self, model_uri: str = None, model_dir: str = None):
        super().__init__()
        self.model = None
        self.preprocessing_info = None
        self.feature_names = None
        self.model_uri = model_uri
        self.model_dir = model_dir
        self.ready = False
        
    def load(self):
        """
        Load the XGBoost model and preprocessing info
        """
        try:
            # Load model
            if self.model_uri and self.model_uri.endswith('.pkl'):
                self.model = joblib.load(self.model_uri)
            elif self.model_uri and self.model_uri.endswith('.json'):
                self.model = xgb.XGBRegressor()
                self.model.load_model(self.model_uri)
            else:
                # Try to load from model directory
                model_path = os.path.join(self.model_dir, "xgboost_model.pkl")
                if os.path.exists(model_path):
                    self.model = joblib.load(model_path)
                else:
                    model_path = os.path.join(self.model_dir, "xgboost_model.json")
                    if os.path.exists(model_path):
                        self.model = xgb.XGBRegressor()
                        self.model.load_model(model_path)
            
            # Load preprocessing info
            preprocessing_path = os.path.join(self.model_dir, "preprocessing_info.pkl")
            if os.path.exists(preprocessing_path):
                self.preprocessing_info = joblib.load(preprocessing_path)
                self.feature_names = self.preprocessing_info.get('feature_names', [])
            
            self.ready = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.ready = False
    
    def predict(self, X: Union[np.ndarray, List, Dict], feature_names: List[str] = None) -> Union[np.ndarray, List, Dict]:
        """
        Make predictions using the loaded XGBoost model
        """
        if not self.ready:
            raise RuntimeError("Model not loaded")
        
        try:
            # Convert input to numpy array
            if isinstance(X, dict):
                # Handle dict input
                if 'data' in X:
                    X = X['data']
                elif 'instances' in X:
                    X = X['instances']
                else:
                    X = np.array(list(X.values())).reshape(1, -1)
            
            if isinstance(X, list):
                X = np.array(X)
            
            # Ensure 2D array
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            # Convert to DataFrame with proper feature names
            if self.feature_names and len(self.feature_names) == X.shape[1]:
                X_df = pd.DataFrame(X, columns=self.feature_names)
            else:
                X_df = pd.DataFrame(X)
            
            # Make prediction
            predictions = self.model.predict(X_df)
            
            # Return predictions
            if len(predictions) == 1:
                return float(predictions[0])
            else:
                return predictions.tolist()
                
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def health_status(self) -> Dict[str, Any]:
        """
        Return health status
        """
        return {
            "status": "healthy" if self.ready else "unhealthy",
            "model_loaded": self.ready
        }
    
    def metadata(self) -> Dict[str, Any]:
        """
        Return model metadata
        """
        if self.preprocessing_info:
            return {
                "name": "xgboost-model",
                "version": "1.0.0",
                "model_type": self.preprocessing_info.get('model_type', 'xgboost_regressor'),
                "feature_names": self.feature_names,
                "training_date": self.preprocessing_info.get('training_date', 'unknown'),
                "metrics": self.preprocessing_info.get('metrics', {})
            }
        else:
            return {
                "name": "xgboost-model",
                "version": "1.0.0",
                "model_type": "xgboost_regressor"
            }

# Create model instance
model = XGBoostModel()

# Load model on startup
model.load()

# Export the model for Seldon Core
def predict(X: Union[np.ndarray, List, Dict], feature_names: List[str] = None) -> Union[np.ndarray, List, Dict]:
    """
    Prediction function for Seldon Core
    """
    return model.predict(X, feature_names)

def health_status() -> Dict[str, Any]:
    """
    Health check function for Seldon Core
    """
    return model.health_status()

def metadata() -> Dict[str, Any]:
    """
    Metadata function for Seldon Core
    """
    return model.metadata() 