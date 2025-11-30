"""
LSTM Neural Network for Price Prediction
Time-series forecasting using TensorFlow/Keras
"""

import numpy as np
import pandas as pd
import logging
import os
from typing import Tuple, Dict, Optional
import json
from pathlib import Path
from datetime import datetime

# Suppress TensorFlow and other verbose logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO, WARNING, and ERROR
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# TensorFlow/Keras imports with fallback
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf 
    tf.get_logger().setLevel('ERROR')
    from tensorflow import keras 
    from tensorflow.keras import layers, models 
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau 
    TENSORFLOW_AVAILABLE = True
except Exception as e:
    logging.debug(f"TensorFlow not available ({type(e).__name__}) - using numpy fallback")

logger = logging.getLogger(__name__)

class LSTMPricePredictor:
    """LSTM-based price prediction model."""
    
    def __init__(self, lookback: int = 30, output_steps: int = 7):
        """
        Initialize LSTM model.
        
        Args:
            lookback: Number of past days to use for prediction
            output_steps: Number of days ahead to predict
        """
        self.lookback = lookback
        self.output_steps = output_steps
        self.model = None
        self.history = None
        self.is_trained = False
        
        if TENSORFLOW_AVAILABLE:
            self._build_model()
    
    def _build_model(self):
        """Build LSTM architecture."""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("Cannot build model without TensorFlow")
            return
        
        self.model = models.Sequential([
            # First LSTM layer with dropout
            layers.LSTM(64, activation='relu', input_shape=(self.lookback, 1),
                       return_sequences=True),
            layers.Dropout(0.2),
            
            # Second LSTM layer with dropout
            layers.LSTM(32, activation='relu', return_sequences=False),
            layers.Dropout(0.2),
            
            # Dense layers
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.1),
            
            # Output layer
            layers.Dense(self.output_steps)
        ])
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        logger.info("LSTM model built successfully")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 100, batch_size: int = 32) -> Dict:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training input sequences (samples, lookback, 1)
            y_train: Training target values
            X_val: Validation input sequences
            y_val: Validation target values
            epochs: Number of training epochs
            batch_size: Batch size for training
        
        Returns:
            Training history dictionary
        """
        if not TENSORFLOW_AVAILABLE or self.model is None:
            logger.error("Model not available for training")
            return {"error": "TensorFlow not available"}
        
        # Ensure 3D shape for LSTM input
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        if X_val is not None and len(X_val.shape) == 2:
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        logger.info(f"Model training completed. Final loss: {self.history.history['loss'][-1]:.4f}")
        
        return {
            "epochs_trained": len(self.history.history['loss']),
            "final_loss": float(self.history.history['loss'][-1]),
            "final_mae": float(self.history.history['mae'][-1]),
            "success": True
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on input sequences.
        
        Args:
            X: Input sequences (samples, lookback) or (samples, lookback, 1)
        
        Returns:
            Predictions array (samples, output_steps)
        """
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained. Using fallback prediction.")
            return self._fallback_predict(X)
        
        # Ensure 3D shape
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        predictions = self.model.predict(X, verbose=0)
        return predictions
    
    def _fallback_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fallback prediction using simple trend continuation.
        Used when TensorFlow is not available.
        
        Args:
            X: Input sequences
        
        Returns:
            Predicted values
        """
        if len(X.shape) == 3:
            X = X.squeeze()
        
        predictions = []
        for sequence in X:
            # Simple linear extrapolation
            if len(sequence) >= 2:
                trend = sequence[-1] - sequence[-2]
                pred = [sequence[-1] + trend * (i + 1) for i in range(self.output_steps)]
            else:
                pred = [sequence[-1]] * self.output_steps
            predictions.append(pred)
        
        return np.array(predictions)
    
    def predict_next_days(self, recent_prices: np.ndarray, days: int = 7) -> np.ndarray:
        """
        Predict prices for next N days from recent price history.
        
        Args:
            recent_prices: Array of recent prices (at least lookback days)
            days: Number of days to predict
        
        Returns:
            Predicted prices for next days
        """
        if len(recent_prices) < self.lookback:
            logger.warning(f"Insufficient data. Need at least {self.lookback} days")
            return np.array([recent_prices[-1]] * days)
        
        # Use last lookback days as input
        X = recent_prices[-self.lookback:].reshape(1, -1)
        
        # Predict
        predictions = self.predict(X)[0]
        
        # Return only requested days
        return predictions[:days]
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test input sequences
            y_test: Test target values
        
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained or self.model is None:
            return {"error": "Model not trained"}
        
        if len(X_test.shape) == 2:
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        loss, mae, mape = self.model.evaluate(X_test, y_test, verbose=0)
        predictions = self.predict(X_test)
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        
        # Calculate MAPE manually for all outputs
        mape_all = np.mean(np.abs((y_test - predictions) / (np.abs(y_test) + 1))) * 100
        
        metrics = {
            "loss": float(loss),
            "mae": float(mae),
            "mape": float(mape),
            "rmse": float(rmse),
            "mape_calculated": float(mape_all)
        }
        
        logger.info(f"Model evaluation - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape_all:.2f}%")
        return metrics
    
    def save(self, filepath: str):
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save model (without extension)
        """
        if not TENSORFLOW_AVAILABLE or self.model is None:
            logger.warning("Cannot save model without TensorFlow")
            return
        
        try:
            self.model.save(f"{filepath}.h5")
            
            # Save metadata
            metadata = {
                "lookback": self.lookback,
                "output_steps": self.output_steps,
                "is_trained": self.is_trained,
                "saved_at": datetime.now().isoformat()
            }
            with open(f"{filepath}_meta.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model saved to {filepath}.h5")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load(self, filepath: str):
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to saved model (without extension)
        """
        if not TENSORFLOW_AVAILABLE:
            logger.warning("Cannot load model without TensorFlow")
            return
        
        try:
            self.model = keras.models.load_model(f"{filepath}.h5")
            self.is_trained = True
            
            # Load metadata
            try:
                with open(f"{filepath}_meta.json", 'r') as f:
                    metadata = json.load(f)
                    self.lookback = metadata.get("lookback", self.lookback)
                    self.output_steps = metadata.get("output_steps", self.output_steps)
            except:
                pass
            
            logger.info(f"Model loaded from {filepath}.h5")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if self.model is None:
            return "No model built"
        
        from io import StringIO
        stream = StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()


class SimpleARIMAPredictor:
    """
    Simple ARIMA-like predictor using exponential smoothing.
    Fallback when TensorFlow is not available.
    """
    
    def __init__(self, alpha: float = 0.3):
        """
        Initialize simple exponential smoothing predictor.
        
        Args:
            alpha: Smoothing factor (0-1)
        """
        self.alpha = alpha
        self.last_value = None
        self.trend = None
    
    def fit(self, data: np.ndarray):
        """
        Fit the predictor to historical data.
        
        Args:
            data: Historical price data
        """
        if len(data) < 2:
            self.last_value = data[-1]
            self.trend = 0
            return
        
        # Simple exponential smoothing
        smoothed = data[0]
        for val in data[1:]:
            smoothed = self.alpha * val + (1 - self.alpha) * smoothed
        
        self.last_value = smoothed
        self.trend = data[-1] - data[-2]  # Simple trend from last two points
    
    def predict(self, steps: int = 7) -> np.ndarray:
        """
        Predict future values.
        
        Args:
            steps: Number of steps ahead to predict
        
        Returns:
            Predicted values
        """
        if self.last_value is None:
            return np.array([0] * steps)
        
        predictions = []
        current = self.last_value
        
        for i in range(steps):
            current = current + self.trend
            predictions.append(current)
        
        return np.array(predictions)
