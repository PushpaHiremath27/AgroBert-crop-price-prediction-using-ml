"""
Inference Wrapper for ML Models
Load and use trained models for predictions with fallback support
"""

import numpy as np
import pandas as pd
import logging
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

# Suppress TensorFlow and other verbose logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logger = logging.getLogger(__name__)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Optional imports for ML
try:
    from .models import LSTMPricePredictor
    ML_AVAILABLE = True
except (ImportError, Exception):
    ML_AVAILABLE = False
    logging.debug("ML modules not fully available")

class ModelInference:
    """Unified interface for model inference with fallback support."""
    
    def __init__(self, models_dir: str = "models/"):
        """
        Initialize inference engine.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.metadata = {}
        self.fallback_enabled = True
        
        self._load_models()
    
    def _load_models(self):
        """Load all available trained models."""
        try:
            metadata_file = self.models_dir / "training_summary.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded model metadata from {metadata_file}")
        except Exception as e:
            logger.warning(f"Could not load model metadata: {e}")
    
    def predict_price(self, commodity: str, recent_prices: np.ndarray,
                     days: int = 7, use_lstm: bool = True) -> Dict:
        """
        Predict future prices for a commodity.
        
        Args:
            commodity: Commodity name
            recent_prices: Array of recent prices (at least 30 days)
            days: Number of days to predict (default 7)
            use_lstm: Use LSTM if available, fallback to ARIMA/simulated
        
        Returns:
            Dictionary with predictions and metadata
        """
        predictions = []
        confidence = []
        model_used = None
        
        # Try LSTM first if requested
        if use_lstm:
            result = self._predict_lstm(commodity, recent_prices, days)
            if result and not result.get("error"):
                return result
        
        # Fallback to simpler methods
        predictions, model_used = self._fallback_predict(
            recent_prices, days
        )
        
        # Calculate confidence based on data quality
        conf_score = self._calculate_confidence(recent_prices)
        confidence = [conf_score] * days
        
        return {
            "commodity": commodity,
            "predictions": predictions,
            "confidence": confidence,
            "model_used": model_used,
            "forecast_days": days,
            "timestamp": datetime.now().isoformat(),
            "warning": "Using fallback prediction - LSTM model not available"
        }
    
    def _predict_lstm(self, commodity: str, recent_prices: np.ndarray,
                     days: int) -> Optional[Dict]:
        """
        Make LSTM predictions.
        
        Args:
            commodity: Commodity name
            recent_prices: Recent price history
            days: Days to predict
        
        Returns:
            Prediction dictionary or None if unavailable
        """
        try:
            if not ML_AVAILABLE:
                logger.debug("ML modules not available for LSTM inference")
                return None
            
            from models import LSTMPricePredictor
            
            model_path = self.models_dir / f"lstm_{commodity.lower()}.h5"
            if not model_path.exists():
                logger.warning(f"LSTM model not found: {model_path}")
                return None
            
            # Load model
            model = LSTMPricePredictor(lookback=30, output_steps=1)
            model.load(str(model_path).replace('.h5', ''))
            
            # Make prediction
            if len(recent_prices) < 30:
                logger.warning(f"Insufficient data for LSTM. Need 30 days, got {len(recent_prices)}")
                return None
            
            predictions = model.predict_next_days(recent_prices, days)
            
            return {
                "commodity": commodity,
                "predictions": predictions.tolist(),
                "confidence": [0.8] * days,  # LSTM confidence
                "model_used": "LSTM",
                "forecast_days": days,
                "timestamp": datetime.now().isoformat()
            }
        
        except ImportError as e:
            logger.debug(f"ImportError in LSTM prediction: {e}")
            return None
        except Exception as e:
            logger.warning(f"LSTM prediction error: {e}")
            return None
    
    def _fallback_predict(self, prices: np.ndarray, days: int) -> Tuple[list, str]:
        """
        Fallback prediction using simple methods.
        
        Args:
            prices: Price history
            days: Days to predict
        
        Returns:
            Tuple of (predictions list, model name)
        """
        if len(prices) < 2:
            # Single point - just repeat it
            return [float(prices[-1])] * days, "static"
        
        # Method 1: Simple trend continuation
        recent_trend = prices[-1] - prices[-7] if len(prices) >= 7 else prices[-1] - prices[-2]
        daily_change = recent_trend / max(7, len(prices) - 1) if len(prices) > 1 else 0
        
        predictions = []
        current = float(prices[-1])
        
        for i in range(days):
            current = current + daily_change
            predictions.append(current)
        
        return predictions, "trend_continuation"
    
    def _calculate_confidence(self, prices: np.ndarray) -> float:
        """
        Calculate prediction confidence based on data quality.
        
        Args:
            prices: Price array
        
        Returns:
            Confidence score (0-1)
        """
        if len(prices) < 7:
            return 0.3  # Low confidence for limited data
        
        if len(prices) < 30:
            return 0.5  # Medium confidence
        
        # Check price stability
        recent_prices = prices[-7:]
        volatility = np.std(recent_prices) / np.mean(recent_prices) if np.mean(recent_prices) > 0 else 0
        
        # Lower confidence for high volatility
        if volatility > 0.1:  # 10% std dev relative to mean
            return 0.6
        
        return 0.8
    
    def predict_batch(self, commodities: list, price_data: Dict[str, np.ndarray],
                     days: int = 7) -> Dict:
        """
        Make predictions for multiple commodities.
        
        Args:
            commodities: List of commodity names
            price_data: Dictionary mapping commodity to price arrays
            days: Days to predict
        
        Returns:
            Dictionary with predictions for all commodities
        """
        results = {}
        
        for commodity in commodities:
            if commodity not in price_data:
                results[commodity] = {"error": "Price data not found"}
                continue
            
            prices = np.array(price_data[commodity])
            result = self.predict_price(commodity, prices, days)
            results[commodity] = result
        
        return results
    
    def get_model_performance(self, commodity: str) -> Dict:
        """
        Get performance metrics for a specific model.
        
        Args:
            commodity: Commodity name
        
        Returns:
            Dictionary with model performance metrics
        """
        if not self.metadata or "models" not in self.metadata:
            return {"error": "No model metadata available"}
        
        model_key = f"lstm_{commodity.lower()}"
        
        for key, meta in self.metadata.get("models", {}).items():
            if key == model_key:
                metrics = meta.get("metrics", {})
                return {
                    "commodity": commodity,
                    "model": meta.get("type", "unknown"),
                    "rmse": metrics.get("rmse"),
                    "mae": metrics.get("mae"),
                    "mape": metrics.get("mape"),
                    "trained_at": meta.get("saved_at")
                }
        
        return {"error": f"No performance data for {commodity}"}
    
    def get_all_models_summary(self) -> Dict:
        """
        Get summary of all available models.
        
        Returns:
            Dictionary with all model summaries
        """
        summary = {}
        
        if not self.metadata or "models" not in self.metadata:
            return {"error": "No models available"}
        
        for key, meta in self.metadata.get("models", {}).items():
            summary[key] = {
                "type": meta.get("type"),
                "commodity": meta.get("commodity"),
                "metrics": meta.get("metrics", {}),
                "trained_at": meta.get("saved_at")
            }
        
        return summary
    
    def explain_prediction(self, commodity: str, recent_prices: np.ndarray,
                          prediction: float) -> Dict:
        """
        Provide explanation for a prediction.
        
        Args:
            commodity: Commodity name
            recent_prices: Recent price history
            prediction: The prediction made
        
        Returns:
            Dictionary with explanation
        """
        explanation = {
            "commodity": commodity,
            "prediction": prediction,
            "factors": []
        }
        
        if len(recent_prices) >= 7:
            # Recent trend
            recent_trend = recent_prices[-1] - recent_prices[-7]
            if recent_trend > 0:
                explanation["factors"].append({
                    "factor": "Recent upward trend",
                    "impact": "positive",
                    "magnitude": float(recent_trend),
                    "description": f"Price increased by {recent_trend:.2f} over last 7 days"
                })
            elif recent_trend < 0:
                explanation["factors"].append({
                    "factor": "Recent downward trend",
                    "impact": "negative",
                    "magnitude": float(abs(recent_trend)),
                    "description": f"Price decreased by {abs(recent_trend):.2f} over last 7 days"
                })
            
            # Volatility
            volatility = np.std(recent_prices[-7:])
            explanation["factors"].append({
                "factor": "Price volatility",
                "impact": "neutral",
                "magnitude": float(volatility),
                "description": f"Standard deviation: {volatility:.2f}"
            })
        
        # Price level
        current_price = recent_prices[-1]
        explanation["factors"].append({
            "factor": "Current price",
            "impact": "neutral",
            "magnitude": float(current_price),
            "description": f"Last recorded price: {current_price:.2f}"
        })
        
        return explanation
    
    def validate_prediction(self, commodity: str, prediction: float,
                           recent_prices: np.ndarray) -> Dict:
        """
        Validate if a prediction is reasonable.
        
        Args:
            commodity: Commodity name
            prediction: Predicted price
            recent_prices: Recent price history
        
        Returns:
            Dictionary with validation results
        """
        if len(recent_prices) == 0:
            return {"valid": False, "reason": "No price history available"}
        
        current_price = recent_prices[-1]
        avg_price = np.mean(recent_prices)
        
        # Check if prediction is within reasonable bounds
        min_price = np.min(recent_prices) * 0.7
        max_price = np.max(recent_prices) * 1.3
        
        if prediction < min_price or prediction > max_price:
            return {
                "valid": False,
                "reason": f"Prediction outside reasonable range [{min_price:.2f}, {max_price:.2f}]",
                "prediction": prediction
            }
        
        # Check for reasonable change from current price
        price_change_pct = abs(prediction - current_price) / current_price * 100 if current_price > 0 else 0
        
        if price_change_pct > 50:  # More than 50% change in 1 day is suspicious
            return {
                "valid": False,
                "reason": f"Unreasonable price change: {price_change_pct:.1f}%",
                "prediction": prediction,
                "current_price": float(current_price)
            }
        
        return {
            "valid": True,
            "prediction": prediction,
            "current_price": float(current_price),
            "expected_change_pct": float(price_change_pct)
        }


class EnsemblePredictor:
    """Ensemble predictions from multiple models."""
    
    def __init__(self, models_dir: str = "models/"):
        """
        Initialize ensemble predictor.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.inference = ModelInference(models_dir)
    
    def predict_ensemble(self, commodity: str, recent_prices: np.ndarray,
                        days: int = 7) -> Dict:
        """
        Make ensemble predictions combining multiple models.
        
        Args:
            commodity: Commodity name
            recent_prices: Recent price history
            days: Days to predict
        
        Returns:
            Dictionary with ensemble predictions
        """
        # Get individual predictions
        lstm_pred = self._try_lstm_predict(commodity, recent_prices, days)
        fallback_pred = self.inference._fallback_predict(recent_prices, days)[0]
        
        # Combine predictions
        if lstm_pred:
            ensemble = [
                0.7 * lstm_val + 0.3 * fallback_val
                for lstm_val, fallback_val in zip(lstm_pred, fallback_pred)
            ]
            weight_info = "LSTM (70%) + Fallback (30%)"
        else:
            ensemble = fallback_pred
            weight_info = "Fallback only (LSTM unavailable)"
        
        confidence = self.inference._calculate_confidence(recent_prices)
        
        return {
            "commodity": commodity,
            "predictions": ensemble,
            "confidence": [confidence] * days,
            "model_used": "Ensemble",
            "weights": weight_info,
            "forecast_days": days,
            "timestamp": datetime.now().isoformat()
        }
    
    def _try_lstm_predict(self, commodity: str, recent_prices: np.ndarray,
                         days: int) -> Optional[list]:
        """
        Try to get LSTM predictions.
        
        Args:
            commodity: Commodity name
            recent_prices: Recent price history
            days: Days to predict
        
        Returns:
            List of LSTM predictions or None
        """
        try:
            result = self.inference._predict_lstm(commodity, recent_prices, days)
            if result and "predictions" in result:
                return result["predictions"]
        except:
            pass
        return None
