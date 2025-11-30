"""
ML Integration Module for Flask
Provides simplified API for using ML models in Flask backend
"""

import logging
import os
import numpy as np
import json
from typing import Dict, Optional, List
from datetime import datetime

# Suppress verbose logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

class MLPipelineManager:
    """Manages ML pipeline for Flask integration."""
    
    def __init__(self, models_dir: str = "models/"):
        """
        Initialize ML pipeline manager.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = models_dir
        self.inference = None
        self.explainer = None
        self._initialize()
    
    def _initialize(self):
        """Initialize ML components with graceful fallback."""
        try:
            from .inference import EnsemblePredictor
            self.inference = EnsemblePredictor(models_dir=self.models_dir)
            logger.info("ML inference engine initialized")
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"ML modules not available ({type(e).__name__}) - predictions will use fallback")
            self.inference = None
        except Exception as e:
            logger.warning(f"Could not initialize ML pipeline: {e}")
            self.inference = None
    
    def predict(self, commodity: str, prices: List[float], 
               days: int = 7) -> Dict:
        """
        Make price prediction.
        
        Args:
            commodity: Commodity name
            prices: List of recent prices
            days: Number of days to predict
        
        Returns:
            Prediction dictionary
        """
        try:
            if not prices or len(prices) < 1:
                return {
                    "error": "No price data provided",
                    "commodity": commodity
                }
            
            prices_array = np.array(prices, dtype=float)
            
            if self.inference:
                # Use ML-based prediction
                result = self.inference.predict_ensemble(
                    commodity=commodity,
                    recent_prices=prices_array,
                    days=days
                )
                return result
            else:
                # Use simple fallback
                return self._fallback_predict(commodity, prices_array, days)
        
        except Exception as e:
            logger.error(f"Prediction error for {commodity}: {e}")
            return {
                "error": str(e),
                "commodity": commodity,
                "predictions": [prices[-1]] * days if prices else []
            }
    
    def batch_predict(self, price_data: Dict[str, List[float]], 
                     days: int = 7) -> Dict:
        """
        Make predictions for multiple commodities.
        
        Args:
            price_data: Dict mapping commodity names to price lists
            days: Number of days to predict
        
        Returns:
            Dictionary with predictions for all commodities
        """
        results = {}
        
        for commodity, prices in price_data.items():
            try:
                results[commodity] = self.predict(commodity, prices, days)
            except Exception as e:
                logger.error(f"Error predicting {commodity}: {e}")
                results[commodity] = {"error": str(e)}
        
        return results
    
    def _fallback_predict(self, commodity: str, prices: np.ndarray,
                         days: int) -> Dict:
        """
        Fallback prediction when ML unavailable.
        
        Args:
            commodity: Commodity name
            prices: Price array
            days: Days to predict
        
        Returns:
            Prediction dictionary
        """
        if len(prices) < 2:
            forecast = [float(prices[-1])] * days
        else:
            trend = (prices[-1] - prices[-7]) / 7 if len(prices) >= 7 else (prices[-1] - prices[0]) / len(prices)
            forecast = [float(prices[-1] + trend * (i+1)) for i in range(days)]
        
        return {
            "commodity": commodity,
            "predictions": forecast,
            "confidence": [0.4] * days,
            "model_used": "Fallback (trend)",
            "forecast_days": days,
            "warning": "Using fallback prediction - ML models not available"
        }
    
    def get_explanation(self, commodity: str, prediction: float,
                       prices: List[float]) -> Dict:
        """
        Get explanation for a prediction.
        
        Args:
            commodity: Commodity name
            prediction: Predicted price
            prices: Recent price history
        
        Returns:
            Explanation dictionary
        """
        try:
            if not self.inference:
                return self._simple_explanation(commodity, prediction, prices)
            
            prices_array = np.array(prices, dtype=float)
            explanation = self.inference.inference.explain_prediction(
                commodity=commodity,
                recent_prices=prices_array,
                prediction=prediction
            )
            return explanation
        
        except Exception as e:
            logger.warning(f"Error getting explanation: {e}")
            return self._simple_explanation(commodity, prediction, prices)
    
    def _simple_explanation(self, commodity: str, prediction: float,
                           prices: List[float]) -> Dict:
        """
        Generate simple explanation without ML.
        
        Args:
            commodity: Commodity name
            prediction: Predicted price
            prices: Recent price history
        
        Returns:
            Simple explanation
        """
        if not prices:
            return {"error": "No price history for explanation"}
        
        prices_array = np.array(prices, dtype=float)
        current = prices_array[-1]
        avg = prices_array.mean()
        
        explanation = {
            "commodity": commodity,
            "prediction": prediction,
            "current_price": float(current),
            "average_price": float(avg),
            "change_from_current": float(prediction - current),
            "change_percent": float((prediction - current) / current * 100) if current > 0 else 0,
            "factors": []
        }
        
        # Add trend factor
        if len(prices_array) >= 7:
            trend = prices_array[-1] - prices_array[-7]
            if trend > 0:
                explanation["factors"].append({
                    "name": "Upward trend (7-day)",
                    "value": float(trend),
                    "impact": "positive"
                })
            elif trend < 0:
                explanation["factors"].append({
                    "name": "Downward trend (7-day)",
                    "value": float(abs(trend)),
                    "impact": "negative"
                })
        
        # Add volatility factor
        if len(prices_array) >= 7:
            volatility = np.std(prices_array[-7:])
            explanation["factors"].append({
                "name": "Price volatility (7-day)",
                "value": float(volatility),
                "impact": "neutral"
            })
        
        return explanation
    
    def validate_prediction(self, commodity: str, prediction: float,
                           prices: List[float]) -> Dict:
        """
        Validate prediction reasonableness.
        
        Args:
            commodity: Commodity name
            prediction: Predicted price
            prices: Recent price history
        
        Returns:
            Validation result
        """
        try:
            prices_array = np.array(prices, dtype=float)
            
            if self.inference:
                validation = self.inference.inference.validate_prediction(
                    commodity=commodity,
                    prediction=prediction,
                    recent_prices=prices_array
                )
                return validation
            else:
                return self._simple_validation(prediction, prices_array)
        
        except Exception as e:
            logger.warning(f"Error validating prediction: {e}")
            return {"valid": True, "prediction": prediction}
    
    def _simple_validation(self, prediction: float,
                          prices: np.ndarray) -> Dict:
        """
        Simple validation without ML.
        
        Args:
            prediction: Predicted price
            prices: Recent price history
        
        Returns:
            Validation result
        """
        if len(prices) == 0:
            return {"valid": False, "reason": "No price history"}
        
        min_price = prices.min() * 0.7
        max_price = prices.max() * 1.3
        
        if prediction < min_price or prediction > max_price:
            return {
                "valid": False,
                "reason": f"Outside reasonable range [{min_price:.2f}, {max_price:.2f}]"
            }
        
        return {"valid": True}
    
    def get_model_info(self, commodity: Optional[str] = None) -> Dict:
        """
        Get information about available models.
        
        Args:
            commodity: Specific commodity (None for all)
        
        Returns:
            Model information
        """
        try:
            if not self.inference:
                return {"status": "ML models not available"}
            
            if commodity:
                return self.inference.inference.get_model_performance(commodity)
            else:
                return self.inference.inference.get_all_models_summary()
        
        except Exception as e:
            logger.warning(f"Error getting model info: {e}")
            return {"error": str(e)}
    
    def health_check(self) -> Dict:
        """
        Check ML pipeline health.
        
        Returns:
            Health status dictionary
        """
        status = {
            "timestamp": datetime.now().isoformat(),
            "ml_available": self.inference is not None,
            "components": {}
        }
        
        if self.inference:
            try:
                summary = self.inference.inference.get_all_models_summary()
                status["components"]["models"] = {
                    "count": len(summary),
                    "available": list(summary.keys())
                }
            except:
                status["components"]["models"] = {"error": "Could not retrieve models"}
        
        return status


# Global ML manager instance
_ml_manager = None

def get_ml_manager(models_dir: str = "models/") -> MLPipelineManager:
    """
    Get or create global ML manager instance.
    
    Args:
        models_dir: Directory containing trained models
    
    Returns:
        MLPipelineManager instance
    """
    global _ml_manager
    
    if _ml_manager is None:
        _ml_manager = MLPipelineManager(models_dir=models_dir)
    
    return _ml_manager

def init_ml_pipeline(app, models_dir: str = "models/"):
    """
    Initialize ML pipeline for Flask app.
    
    Args:
        app: Flask application instance
        models_dir: Directory containing trained models
    """
    global _ml_manager
    
    _ml_manager = MLPipelineManager(models_dir=models_dir)
    logger.info("ML pipeline initialized for Flask")
    
    # Add ML manager to app context
    app.ml_manager = _ml_manager

# Example Flask integration
def example_flask_integration():
    """
    Example of using ML manager in Flask app.
    """
    
    example_code = '''
    from flask import Flask, request, jsonify
    from ml_integration import init_ml_pipeline, get_ml_manager
    
    app = Flask(__name__)
    init_ml_pipeline(app, models_dir="models/")
    
    @app.route('/api/v1/predict', methods=['POST'])
    def predict():
        """Make price prediction using ML."""
        data = request.json
        commodity = data.get('commodity')
        prices = data.get('prices', [])
        days = data.get('days', 7)
        
        ml_manager = get_ml_manager()
        prediction = ml_manager.predict(commodity, prices, days)
        
        return jsonify(prediction)
    
    @app.route('/api/v1/explain', methods=['POST'])
    def explain():
        """Get explanation for prediction."""
        data = request.json
        commodity = data.get('commodity')
        prediction = data.get('prediction')
        prices = data.get('prices', [])
        
        ml_manager = get_ml_manager()
        explanation = ml_manager.get_explanation(commodity, prediction, prices)
        
        return jsonify(explanation)
    
    @app.route('/api/v1/ml-health', methods=['GET'])
    def ml_health():
        """Check ML pipeline health."""
        ml_manager = get_ml_manager()
        health = ml_manager.health_check()
        return jsonify(health)
    '''
    
    return example_code
