"""
Model Explainability using SHAP
Feature importance and prediction explanations
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
import json

# SHAP imports with fallback
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available - using feature importance fallback")

logger = logging.getLogger(__name__)

class ModelExplainer:
    """Explain model predictions using SHAP or fallback methods."""
    
    def __init__(self, model=None, model_type: str = "lstm"):
        """
        Initialize explainer.
        
        Args:
            model: Trained model to explain
            model_type: Type of model ('lstm', 'rf', 'linear', etc.)
        """
        self.model = model
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        self.base_value = None
    
    def explain_prediction(self, X: np.ndarray, y_pred: np.ndarray,
                          sample_index: int = 0) -> Dict:
        """
        Explain a single prediction.
        
        Args:
            X: Input features
            y_pred: Predicted values
            sample_index: Index of sample to explain
        
        Returns:
            Dictionary with explanation
        """
        explanation = {
            "prediction": float(y_pred[sample_index]) if len(y_pred.shape) > 1 
                         else float(y_pred),
            "input_shape": X[sample_index].shape if len(X.shape) > 2 else "sequence",
            "sample_index": sample_index
        }
        
        # Get SHAP explanation if available
        if SHAP_AVAILABLE and self.explainer is not None:
            try:
                shap_values = self.explainer.shap_values(X[sample_index:sample_index+1])
                explanation["shap_available"] = True
                explanation["shap_base_value"] = float(self.base_value[0]) if isinstance(self.base_value, np.ndarray) else float(self.base_value)
                
                if isinstance(shap_values, list):
                    explanation["shap_values"] = [sv.tolist() for sv in shap_values]
                else:
                    explanation["shap_values"] = shap_values.tolist()
            except Exception as e:
                logger.warning(f"Error computing SHAP values: {e}")
                explanation["shap_available"] = False
        else:
            explanation["shap_available"] = False
        
        # Fallback feature importance
        explanation["feature_importance"] = self._get_fallback_importance(X[sample_index])
        
        return explanation
    
    def explain_predictions_batch(self, X: np.ndarray, y_pred: np.ndarray,
                                  max_samples: int = 10) -> List[Dict]:
        """
        Explain multiple predictions.
        
        Args:
            X: Input features for multiple samples
            y_pred: Predicted values
            max_samples: Maximum number of samples to explain
        
        Returns:
            List of explanation dictionaries
        """
        explanations = []
        num_samples = min(len(X), max_samples)
        
        for i in range(num_samples):
            exp = self.explain_prediction(X, y_pred, i)
            explanations.append(exp)
        
        return explanations
    
    def _get_fallback_importance(self, x_sample: np.ndarray) -> Dict:
        """
        Get feature importance using fallback method (no SHAP).
        
        Args:
            x_sample: Single input sample
        
        Returns:
            Dictionary with feature importance scores
        """
        if len(x_sample.shape) == 2:
            # Time series data - importance by position
            importance = {}
            for i in range(x_sample.shape[0]):
                importance[f"timestep_{i}"] = float(np.abs(x_sample[i, 0]))
            return importance
        elif len(x_sample.shape) == 1:
            # Tabular data - importance by value
            importance = {}
            for i, val in enumerate(x_sample):
                importance[f"feature_{i}"] = float(np.abs(val))
            return importance
        else:
            return {}
    
    def feature_importance_summary(self, X: np.ndarray, num_features: int = 10) -> Dict:
        """
        Get summary of feature importance across samples.
        
        Args:
            X: Input features
            num_features: Number of top features to return
        
        Returns:
            Dictionary with aggregated feature importance
        """
        if len(X.shape) == 3:
            # Time series - importance by timestep
            abs_values = np.abs(X).mean(axis=2)  # Average across features
            importance = {}
            for i in range(abs_values.shape[1]):
                importance[f"timestep_{i}"] = float(abs_values[:, i].mean())
            
            # Sort and return top
            sorted_importance = sorted(importance.items(), 
                                      key=lambda x: x[1], reverse=True)
            return dict(sorted_importance[:num_features])
        
        else:
            # Fallback
            return self._get_fallback_importance(X[0])
    
    def waterfall_plot_data(self, X: np.ndarray, y_pred: np.ndarray,
                           sample_index: int = 0) -> Dict:
        """
        Generate data for waterfall plot (SHAP-style explanation).
        
        Args:
            X: Input features
            y_pred: Predicted values
            sample_index: Index of sample to explain
        
        Returns:
            Dictionary with waterfall plot data
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            return self._waterfall_fallback(X, y_pred, sample_index)
        
        try:
            sample = X[sample_index:sample_index+1]
            shap_values = self.explainer.shap_values(sample)
            base_value = self.base_value[0] if isinstance(self.base_value, np.ndarray) else self.base_value
            
            # Prepare waterfall data
            waterfall_data = {
                "base_value": float(base_value),
                "prediction": float(y_pred[sample_index]) if len(y_pred.shape) > 1 else float(y_pred),
                "features": []
            }
            
            # Add feature contributions
            if isinstance(shap_values, np.ndarray) and len(shap_values.shape) > 1:
                shap_vals = shap_values[0].flatten()
            else:
                shap_vals = shap_values.flatten() if hasattr(shap_values, 'flatten') else [shap_values]
            
            for i, sv in enumerate(shap_vals[:20]):  # Top 20 features
                waterfall_data["features"].append({
                    "name": f"feature_{i}",
                    "value": float(sv),
                    "abs_value": float(np.abs(sv))
                })
            
            # Sort by absolute value
            waterfall_data["features"].sort(key=lambda x: x["abs_value"], reverse=True)
            
            return waterfall_data
        
        except Exception as e:
            logger.warning(f"Error in waterfall plot: {e}")
            return self._waterfall_fallback(X, y_pred, sample_index)
    
    def _waterfall_fallback(self, X: np.ndarray, y_pred: np.ndarray,
                           sample_index: int) -> Dict:
        """
        Fallback waterfall plot data without SHAP.
        
        Args:
            X: Input features
            y_pred: Predicted values
            sample_index: Index of sample
        
        Returns:
            Waterfall-style explanation data
        """
        sample = X[sample_index]
        
        waterfall_data = {
            "base_value": float(np.mean(X)),
            "prediction": float(y_pred[sample_index]) if len(y_pred.shape) > 1 else float(y_pred),
            "features": []
        }
        
        if len(sample.shape) == 1:
            # Tabular features
            for i, val in enumerate(sample[:20]):
                waterfall_data["features"].append({
                    "name": f"feature_{i}",
                    "value": float(val * 0.1),  # Rough approximation
                    "abs_value": float(np.abs(val * 0.1))
                })
        else:
            # Time series - use variance
            for i in range(sample.shape[0]):
                waterfall_data["features"].append({
                    "name": f"timestep_{i}",
                    "value": float((sample[i] - np.mean(X)) * 0.1),
                    "abs_value": float(np.abs((sample[i] - np.mean(X)) * 0.1))
                })
        
        waterfall_data["features"].sort(key=lambda x: x["abs_value"], reverse=True)
        return waterfall_data
    
    def partial_dependence(self, X: np.ndarray, feature_index: int,
                          num_points: int = 20) -> Dict:
        """
        Calculate partial dependence of prediction on a feature.
        
        Args:
            X: Input features
            feature_index: Index of feature to analyze
            num_points: Number of points to evaluate
        
        Returns:
            Dictionary with partial dependence values
        """
        if len(X.shape) < 2:
            return {"error": "Need at least 2D input"}
        
        # Get feature values
        feature_values = X[:, feature_index] if len(X.shape) == 2 else X[:, feature_index, 0]
        feature_min = feature_values.min()
        feature_max = feature_values.max()
        
        # Create grid
        grid = np.linspace(feature_min, feature_max, num_points)
        
        # This is a simplified version - full PDP requires retraining
        # Here we just show the relationship in the input data
        dependence = {
            "feature_index": feature_index,
            "grid": grid.tolist(),
            "values": []
        }
        
        for val in grid:
            # Find samples close to this value
            if len(X.shape) == 2:
                mask = np.abs(X[:, feature_index] - val) < (feature_max - feature_min) / num_points
            else:
                mask = np.abs(X[:, feature_index, 0] - val) < (feature_max - feature_min) / num_points
            
            if mask.any():
                mean_pred = np.mean([X[i] for i in range(len(X)) if mask[i]])
                dependence["values"].append(float(mean_pred))
            else:
                dependence["values"].append(None)
        
        return dependence


class TreeExplainer:
    """SHAP TreeExplainer wrapper for tree-based models."""
    
    def __init__(self, model, feature_names: Optional[List[str]] = None):
        """
        Initialize TreeExplainer.
        
        Args:
            model: Trained tree-based model (RandomForest, XGBoost, etc.)
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names or []
        self.explainer = None
        
        if SHAP_AVAILABLE:
            try:
                self.explainer = shap.TreeExplainer(model)
                logger.info("TreeExplainer initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize TreeExplainer: {e}")
    
    def explain(self, X: np.ndarray) -> Dict:
        """
        Explain predictions.
        
        Args:
            X: Input features
        
        Returns:
            Explanation dictionary
        """
        if self.explainer is None:
            return {"error": "Explainer not available"}
        
        try:
            shap_values = self.explainer.shap_values(X)
            return {
                "shap_values": shap_values,
                "base_value": float(self.explainer.expected_value),
                "feature_names": self.feature_names
            }
        except Exception as e:
            logger.error(f"Error explaining: {e}")
            return {"error": str(e)}


class KernelExplainer:
    """SHAP KernelExplainer wrapper for model-agnostic explanations."""
    
    def __init__(self, model, background_data: np.ndarray,
                 feature_names: Optional[List[str]] = None):
        """
        Initialize KernelExplainer.
        
        Args:
            model: Model predict function
            background_data: Background data for baseline
            feature_names: List of feature names
        """
        self.model = model
        self.background_data = background_data
        self.feature_names = feature_names or []
        self.explainer = None
        
        if SHAP_AVAILABLE:
            try:
                self.explainer = shap.KernelExplainer(
                    model,
                    shap.sample(background_data, min(100, len(background_data)))
                )
                logger.info("KernelExplainer initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize KernelExplainer: {e}")
    
    def explain(self, X: np.ndarray, max_samples: int = 5) -> Dict:
        """
        Explain predictions.
        
        Args:
            X: Input features
            max_samples: Maximum samples to explain
        
        Returns:
            Explanation dictionary
        """
        if self.explainer is None:
            return {"error": "Explainer not available"}
        
        try:
            # Limit samples for efficiency
            X_subset = X[:max_samples]
            shap_values = self.explainer.shap_values(X_subset)
            
            return {
                "shap_values": shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values,
                "base_value": float(self.explainer.expected_value),
                "feature_names": self.feature_names,
                "num_samples": len(X_subset)
            }
        except Exception as e:
            logger.error(f"Error explaining: {e}")
            return {"error": str(e)}


def create_explanation_report(model_explainer: ModelExplainer, 
                             X_test: np.ndarray, 
                             y_pred: np.ndarray,
                             commodity: str = "wheat") -> Dict:
    """
    Create comprehensive explanation report for model predictions.
    
    Args:
        model_explainer: ModelExplainer instance
        X_test: Test input features
        y_pred: Test predictions
        commodity: Commodity name for context
    
    Returns:
        Comprehensive explanation report
    """
    report = {
        "commodity": commodity,
        "total_samples": len(X_test),
        "sample_explanations": [],
        "feature_importance_summary": {}
    }
    
    # Get explanations for subset of samples
    num_samples = min(5, len(X_test))
    for i in range(num_samples):
        exp = model_explainer.explain_prediction(X_test, y_pred, i)
        report["sample_explanations"].append(exp)
    
    # Get feature importance summary
    report["feature_importance_summary"] = model_explainer.feature_importance_summary(X_test)
    
    return report
