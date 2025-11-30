"""
AgroBERT ML Module
Comprehensive machine learning infrastructure for crop price prediction
"""

__version__ = "1.0.0"

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Try to import components with graceful fallback
try:
    from .preprocessing import DataPreprocessor, DataValidator
except (ImportError, ModuleNotFoundError):
    logger.debug("Preprocessing module not fully available")
    DataPreprocessor = None
    DataValidator = None

try:
    from .models import LSTMPricePredictor, SimpleARIMAPredictor
except (ImportError, ModuleNotFoundError):
    logger.debug("Models module not fully available (TensorFlow may be missing)")
    LSTMPricePredictor = None
    SimpleARIMAPredictor = None

try:
    from .explainer import ModelExplainer, TreeExplainer, KernelExplainer, create_explanation_report
except (ImportError, ModuleNotFoundError):
    logger.debug("Explainer module not fully available (SHAP may be missing)")
    ModelExplainer = None
    TreeExplainer = None
    KernelExplainer = None
    create_explanation_report = None

try:
    from .inference import ModelInference, EnsemblePredictor
except (ImportError, ModuleNotFoundError):
    logger.debug("Inference module not fully available")
    ModelInference = None
    EnsemblePredictor = None

try:
    from .train import ModelTrainer
except (ImportError, ModuleNotFoundError):
    logger.debug("Training module not fully available")
    ModelTrainer = None

__all__ = [
    'DataPreprocessor',
    'DataValidator',
    'LSTMPricePredictor',
    'SimpleARIMAPredictor',
    'ModelExplainer',
    'TreeExplainer',
    'KernelExplainer',
    'create_explanation_report',
    'ModelInference',
    'EnsemblePredictor',
    'ModelTrainer',
]
