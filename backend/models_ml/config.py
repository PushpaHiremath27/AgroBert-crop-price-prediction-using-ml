"""
ML Configuration
Central configuration for all ML components
"""

import os
from pathlib import Path

# Directory configuration
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data configuration
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
AGMARK_CACHE_FILE = DATA_RAW_DIR / "agmark_cache.json"
TRAINING_CACHE_FILE = DATA_PROCESSED_DIR / "training_data.parquet"

# Model configuration
MODELS_METADATA_FILE = MODELS_DIR / "training_summary.json"
LSTM_MODEL_TEMPLATE = "lstm_{commodity}.h5"
ARIMA_MODEL_TEMPLATE = "arima_{commodity}.pkl"

# LSTM hyperparameters
LSTM_CONFIG = {
    "lookback": 30,
    "output_steps": 7,
    "lstm_units": [64, 32],
    "dropout_rate": 0.2,
    "dense_units": 16,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "early_stopping_patience": 10,
    "validation_split": 0.2
}

# Data preprocessing configuration
PREPROCESSING_CONFIG = {
    "missing_value_method": "interpolate",  # 'forward_fill', 'interpolate', 'drop'
    "outlier_threshold": 3.0,  # Z-score threshold
    "normalization_method": "minmax",  # 'minmax' or 'zscore'
    "sequence_lookback": 30,
    "sequence_lookahead": 1
}

# Commodities to model
COMMODITIES = [
    'wheat',
    'rice',
    'corn',
    'soybean',
    'cotton',
    'barley',
    'oats',
    'potato',
    'onion',
    'tomato'
]

# Markets (Indian agricultural mandis)
MARKETS = [
    'Delhi',
    'Mumbai',
    'Bangalore',
    'Chennai',
    'Kolkata',
    'Hyderabad',
    'Pune',
    'Jaipur',
    'Lucknow',
    'Ahmedabad'
]

# Model evaluation thresholds
MODEL_PERFORMANCE = {
    "rmse_threshold": None,  # Will be set to 10% of average price
    "mae_threshold": None,   # Will be set to 5% of average price
    "mape_threshold": 10.0,  # %
    "min_r2_score": 0.7
}

# Inference configuration
INFERENCE_CONFIG = {
    "use_ensemble": True,  # Use ensemble predictions
    "lstm_weight": 0.7,     # Weight for LSTM in ensemble
    "fallback_weight": 0.3, # Weight for fallback in ensemble
    "min_data_points": 30,  # Minimum historical data needed
    "confidence_threshold": 0.5,
    "reasonable_price_range": (0.7, 1.3)  # Multipliers for min/max
}

# Explainability configuration
EXPLAINER_CONFIG = {
    "use_shap": True,
    "num_background_samples": 100,
    "max_samples_to_explain": 5,
    "feature_importance_top_n": 10
}

# Training configuration
TRAINING_CONFIG = {
    "test_split": 0.2,
    "validation_split": 0.15,
    "random_seed": 42,
    "shuffle": True,
    "retraining_frequency": "weekly",  # How often to retrain models
    "min_samples_per_commodity": 60  # Minimum days of data needed
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "ml_training.log",
    "max_file_size": 10_000_000,  # 10 MB
    "backup_count": 5
}

# Feature engineering
FEATURES = {
    "temporal": {
        "day_of_week": True,
        "day_of_month": True,
        "month": True,
        "quarter": True,
        "is_weekend": True,
        "season": True  # Indian agricultural seasons
    },
    "price_features": {
        "price_range": True,
        "volatility_7d": True,
        "volatility_30d": True,
        "sma_7": True,
        "sma_30": True,
        "price_change_7d": True,
        "price_change_30d": True
    },
    "external": {
        "temperature": False,  # Not yet integrated
        "rainfall": False,
        "humidity": False
    }
}

# Cache configuration
CACHE_CONFIG = {
    "enabled": True,
    "ttl_seconds": 3600,  # 1 hour
    "max_size": 100,  # MB
    "format": "json"
}

# Performance monitoring
MONITORING_CONFIG = {
    "track_predictions": True,
    "track_latency": True,
    "alert_on_low_confidence": True,
    "alert_on_performance_drop": True,
    "performance_degradation_threshold": 0.1  # 10% drop
}

def get_config(section: str = None) -> dict:
    """
    Get configuration dictionary.
    
    Args:
        section: Specific section to retrieve (None for all)
    
    Returns:
        Configuration dictionary
    """
    config = {
        "paths": {
            "base": str(BASE_DIR),
            "data": str(DATA_DIR),
            "models": str(MODELS_DIR),
            "logs": str(LOGS_DIR)
        },
        "lstm": LSTM_CONFIG,
        "preprocessing": PREPROCESSING_CONFIG,
        "commodities": COMMODITIES,
        "markets": MARKETS,
        "performance": MODEL_PERFORMANCE,
        "inference": INFERENCE_CONFIG,
        "explainer": EXPLAINER_CONFIG,
        "training": TRAINING_CONFIG,
        "logging": LOGGING_CONFIG,
        "features": FEATURES,
        "cache": CACHE_CONFIG,
        "monitoring": MONITORING_CONFIG
    }
    
    if section:
        return config.get(section, {})
    
    return config

if __name__ == "__main__":
    import json
    # Print configuration for verification
    print(json.dumps(get_config(), indent=2, default=str))
