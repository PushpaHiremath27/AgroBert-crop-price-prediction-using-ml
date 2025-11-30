# AgroBERT ML Module

Comprehensive machine learning infrastructure for crop price prediction and explainability.

## Overview

The ML module provides:

- **Data Preprocessing**: Normalization, outlier detection, feature engineering
- **LSTM Models**: Deep learning-based time-series forecasting
- **ARIMA Fallback**: Statistical models for robustness
- **Model Explainability**: SHAP-based feature importance and prediction explanations
- **Ensemble Predictions**: Combine multiple models for improved accuracy
- **Model Training Pipeline**: Automated training on historical Agmarknet data
- **Inference Engine**: Production-ready prediction serving with fallbacks

## Directory Structure

```
backend/models_ml/
├── __init__.py              # Package initialization
├── config.py                # Configuration and hyperparameters
├── preprocessing.py         # Data preprocessing and validation
├── models.py                # LSTM and ARIMA model implementations
├── explainer.py             # SHAP-based model explainability
├── train.py                 # Training pipeline
├── inference.py             # Inference engine and ensemble predictor
├── ml_requirements.txt      # ML-specific dependencies
└── README.md                # This file
```

## Installation

### 1. Install ML Dependencies

```bash
# Install ML-specific packages
pip install -r backend/models_ml/ml_requirements.txt

# Or install into main environment
pip install tensorflow numpy pandas scikit-learn shap statsmodels
```

### 2. Verify Installation

```python
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

## Quick Start

### Training Models

```python
from backend.models_ml.train import ModelTrainer

# Initialize trainer
trainer = ModelTrainer(data_dir="data/", models_dir="models/")

# Train models for multiple commodities
results = trainer.train_all_commodities(
    commodities=['wheat', 'rice', 'corn'],
    epochs=50
)

# Save summary
trainer.save_models_summary("models/training_summary.json")
```

### Making Predictions

```python
from backend.models_ml.inference import ModelInference
import numpy as np

# Initialize inference engine
inference = ModelInference(models_dir="models/")

# Recent 30+ days of prices
recent_prices = np.array([...])  

# Predict next 7 days
prediction = inference.predict_price(
    commodity='wheat',
    recent_prices=recent_prices,
    days=7
)

print(f"Predictions: {prediction['predictions']}")
print(f"Model used: {prediction['model_used']}")
print(f"Confidence: {prediction['confidence']}")
```

### Getting Predictions with Explanations

```python
from backend.models_ml.explainer import ModelExplainer

# Create explainer
explainer = ModelExplainer(model=model, model_type="lstm")

# Get explanation for prediction
explanation = explainer.explain_prediction(X_test, predictions, sample_index=0)

print(f"Prediction: {explanation['prediction']}")
print(f"Feature importance: {explanation['feature_importance']}")
if explanation.get('shap_available'):
    print(f"SHAP values: {explanation['shap_values']}")
```

### Ensemble Predictions

```python
from backend.models_ml.inference import EnsemblePredictor

# Initialize ensemble
ensemble = EnsemblePredictor(models_dir="models/")

# Get ensemble predictions (70% LSTM + 30% fallback)
prediction = ensemble.predict_ensemble(
    commodity='wheat',
    recent_prices=recent_prices,
    days=7
)
```

## Components

### 1. Data Preprocessing (`preprocessing.py`)

```python
from backend.models_ml.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()

# Handle missing values
data_clean = preprocessor.handle_missing_values(data, method='interpolate')

# Remove outliers
data_clean = preprocessor.remove_outliers(data, threshold=3.0)

# Normalize prices
normalized, params = preprocessor.normalize_data(data, method='minmax')

# Extract temporal features
data_features = preprocessor.extract_temporal_features(data, date_column='date')

# Extract price features (technical indicators)
data_features = preprocessor.extract_price_features(data)

# Create sequences for LSTM
X, y = preprocessor.create_sequences(prices, lookback=30, lookahead=1)
```

### 2. LSTM Model (`models.py`)

```python
from backend.models_ml.models import LSTMPricePredictor

# Create model
model = LSTMPricePredictor(lookback=30, output_steps=7)

# Train
history = model.train(
    X_train, y_train,
    X_val=X_test, y_val=y_test,
    epochs=100,
    batch_size=32
)

# Predict
predictions = model.predict(X_test)

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(f"RMSE: {metrics['rmse']:.4f}, MAPE: {metrics['mape']:.2f}%")

# Predict next days from recent prices
next_7_days = model.predict_next_days(recent_prices[-30:], days=7)

# Save/Load
model.save("models/lstm_wheat")
model.load("models/lstm_wheat")
```

### 3. Model Explainability (`explainer.py`)

```python
from backend.models_ml.explainer import ModelExplainer

explainer = ModelExplainer(model=model, model_type="lstm")

# Explain single prediction
explanation = explainer.explain_prediction(X_test, y_pred, sample_index=0)

# Batch explanations
explanations = explainer.explain_predictions_batch(X_test, y_pred, max_samples=10)

# Feature importance summary
importance = explainer.feature_importance_summary(X_test, num_features=10)

# Waterfall plot data (SHAP-style)
waterfall = explainer.waterfall_plot_data(X_test, y_pred, sample_index=0)

# Partial dependence
dependence = explainer.partial_dependence(X_test, feature_index=0)
```

### 4. Training Pipeline (`train.py`)

```python
from backend.models_ml.train import ModelTrainer

trainer = ModelTrainer(data_dir="data/", models_dir="models/")

# Load data from cache
df = trainer.load_agmarknet_data("data/raw/agmark_cache.json")

# Or generate sample data
df = trainer.generate_sample_data(num_records=2000)

# Prepare data for specific commodity
X_train, y_train, X_test, y_test, metadata = trainer.prepare_training_data(
    df, commodity='wheat', market=None, lookback=30
)

# Train LSTM
metrics = trainer.train_lstm_model(
    X_train, y_train, X_test, y_test,
    commodity='wheat', epochs=50
)

# Train all commodities
results = trainer.train_all_commodities(
    commodities=['wheat', 'rice', 'corn'],
    epochs=50
)

# Save training summary
trainer.save_models_summary("models/training_summary.json")
```

### 5. Inference Engine (`inference.py`)

```python
from backend.models_ml.inference import ModelInference, EnsemblePredictor

# Standard inference
inference = ModelInference(models_dir="models/")

# Predict
prediction = inference.predict_price(
    commodity='wheat',
    recent_prices=recent_prices,
    days=7
)

# Get model performance
performance = inference.get_model_performance('wheat')

# Get all models summary
summary = inference.get_all_models_summary()

# Explain prediction
explanation = inference.explain_prediction(
    commodity='wheat',
    recent_prices=recent_prices,
    prediction=prediction['predictions'][0]
)

# Validate prediction
validation = inference.validate_prediction(
    commodity='wheat',
    prediction=predicted_price,
    recent_prices=recent_prices
)

# Ensemble predictions
ensemble = EnsemblePredictor(models_dir="models/")
ensemble_pred = ensemble.predict_ensemble('wheat', recent_prices, days=7)
```

## Configuration

Edit `backend/models_ml/config.py` to customize:

```python
# LSTM hyperparameters
LSTM_CONFIG = {
    "lookback": 30,
    "output_steps": 7,
    "epochs": 100,
    "batch_size": 32,
    ...
}

# Preprocessing settings
PREPROCESSING_CONFIG = {
    "missing_value_method": "interpolate",
    "outlier_threshold": 3.0,
    "normalization_method": "minmax",
}

# Training settings
TRAINING_CONFIG = {
    "test_split": 0.2,
    "random_seed": 42,
    ...
}
```

## Model Architecture

### LSTM Architecture

```
Input (30 days) 
    ↓
LSTM Layer (64 units, dropout=0.2)
    ↓
LSTM Layer (32 units, dropout=0.2)
    ↓
Dense Layer (16 units)
    ↓
Dropout (0.1)
    ↓
Output (7-day forecast)
```

### Training Details

- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**: MAE, MAPE
- **Early Stopping**: Patience=10 epochs
- **Learning Rate Reduction**: Factor=0.5, Patience=5

## Model Performance

Expected performance targets:

- **RMSE**: < 10% of average price
- **MAE**: < 5% of average price
- **MAPE**: < 10%
- **Inference Time**: < 500ms per prediction

## Fallback Strategy

The system provides graceful fallbacks:

1. **LSTM Available**: Use LSTM predictions (high accuracy)
2. **LSTM Unavailable**: Use fallback trend-based prediction
3. **Insufficient Data**: Use static prediction (last price)

This ensures predictions are always available, even if ML models fail.

## Data Requirements

### For Training

- **Minimum**: 60 days of historical price data per commodity
- **Recommended**: 2+ years of daily prices
- **Fields**: date, commodity, market, modal_price, min_price, max_price

### For Prediction

- **Minimum**: 30 days of recent prices
- **Recommended**: 90+ days for better trend detection

## Integration with Flask Backend

The ML module integrates with `app_flask.py`:

```python
from backend.models_ml.inference import EnsemblePredictor

# In Flask app initialization
ensemble = EnsemblePredictor(models_dir="models/")

# In prediction endpoint
@app.route('/api/v1/predict', methods=['POST'])
def predict():
    data = request.json
    commodity = data.get('commodity')
    recent_prices = np.array(data.get('prices', []))
    
    prediction = ensemble.predict_ensemble(commodity, recent_prices, days=7)
    return jsonify(prediction)
```

## Performance Tips

1. **Batch Predictions**: Use `predict_batch()` for multiple commodities
2. **Model Caching**: Models are loaded once and reused
3. **Data Normalization**: Normalize input data before predictions
4. **Ensemble Weighting**: Adjust weights based on LSTM accuracy

## Troubleshooting

### TensorFlow Not Installed

```bash
pip install tensorflow
```

### SHAP Not Available

The system will use fallback feature importance if SHAP is not installed.

### Insufficient Historical Data

The system will use simulated/fallback predictions for commodities with < 30 days of data.

### Model Not Found

Ensure models are trained first:

```python
trainer = ModelTrainer()
trainer.train_all_commodities()
trainer.save_models_summary()
```

## Advanced Usage

### Custom Model Training

```python
from backend.models_ml.preprocessing import DataPreprocessor
from backend.models_ml.models import LSTMPricePredictor

# Load and preprocess your data
preprocessor = DataPreprocessor()
data = preprocessor.load_data("custom_data.csv")
data_clean = preprocessor.handle_missing_values(data)
data_norm, params = preprocessor.normalize_data(data_clean)

# Create and train model
model = LSTMPricePredictor(lookback=30, output_steps=7)
history = model.train(X_train, y_train, epochs=100)

# Evaluate
metrics = model.evaluate(X_test, y_test)
```

### Model Evaluation and Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true))

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.4f}")
```

## Future Enhancements

- [ ] Multi-step forecasting (predict multiple steps ahead)
- [ ] External feature integration (weather, sentiment)
- [ ] Attention mechanisms for better temporal modeling
- [ ] Uncertainty quantification (prediction intervals)
- [ ] Online learning for continuous model updates
- [ ] Real-time model monitoring and alerts
- [ ] GPU support for faster training
- [ ] Model versioning and rollback

## References

- [TensorFlow LSTM Documentation](https://www.tensorflow.org/guide/keras/rnn)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Time Series Forecasting Best Practices](https://otexts.com/fpp2/)

## License

Same as main AgroBERT project

## Support

For issues or questions, refer to the main project documentation or create an issue.
