"""
Model Training Pipeline
Train LSTM models on historical Agmarknet price data
"""

import numpy as np
import pandas as pd
import logging
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, Optional

# Import our modules
import sys
sys.path.insert(0, os.path.dirname(__file__))

try:
    from preprocessing import DataPreprocessor, DataValidator
    from models import LSTMPricePredictor, SimpleARIMAPredictor
    from explainer import ModelExplainer, create_explanation_report
    ML_MODULES_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    logging.warning(f"ML modules partially unavailable: {e}")
    ML_MODULES_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class ModelTrainer:
    """Train and manage ML models for price prediction."""
    
    def __init__(self, data_dir: str = "data/", models_dir: str = "models/"):
        """
        Initialize trainer.
        
        Args:
            data_dir: Directory with training data
            models_dir: Directory to save trained models
        """
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.preprocessor = DataPreprocessor()
        self.models = {}
        self.model_metadata = {}
    
    def load_agmarknet_data(self, filepath: str = "data/raw/agmark_cache.json") -> pd.DataFrame:
        """
        Load Agmarknet price data from cache file.
        
        Args:
            filepath: Path to cache file
        
        Returns:
            DataFrame with price data
        """
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Cache file not found: {filepath}")
                return pd.DataFrame()
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            records = []
            if isinstance(data, dict) and 'records' in data:
                records = data['records']
            elif isinstance(data, list):
                records = data
            
            if not records:
                logger.warning("No records found in cache file")
                return pd.DataFrame()
            
            df = pd.DataFrame(records)
            logger.info(f"Loaded {len(df)} records from {filepath}")
            
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def generate_sample_data(self, num_records: int = 1000) -> pd.DataFrame:
        """
        Generate sample price data for testing (when real data unavailable).
        
        Args:
            num_records: Number of records to generate
        
        Returns:
            DataFrame with synthetic price data
        """
        np.random.seed(42)
        
        commodities = ['wheat', 'rice', 'corn', 'soybean', 'cotton']
        markets = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
        
        dates = pd.date_range(end=datetime.now(), periods=num_records, freq='D')
        
        data = {
            'date': [],
            'commodity': [],
            'market': [],
            'min_price': [],
            'max_price': [],
            'modal_price': [],
            'arrival_quantity': []
        }
        
        for commodity in commodities:
            # Base price varies by commodity
            base_price = {'wheat': 2000, 'rice': 3000, 'corn': 1800, 
                         'soybean': 4000, 'cotton': 5000}[commodity]
            
            price = base_price
            for date in dates:
                market = np.random.choice(markets)
                
                # Random walk for price
                change = np.random.normal(0, base_price * 0.02)
                price = price + change
                price = max(price * 0.7, price)  # Keep reasonable
                
                data['date'].append(date)
                data['commodity'].append(commodity)
                data['market'].append(market)
                data['modal_price'].append(float(price))
                data['min_price'].append(float(price * 0.95))
                data['max_price'].append(float(price * 1.05))
                data['arrival_quantity'].append(float(np.random.uniform(100, 1000)))
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} synthetic price records")
        return df
    
    def prepare_training_data(self, df: pd.DataFrame, 
                             commodity: str,
                             market: Optional[str] = None,
                             lookback: int = 30,
                             test_split: float = 0.2) -> Tuple[np.ndarray, np.ndarray, 
                                                               np.ndarray, np.ndarray, 
                                                               Dict]:
        """
        Prepare data for training.
        
        Args:
            df: Input DataFrame
            commodity: Commodity to train on
            market: Specific market (None for all)
            lookback: Number of past days for sequences
            test_split: Test set fraction
        
        Returns:
            Tuple of (X_train, y_train, X_test, y_test, metadata)
        """
        # Filter by commodity
        data = df[df['commodity'].str.lower() == commodity.lower()].copy()
        
        if market:
            data = data[data['market'] == market]
        
        if len(data) < lookback + 10:
            logger.warning(f"Insufficient data for {commodity}: {len(data)} records")
            return None
        
        # Sort by date
        data = data.sort_values('date').reset_index(drop=True)
        
        # Use modal price
        prices = data['modal_price'].values
        
        # Normalize prices
        prices_normalized, norm_params = self.preprocessor.normalize_data(
            pd.DataFrame({'price': prices}), method='minmax'
        )
        prices_normalized = prices_normalized['price'].values
        
        # Create sequences
        X, y = self.preprocessor.create_sequences(
            prices_normalized, lookback=lookback, lookahead=1
        )
        
        # Split into train/test
        split_idx = int(len(X) * (1 - test_split))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        metadata = {
            "commodity": commodity,
            "market": market or "all markets",
            "num_samples": len(X),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "price_range": [float(prices.min()), float(prices.max())],
            "normalization_params": norm_params,
            "lookback": lookback
        }
        
        logger.info(f"Prepared {len(X)} sequences for {commodity}")
        return X_train, y_train, X_test, y_test, metadata
    
    def train_lstm_model(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        commodity: str, epochs: int = 50) -> Dict:
        """
        Train LSTM model.
        
        Args:
            X_train: Training input sequences
            y_train: Training targets
            X_test: Test input sequences
            y_test: Test targets
            commodity: Commodity name for model identification
            epochs: Number of training epochs
        
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training LSTM model for {commodity}...")
        
        model = LSTMPricePredictor(lookback=X_train.shape[1], output_steps=1)
        
        # Train
        history = model.train(
            X_train, y_train,
            X_val=X_test, y_val=y_test,
            epochs=epochs,
            batch_size=32
        )
        
        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        
        # Save model
        model_path = self.models_dir / f"lstm_{commodity.lower()}"
        model.save(str(model_path))
        
        self.models[f"lstm_{commodity}"] = model
        self.model_metadata[f"lstm_{commodity}"] = {
            "type": "LSTM",
            "commodity": commodity,
            "metrics": metrics,
            "training_history": history,
            "saved_at": datetime.now().isoformat()
        }
        
        logger.info(f"LSTM model trained. RMSE: {metrics['rmse']:.4f}, MAPE: {metrics['mape']:.2f}%")
        
        return metrics
    
    def train_arima_fallback(self, prices: np.ndarray, 
                            commodity: str) -> Dict:
        """
        Train simple ARIMA-like fallback model.
        
        Args:
            prices: Price array
            commodity: Commodity name
        
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training ARIMA fallback for {commodity}...")
        
        model = SimpleARIMAPredictor(alpha=0.3)
        model.fit(prices)
        
        self.models[f"arima_{commodity}"] = model
        self.model_metadata[f"arima_{commodity}"] = {
            "type": "ARIMA",
            "commodity": commodity,
            "saved_at": datetime.now().isoformat()
        }
        
        logger.info(f"ARIMA fallback model trained for {commodity}")
        return {"status": "trained"}
    
    def train_all_commodities(self, commodities: list = None,
                             df: pd.DataFrame = None,
                             epochs: int = 50):
        """
        Train models for multiple commodities.
        
        Args:
            commodities: List of commodities (default: common ones)
            df: Training DataFrame (loads from cache if None)
            epochs: Number of training epochs
        
        Returns:
            Dictionary with results for all commodities
        """
        if commodities is None:
            commodities = ['wheat', 'rice', 'corn', 'soybean', 'cotton']
        
        if df is None:
            df = self.load_agmarknet_data()
            if df.empty:
                df = self.generate_sample_data()
        
        results = {}
        
        for commodity in commodities:
            try:
                # Prepare data
                prep_result = self.prepare_training_data(df, commodity)
                
                if prep_result is None:
                    logger.warning(f"Skipping {commodity} - insufficient data")
                    continue
                
                X_train, y_train, X_test, y_test, metadata = prep_result
                
                # Train LSTM
                metrics = self.train_lstm_model(
                    X_train, y_train, X_test, y_test,
                    commodity, epochs=epochs
                )
                
                # Train ARIMA fallback
                prices = np.concatenate([y_train.flatten(), y_test.flatten()])
                self.train_arima_fallback(prices, commodity)
                
                results[commodity] = {
                    "status": "success",
                    "metrics": metrics,
                    "metadata": metadata
                }
                
            except Exception as e:
                logger.error(f"Error training {commodity}: {e}")
                results[commodity] = {"status": "failed", "error": str(e)}
        
        return results
    
    def save_models_summary(self, filepath: str = "models/training_summary.json"):
        """Save summary of all trained models."""
        summary = {
            "trained_at": datetime.now().isoformat(),
            "models": self.model_metadata
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training summary saved to {filepath}")
    
    def evaluate_and_explain(self, commodity: str,
                            X_test: np.ndarray, y_test: np.ndarray,
                            y_pred: np.ndarray) -> Dict:
        """
        Evaluate model and generate explanations.
        
        Args:
            commodity: Commodity name
            X_test: Test input features
            y_test: Test targets
            y_pred: Test predictions
        
        Returns:
            Dictionary with evaluation and explanations
        """
        model_key = f"lstm_{commodity}"
        if model_key not in self.models:
            return {"error": f"Model not found for {commodity}"}
        
        model = self.models[model_key]
        explainer = ModelExplainer(model, model_type="lstm")
        
        report = create_explanation_report(
            explainer, X_test, y_pred, commodity
        )
        
        return report


def main():
    """Main training script."""
    # Initialize trainer
    trainer = ModelTrainer(data_dir="data/", models_dir="models/")
    
    # Load or generate data
    logger.info("Loading training data...")
    df = trainer.load_agmarknet_data("data/raw/agmark_cache.json")
    
    if df.empty:
        logger.info("No cached data found, generating sample data...")
        df = trainer.generate_sample_data(num_records=2000)
    
    # Train models for common commodities
    logger.info("Starting model training...")
    results = trainer.train_all_commodities(
        commodities=['wheat', 'rice', 'corn', 'soybean', 'cotton'],
        df=df,
        epochs=50
    )
    
    # Save summary
    trainer.save_models_summary("models/training_summary.json")
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("TRAINING RESULTS")
    logger.info("="*60)
    for commodity, result in results.items():
        logger.info(f"\n{commodity.upper()}:")
        if result['status'] == 'success':
            metrics = result['metrics']
            logger.info(f"  RMSE: {metrics['rmse']:.4f}")
            logger.info(f"  MAE: {metrics['mae']:.4f}")
            logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        else:
            logger.info(f"  Status: {result['status']}")
            if 'error' in result:
                logger.info(f"  Error: {result['error']}")


if __name__ == "__main__":
    main()
