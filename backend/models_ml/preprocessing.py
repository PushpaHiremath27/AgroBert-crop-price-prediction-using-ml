"""
Data Preprocessing Pipeline for AgroBERT
Handles missing values, normalization, feature engineering
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Data preprocessing and feature engineering pipeline."""
    
    def __init__(self):
        """Initialize preprocessor."""
        self.scaler_params = {}
        self.feature_stats = {}
    
    def handle_missing_values(self, data: pd.DataFrame, method: str = "forward_fill") -> pd.DataFrame:
        """
        Handle missing values in data.
        
        Args:
            data: Input DataFrame
            method: 'forward_fill', 'interpolate', or 'drop'
        
        Returns:
            DataFrame with missing values handled
        """
        data = data.copy()
        
        if method == "forward_fill":
            data = data.fillna(method='ffill').fillna(method='bfill')
        elif method == "interpolate":
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                data[col] = data[col].interpolate(method='linear', limit_direction='both')
        elif method == "drop":
            data = data.dropna()
        
        logger.info(f"Missing values handled using {method} method")
        return data
    
    def remove_outliers(self, data: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers using z-score method.
        
        Args:
            data: Input DataFrame
            threshold: Z-score threshold (default 3.0)
        
        Returns:
            DataFrame with outliers removed
        """
        data = data.copy()
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        initial_rows = len(data)
        
        for col in numeric_columns:
            if data[col].std() > 0:
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                data = data[z_scores < threshold]
        
        removed = initial_rows - len(data)
        logger.info(f"Removed {removed} outlier rows (threshold={threshold})")
        return data
    
    def normalize_data(self, data: pd.DataFrame, method: str = "minmax") -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize numerical data.
        
        Args:
            data: Input DataFrame
            method: 'minmax' or 'zscore'
        
        Returns:
            Tuple of (normalized data, scaling parameters)
        """
        data = data.copy()
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        params = {}
        
        for col in numeric_columns:
            if method == "minmax":
                min_val = data[col].min()
                max_val = data[col].max()
                range_val = max_val - min_val
                if range_val > 0:
                    data[col] = (data[col] - min_val) / range_val
                params[col] = {"min": float(min_val), "max": float(max_val), "method": "minmax"}
            
            elif method == "zscore":
                mean_val = data[col].mean()
                std_val = data[col].std()
                if std_val > 0:
                    data[col] = (data[col] - mean_val) / std_val
                params[col] = {"mean": float(mean_val), "std": float(std_val), "method": "zscore"}
        
        self.scaler_params = params
        logger.info(f"Data normalized using {method} method")
        return data, params
    
    def extract_temporal_features(self, data: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
        """
        Extract temporal features from date column.
        
        Args:
            data: Input DataFrame
            date_column: Name of date column
        
        Returns:
            DataFrame with added temporal features
        """
        data = data.copy()
        
        if date_column not in data.columns:
            logger.warning(f"Date column '{date_column}' not found")
            return data
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            data[date_column] = pd.to_datetime(data[date_column])
        
        # Extract temporal features
        data['day_of_week'] = data[date_column].dt.dayofweek
        data['day_of_month'] = data[date_column].dt.day
        data['month'] = data[date_column].dt.month
        data['quarter'] = data[date_column].dt.quarter
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        # Seasonal indicators (Indian agricultural seasons)
        def get_season(month):
            if month in [10, 11, 12]:  # Rabi season (winter)
                return 1
            elif month in [1, 2, 3]:   # Rabi season (spring)
                return 1
            elif month in [4, 5, 6]:   # Kharif preparation
                return 2
            else:                       # Kharif season (monsoon)
                return 3
        
        data['season'] = data[date_column].dt.month.apply(get_season)
        
        logger.info("Temporal features extracted")
        return data
    
    def extract_price_features(self, data: pd.DataFrame, 
                               min_price_col: str = 'min_price',
                               max_price_col: str = 'max_price',
                               modal_price_col: str = 'modal_price') -> pd.DataFrame:
        """
        Extract price-based features (technical indicators).
        
        Args:
            data: Input DataFrame with price columns
            min_price_col: Name of min price column
            max_price_col: Name of max price column
            modal_price_col: Name of modal price column
        
        Returns:
            DataFrame with added price features
        """
        data = data.copy()
        
        # Price range
        if min_price_col in data.columns and max_price_col in data.columns:
            data['price_range'] = data[max_price_col] - data[min_price_col]
        
        # Price volatility (rolling std)
        if modal_price_col in data.columns:
            data['volatility_7d'] = data[modal_price_col].rolling(window=7, min_periods=1).std()
            data['volatility_30d'] = data[modal_price_col].rolling(window=30, min_periods=1).std()
            
            # Moving averages
            data['sma_7'] = data[modal_price_col].rolling(window=7, min_periods=1).mean()
            data['sma_30'] = data[modal_price_col].rolling(window=30, min_periods=1).mean()
            
            # Price momentum
            data['price_change_7d'] = data[modal_price_col].diff(7)
            data['price_change_30d'] = data[modal_price_col].diff(30)
        
        logger.info("Price features extracted")
        return data
    
    def create_sequences(self, data: np.ndarray, lookback: int = 30, 
                        lookahead: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data: 1D or 2D array of data
            lookback: Number of past timesteps to use (default 30 days)
            lookahead: Number of future timesteps to predict (default 1 day)
        
        Returns:
            Tuple of (X sequences, y targets)
        """
        if len(data) < lookback + lookahead:
            raise ValueError(f"Data length ({len(data)}) must be >= lookback ({lookback}) + lookahead ({lookahead})")
        
        X, y = [], []
        
        for i in range(len(data) - lookback - lookahead + 1):
            X.append(data[i:i + lookback])
            y.append(data[i + lookback + lookahead - 1])
        
        return np.array(X), np.array(y)
    
    def save_scaler_params(self, filepath: str):
        """Save normalization parameters to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.scaler_params, f, indent=2)
            logger.info(f"Scaler parameters saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving scaler params: {e}")
    
    def load_scaler_params(self, filepath: str):
        """Load normalization parameters from file."""
        try:
            with open(filepath, 'r') as f:
                self.scaler_params = json.load(f)
            logger.info(f"Scaler parameters loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading scaler params: {e}")
    
    def inverse_normalize(self, data: np.ndarray, column_name: str, 
                         method: Optional[str] = None) -> np.ndarray:
        """
        Reverse normalization for a specific column.
        
        Args:
            data: Normalized data
            column_name: Name of column being denormalized
            method: Normalization method used ('minmax' or 'zscore')
        
        Returns:
            Denormalized data
        """
        if column_name not in self.scaler_params:
            logger.warning(f"No scaler params found for {column_name}")
            return data
        
        params = self.scaler_params[column_name]
        method = method or params.get('method', 'minmax')
        
        if method == "minmax":
            min_val = params['min']
            max_val = params['max']
            return data * (max_val - min_val) + min_val
        
        elif method == "zscore":
            mean_val = params['mean']
            std_val = params['std']
            return data * std_val + mean_val
        
        return data


class DataValidator:
    """Validate data quality and consistency."""
    
    @staticmethod
    def validate_price_data(data: pd.DataFrame) -> bool:
        """
        Validate agricultural price data.
        
        Args:
            data: Price DataFrame
        
        Returns:
            True if valid, False otherwise
        """
        required_columns = ['date', 'commodity', 'market', 'modal_price']
        
        # Check required columns
        if not all(col in data.columns for col in required_columns):
            logger.error(f"Missing required columns: {required_columns}")
            return False
        
        # Check for duplicates
        duplicates = data.duplicated(subset=['date', 'commodity', 'market']).sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate records")
        
        # Check price ranges (reasonable for Indian mandis)
        modal_prices = data['modal_price'].dropna()
        if (modal_prices < 0).any():
            logger.error("Found negative prices")
            return False
        
        if (modal_prices > 100000).any():  # Unreasonably high for most commodities
            logger.warning("Some prices seem unusually high (>100k)")
        
        # Check date range
        if not pd.api.types.is_datetime64_any_dtype(data['date']):
            try:
                pd.to_datetime(data['date'])
            except:
                logger.error("Invalid date format")
                return False
        
        logger.info("Data validation passed")
        return True
    
    @staticmethod
    def check_data_completeness(data: pd.DataFrame, min_completeness: float = 0.8) -> bool:
        """
        Check if data is sufficiently complete.
        
        Args:
            data: DataFrame to check
            min_completeness: Minimum completeness ratio (default 0.8 = 80%)
        
        Returns:
            True if data meets completeness threshold
        """
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isna().sum().sum()
        completeness = 1 - (missing_cells / total_cells)
        
        logger.info(f"Data completeness: {completeness:.1%}")
        return completeness >= min_completeness
