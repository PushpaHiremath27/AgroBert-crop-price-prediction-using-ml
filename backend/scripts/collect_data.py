"""
Data Collection Script
Fetches data from external APIs and saves to local storage.
Works with Flask backend (app_flask.py).
"""

import requests
import json
import os
from datetime import datetime, timedelta
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCollector:
    """Collect and store data from various sources via Flask API."""
    
    def __init__(self, backend_url="http://127.0.0.1:5000"):
        """Initialize data collector with Flask backend URL."""
        self.backend_url = backend_url
        self.data_dir = "data/raw"
        self.api_key = os.getenv('DATA_GOV_API_KEY', '')
        self.agmark_resource_id = os.getenv('AGMARK_RESOURCE_ID', '')
        os.makedirs(self.data_dir, exist_ok=True)
        logger.info(f"DataCollector initialized with backend: {backend_url}")
    
    def fetch_agmarknet_data(self):
        """Fetch market prices from data.gov.in Agmarknet API."""
        logger.info("Fetching Agmarknet market data from data.gov.in...")
        
        if not self.api_key or not self.agmark_resource_id:
            logger.warning("Skipping Agmarknet fetch - DATA_GOV_API_KEY or AGMARK_RESOURCE_ID not configured")
            return []
        
        try:
            # Direct API call to data.gov.in
            url = f"https://api.data.gov.in/resource/{self.agmark_resource_id}/records"
            params = {
                'api-key': self.api_key,
                'format': 'json',
                'limit': 1000
            }
            
            logger.info(f"Calling data.gov.in API for Agmarknet data...")
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            records = data.get('records', [])
            
            if records:
                # Save to JSON
                filename = f"{self.data_dir}/agmarknet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w') as f:
                    json.dump({'records': records, 'timestamp': datetime.now().isoformat()}, f, indent=2)
                logger.info(f"✓ Saved {len(records)} Agmarknet price records to {filename}")
            else:
                logger.warning("No records returned from Agmarknet API")
            
            return records
        
        except requests.exceptions.Timeout:
            logger.error("Timeout calling data.gov.in API (slow connection or server issue)")
            return []
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to data.gov.in API")
            return []
        except Exception as e:
            logger.error(f"Error fetching Agmarknet data: {e}")
            return []
    
    def fetch_weather_data(self, locations=None):
        """Fetch weather data for key locations."""
        if locations is None:
            locations = ["Delhi", "Mumbai", "Bangalore", "Kolkata"]
        
        logger.info(f"Fetching weather data for {len(locations)} locations...")
        
        weather_data = []
        
        for location in locations:
            try:
                logger.info(f"  Fetching weather for {location}...")
                # Call local Flask backend weather endpoint
                response = requests.get(
                    f"{self.backend_url}/api/v1/weather",
                    params={'market': location},
                    timeout=5
                )
                
                if response.status_code == 200:
                    weather = response.json()
                    weather['timestamp'] = datetime.now().isoformat()
                    weather_data.append(weather)
                    logger.info(f"  ✓ Got weather for {location}")
                else:
                    logger.warning(f"  ✗ Failed to get weather for {location}: {response.status_code}")
            
            except Exception as e:
                logger.warning(f"  ✗ Error getting weather for {location}: {e}")
                continue
        
        # Save to JSON
        if weather_data:
            filename = f"{self.data_dir}/weather_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(weather_data, f, indent=2)
            logger.info(f"✓ Saved weather data for {len(weather_data)} locations to {filename}")
        
        return weather_data
    
    def fetch_news_data(self):
        """Fetch agricultural news."""
        logger.info("Fetching agricultural news...")
        
        try:
            # Call local Flask backend news endpoint
            response = requests.get(
                f"{self.backend_url}/api/v1/news",
                timeout=5
            )
            
            if response.status_code == 200:
                news_data = response.json()
                news_articles = news_data.get('news', []) if isinstance(news_data, dict) else news_data
                
                # Save to JSON
                filename = f"{self.data_dir}/news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w') as f:
                    json.dump({
                        'articles': news_articles,
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=2)
                
                logger.info(f"✓ Saved {len(news_articles)} news articles to {filename}")
                return news_articles
            else:
                logger.error(f"Failed to fetch news: {response.status_code}")
                return []
        
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    def health_check(self):
        """Check if Flask backend is running."""
        try:
            response = requests.get(f"{self.backend_url}/api/v1/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"✓ Backend is healthy: {response.json()}")
                return True
            else:
                logger.error(f"Backend health check failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Cannot connect to backend at {self.backend_url}: {e}")
            return False
    
    def run_all(self):
        """Run all data collection tasks."""
        logger.info("="*60)
        logger.info("Starting AgroBERT Data Collection Pipeline")
        logger.info("="*60)
        
        # Check backend health
        if not self.health_check():
            logger.error("Backend is not running. Cannot proceed with data collection.")
            logger.info("Start backend with: python backend/app_flask.py")
            return
        
        # Fetch all data types
        self.fetch_agmarknet_data()
        self.fetch_weather_data()
        self.fetch_news_data()
        
        logger.info("="*60)
        logger.info("Data collection complete!")
        logger.info(f"Data saved to: {os.path.abspath(self.data_dir)}")
        logger.info("="*60)

if __name__ == "__main__":
    collector = DataCollector()
    collector.run_all()
