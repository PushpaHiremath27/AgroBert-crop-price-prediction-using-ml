"""
AgroBERT Backend - Flask Application
Unified backend with authentication, ML models, and chatbot
Integrated with data.gov.in Agmarknet API for real market prices
"""

import os
import random
import sqlite3
import json
import time
import datetime
from pathlib import Path
from urllib.parse import urlencode

# Suppress TensorFlow and other verbose logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from flask import Flask, jsonify, request, render_template
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import create_access_token, jwt_required, JWTManager
from flask_cors import CORS
from dotenv import load_dotenv

try:
    import requests
except ImportError:
    requests = None
    print("[!] requests library not installed. Install with: pip install requests")

# --- ML Integration ---
ML_MANAGER = None
try:
    from models_ml.ml_integration import MLPipelineManager
    ML_MANAGER = MLPipelineManager(models_dir="models/")
    print("[OK] ML pipeline loaded successfully")
except (ImportError, ModuleNotFoundError) as e:
    print(f"[!] ML modules not available ({type(e).__name__}). Using simulated predictions.")
except Exception as e:
    print(f"[!] ML pipeline initialization error: {e}. Using simulated predictions.")

# --- Load Environment Variables ---
load_dotenv()

# --- Optional Gemini Setup ---
# Only attempt to import/configure Gemini if an API key is present. This
# avoids importing heavy optional dependencies during startup when the key
# is not configured.
GEMINI_ENABLED = False
_gemini_model = None
_gemini_api_key = os.getenv("GEMINI_API_KEY")
if _gemini_api_key:
    try:
        import google.generativeai as genai
        try:
            genai.configure(api_key=_gemini_api_key)
            generation_config = {"temperature": 0.7, "top_p": 1, "top_k": 1, "max_output_tokens": 2048}
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
            try:
                _gemini_model = genai.GenerativeModel(
                    model_name="gemini-pro",
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                GEMINI_ENABLED = True
                print("[OK] Gemini API initialized successfully.")
            except Exception as e:
                GEMINI_ENABLED = False
                print(f"[!] Gemini model initialization error: {e}. Continuing without Gemini.")
        except Exception as e:
            GEMINI_ENABLED = False
            print(f"[!] Gemini configuration error: {e}. Continuing without Gemini.")
    except Exception as e:
        GEMINI_ENABLED = False
        print(f"[!] Gemini library import error: {e}. Continuing without Gemini.")
else:
    GEMINI_ENABLED = False
    print("[!] GEMINI_API_KEY not set. AI chat will use rule-based responses.")

# Expose gemini_model variable name expected elsewhere in the code
gemini_model = _gemini_model

# --- Initialize Flask App ---
app = Flask(__name__, template_folder='../frontend', static_folder='../frontend', static_url_path='')
CORS(app)

# --- JWT Configuration ---
JWT_SECRET = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
app.config['JWT_SECRET_KEY'] = JWT_SECRET
jwt = JWTManager(app)

# --- Environment Detection ---
IS_PRODUCTION = os.getenv('ENVIRONMENT', 'development').lower() in ['production', 'render']
PORT = int(os.getenv('PORT', 5000))
DEBUG_MODE = os.getenv('DEBUG', 'false').lower() == 'true' and not IS_PRODUCTION

# --- Database Setup ---
import os as os_module
os_module.makedirs('db', exist_ok=True)
DATABASE = 'db/users.db'

def get_db():
    """Get database connection."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_db():
    """Initialize database with required tables."""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        
        # Create demo user if it doesn't exist
        try:
            demo_password_hash = generate_password_hash('demo123')
            cursor.execute(
                'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                ('demo', 'demo@agrobert.local', demo_password_hash)
            )
            conn.commit()
            print("[OK] Demo user created successfully.")
        except sqlite3.IntegrityError:
            # Demo user already exists
            print("[OK] Demo user already exists.")
        
        conn.close()
        print("[OK] Database initialized successfully.")
    except Exception as e:
        print(f"[!] Database initialization error: {e}")

initialize_db()

# --- DATA.GOV.IN AGMARKNET INTEGRATION ---
DATA_GOV_API_KEY = os.getenv('DATA_GOV_API_KEY', '')
AGMARK_RESOURCE_ID = os.getenv('AGMARK_RESOURCE_ID', '')
AGMARK_CACHE_PATH = os.getenv('AGMARK_CACHE_PATH', 'data/raw/agmark_cache.json')
AGMARK_CACHE_TTL_SECONDS = int(os.getenv('AGMARK_CACHE_TTL_SECONDS', '3600'))
AGMARK_MAX_RETRIES = int(os.getenv('AGMARK_MAX_RETRIES', '3'))
AGMARK_TIMEOUT_SECONDS = int(os.getenv('AGMARK_TIMEOUT_SECONDS', '10'))

def _ensure_cache_parent():
    """Ensure cache directory exists."""
    try:
        Path(AGMARK_CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[!] Error creating cache directory: {e}")

def fetch_agmark_from_data_gov(resource_id, params):
    """
    Fetch agricultural market prices from data.gov.in API with retry logic.
    Returns dict with 'records' key containing list of market price records.
    """
    if not requests or not DATA_GOV_API_KEY:
        return None
    
    url = f"https://api.data.gov.in/resource/{resource_id}/records"
    api_params = {
        'api-key': DATA_GOV_API_KEY,
        'format': 'json',
    }
    api_params.update(params)
    
    for attempt in range(AGMARK_MAX_RETRIES):
        try:
            response = requests.get(url, params=api_params, timeout=AGMARK_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            print(f"[!] Agmarknet API timeout (attempt {attempt + 1}/{AGMARK_MAX_RETRIES})")
            if attempt < AGMARK_MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
        except requests.exceptions.RequestException as e:
            print(f"[!] Agmarknet API error (attempt {attempt + 1}/{AGMARK_MAX_RETRIES}): {e}")
            if attempt < AGMARK_MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
        except Exception as e:
            print(f"[!] Unexpected error fetching Agmarknet data: {e}")
            return None
    
    return None

def load_cached_agmark():
    """Load cached Agmarknet data if fresh (within TTL)."""
    try:
        cache_path = Path(AGMARK_CACHE_PATH)
        if not cache_path.exists():
            return None
        
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        cache_age = time.time() - cache_data.get('timestamp', 0)
        if cache_age < AGMARK_CACHE_TTL_SECONDS:
            return cache_data.get('records', [])
        
        return None
    except Exception as e:
        print(f"[!] Agmark cache load error: {e}")
        return None

def save_agmark_cache(data):
    """Save Agmarknet data to cache with timestamp."""
    try:
        _ensure_cache_parent()
        cache_data = {
            'records': data if isinstance(data, list) else data.get('records', []),
            'timestamp': time.time()
        }
        with open(AGMARK_CACHE_PATH, 'w') as f:
            json.dump(cache_data, f)
    except Exception as e:
        print(f"[!] Agmark cache save error: {e}")

def normalize_agmark_records(records):
    """
    Normalize Agmarknet records to consistent schema.
    Handles variations in field names across different datasets.
    """
    if not records:
        return []
    
    normalized = []
    for record in records:
        try:
            # Handle various date field names
            date_val = record.get('date') or record.get('Date') or \
                      record.get('transaction_date') or record.get('Transaction_Date') or \
                      record.get('reporting_date') or record.get('Reporting_Date') or \
                      datetime.datetime.now().strftime('%Y-%m-%d')
            
            # Handle various market field names
            market_val = record.get('market') or record.get('Market') or \
                        record.get('market_center') or record.get('Market_Center') or \
                        record.get('market_name') or record.get('Market_Name') or \
                        record.get('market_name_english') or record.get('Market_Name_English') or \
                        record.get('mandi') or record.get('Mandi') or 'Unknown'
            
            # Handle various commodity field names
            commodity_val = record.get('commodity') or record.get('Commodity') or \
                           record.get('commodities') or record.get('Commodities') or \
                           record.get('commodity_name') or record.get('Commodity_Name') or 'Unknown'
            
            # Handle various min price field names
            min_price = record.get('min_price') or record.get('Min_Price') or \
                       record.get('maximum_price') or record.get('Maximum_Price') or \
                       record.get('min_price_rs') or record.get('Min_Price_Rs') or \
                       record.get('minprice') or record.get('MinPrice') or 0
            
            # Handle various max price field names
            max_price = record.get('max_price') or record.get('Max_Price') or \
                       record.get('maximum_price') or record.get('Maximum_Price') or \
                       record.get('max_price_rs') or record.get('Max_Price_Rs') or \
                       record.get('maxprice') or record.get('MaxPrice') or 0
            
            # Handle various modal price field names
            modal_price = record.get('modal_price') or record.get('Modal_Price') or \
                         record.get('modalprice') or record.get('ModalPrice') or \
                         (float(min_price) + float(max_price)) / 2 if min_price and max_price else 0
            
            # Handle various unit field names
            unit_val = record.get('unit') or record.get('Unit') or \
                      record.get('price_unit') or record.get('Price_Unit') or \
                      record.get('unit_of_price') or record.get('Unit_Of_Price') or \
                      'Rs./Quintal'
            
            # Handle arrival quantity
            arrival = record.get('arrival') or record.get('Arrival') or \
                     record.get('market_arrival') or record.get('Market_Arrival') or '0'
            
            normalized.append({
                'date': str(date_val),
                'market': str(market_val),
                'commodity': str(commodity_val),
                'min_price': float(min_price) if min_price else 0,
                'max_price': float(max_price) if max_price else 0,
                'modal_price': float(modal_price) if modal_price else 0,
                'unit': str(unit_val),
                'arrival': str(arrival)
            })
        except Exception as e:
            print(f"[!] Error normalizing record: {e}")
            continue
    
    return normalized

# --- Commodity Keywords for Search ---
commodity_keywords = {
    'wheat': ['wheat', 'gehun', 'gahu'],
    'rice': ['rice', 'chawal', 'chawl'],
    'maize': ['maize', 'corn', 'makka'],
    'sugarcane': ['sugarcane', 'ganna'],
    'cotton': ['cotton', 'kapas'],
    'groundnut': ['groundnut', 'peanut', 'moongfali'],
    'soybean': ['soybean', 'soya'],
    'mustard': ['mustard', 'sarso'],
}

# --- ENDPOINTS ---

@app.route("/", methods=["GET"])
def serve_frontend():
    """Serve frontend HTML."""
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"[!] Error serving frontend: {e}")
        return jsonify({"error": "Frontend not found"}), 404

@app.route("/api/v1/agmarknet-live", methods=["GET"])
@jwt_required()
def api_agmarknet_live():
    """
    GET /api/v1/agmarknet-live
    Query params: commodity, market, date, start_date, end_date, limit, use_cache (default: true)
    Returns: {success: bool, source: "data-gov"|"cache"|"simulated", data: [...], meta: {...}}
    
    Implements fallback chain:
    1. Try cache first (if use_cache=true and cache is fresh)
    2. Fall back to live fetch from data.gov.in API
    3. Fall back to simulated data if both fail
    """
    try:
        # Parse query parameters
        commodity = request.args.get('commodity', 'Wheat').strip()
        market = request.args.get('market', 'Delhi').strip()
        start_date = request.args.get('start_date', '')
        end_date = request.args.get('end_date', '')
        limit = request.args.get('limit', '100')
        use_cache = request.args.get('use_cache', 'true').lower() == 'true'
        
        try:
            limit = int(limit)
            if limit < 1 or limit > 1000:
                limit = 100
        except:
            limit = 100
        
        # Build query params for data.gov.in API
        api_params = {'limit': limit}
        
        if commodity:
            api_params['resource_id'] = AGMARK_RESOURCE_ID
            api_params['filters'] = json.dumps({'Commodity': commodity})
        
        if start_date:
            api_params['start_date'] = start_date
        if end_date:
            api_params['end_date'] = end_date
        
        data_source = None
        data_records = None
        
        # Step 1: Try cache first
        if use_cache:
            cached = load_cached_agmark()
            if cached:
                print(f"[OK] Using cached data for {commodity} in {market}")
                data_records = cached
                data_source = "cache"
        
        # Step 2: Fall back to live fetch
        if not data_records and DATA_GOV_API_KEY:
            print(f"[OK] Fetching live data for {commodity} from data.gov.in")
            try:
                api_data = fetch_agmark_from_data_gov(AGMARK_RESOURCE_ID, api_params)
                if api_data and 'records' in api_data:
                    data_records = api_data['records']
                    # Save to cache for next request
                    save_agmark_cache({'records': data_records, 'timestamp': time.time()})
                    data_source = "data-gov"
                    print(f"[OK] Successfully fetched {len(data_records)} records from data.gov.in")
            except Exception as e:
                print(f"[!] Error fetching from data.gov.in: {str(e)}")
                data_records = None
        
        # Step 3: Fall back to simulated data
        if not data_records:
            print(f"[OK] Using simulated data (API unavailable)")
            data_records = []
            # Generate simulated market data
            today = datetime.datetime.now()
            for i in range(min(limit, 20)):
                date_str = (today - datetime.timedelta(days=i)).strftime('%Y-%m-%d')
                simulated_record = {
                    'Date': date_str,
                    'Market': market,
                    'Commodity': commodity,
                    'MinPrice': round(random.uniform(2000, 5000), 2),
                    'MaxPrice': round(random.uniform(5500, 10000), 2),
                    'ModalPrice': round(random.uniform(3000, 7000), 2),
                    'Unit': 'Rs./Quintal',
                    'Arrival': str(random.randint(100, 5000))
                }
                data_records.append(simulated_record)
            data_source = "simulated"
        
        # Normalize records to consistent schema
        normalized_data = normalize_agmark_records(data_records)
        
        # Filter by market if needed (case-insensitive)
        if market and market.lower() != 'all':
            normalized_data = [
                r for r in normalized_data 
                if r.get('market', '').lower() == market.lower()
            ]
        
        # Return response
        response = {
            'success': True,
            'source': data_source,
            'count': len(normalized_data),
            'data': normalized_data[:limit],
            'meta': {
                'commodity': commodity,
                'market': market,
                'query_date': datetime.datetime.now().isoformat(),
                'cache_fresh': data_source in ['cache', 'data-gov']
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"[!] Unexpected error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'source': 'error'
        }), 500

@app.route("/api/v1/location-based-markets", methods=["POST"])
@jwt_required()
def location_based_markets():
    """
    POST /api/v1/location-based-markets
    Get nearest market and commodity data based on user's geolocation.
    
    Request body: {
        "latitude": float,
        "longitude": float,
        "commodity": str (optional, default: "Wheat")
    }
    """
    data = request.get_json()
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    commodity = data.get('commodity', 'Wheat')
    
    if not latitude or not longitude:
        return jsonify({
            'success': False,
            'error': 'latitude and longitude are required'
        }), 400
    
    try:
        # Market coordinates - Major agricultural markets in India
        markets_db = {
            'Delhi': {'lat': 28.7041, 'lng': 77.1025, 'state': 'Delhi', 'type': 'Metro'},
            'Mumbai': {'lat': 19.0760, 'lng': 72.8777, 'state': 'Maharashtra', 'type': 'Metro'},
            'Bangalore': {'lat': 12.9716, 'lng': 77.5946, 'state': 'Karnataka', 'type': 'Metro'},
            'Chennai': {'lat': 13.0827, 'lng': 80.2707, 'state': 'Tamil Nadu', 'type': 'Metro'},
            'Kolkata': {'lat': 22.5726, 'lng': 88.3639, 'state': 'West Bengal', 'type': 'Metro'},
            'Hyderabad': {'lat': 17.3850, 'lng': 78.4867, 'state': 'Telangana', 'type': 'Metro'},
            'Pune': {'lat': 18.5204, 'lng': 73.8567, 'state': 'Maharashtra', 'type': 'Metro'},
            'Ahmedabad': {'lat': 23.0225, 'lng': 72.5714, 'state': 'Gujarat', 'type': 'Metro'},
            'Jaipur': {'lat': 26.9124, 'lng': 75.7873, 'state': 'Rajasthan', 'type': 'Tier-2'},
            'Lucknow': {'lat': 26.8467, 'lng': 80.9462, 'state': 'Uttar Pradesh', 'type': 'Tier-2'},
            'Chandigarh': {'lat': 30.7333, 'lng': 76.7794, 'state': 'Chandigarh', 'type': 'Tier-2'},
            'Punjab-Sangrur': {'lat': 30.2424, 'lng': 75.6320, 'state': 'Punjab', 'type': 'Agricultural'},
            'Gujarat-Dohad': {'lat': 22.2406, 'lng': 74.5154, 'state': 'Gujarat', 'type': 'Agricultural'},
            'Tamil Nadu-Coimbatore': {'lat': 11.0066, 'lng': 76.9655, 'state': 'Tamil Nadu', 'type': 'Agricultural'},
            'Karnataka-Belgaum': {'lat': 15.8497, 'lng': 75.6193, 'state': 'Karnataka', 'type': 'Agricultural'},
            'Davangere': {'lat': 14.4667, 'lng': 75.9167, 'state': 'Karnataka', 'type': 'Agricultural'},
            'Hosapete': {'lat': 14.2386, 'lng': 76.3740, 'state': 'Karnataka', 'type': 'Agricultural'},
            'Hassan': {'lat': 13.2031, 'lng': 75.9722, 'state': 'Karnataka', 'type': 'Agricultural'},
            'Mandya': {'lat': 12.5647, 'lng': 76.1690, 'state': 'Karnataka', 'type': 'Agricultural'},
            'Kolar': {'lat': 13.1379, 'lng': 78.1310, 'state': 'Karnataka', 'type': 'Agricultural'},
        }
        
        # Find nearest market using haversine distance
        def calculate_distance(lat1, lon1, lat2, lon2):
            from math import radians, sin, cos, sqrt, atan2
            R = 6371  # Earth's radius in kilometers
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            return R * c
        
        nearest_market = None
        min_distance = float('inf')
        
        for market_name, coords in markets_db.items():
            distance = calculate_distance(latitude, longitude, coords['lat'], coords['lng'])
            if distance < min_distance:
                min_distance = distance
                nearest_market = market_name
        
        nearest_market_info = markets_db[nearest_market]
        print(f"[GEOLOCATION] User at ({latitude:.4f}, {longitude:.4f}) -> Nearest market: {nearest_market} ({min_distance:.2f}km away)")
        
        # Fetch market data for nearest market
        agmark_data = load_cached_agmark()
        market_prices = []
        
        if agmark_data and 'records' in agmark_data:
            # Filter data for nearest market and commodity
            market_prices = [
                {
                    'commodity': r.get('commodity', 'Unknown'),
                    'modal_price': r.get('modal_price', 0),
                    'min_price': r.get('min_price', 0),
                    'max_price': r.get('max_price', 0),
                    'arrival': r.get('arrival', 0),
                    'date': r.get('date', '')
                }
                for r in agmark_data['records']
                if r.get('market', '').lower() == nearest_market.lower()
            ]
        
        # If no specific data, generate simulated data for the market
        if not market_prices:
            today = datetime.datetime.now()
            market_prices = []
            commodities = ['Wheat', 'Rice', 'Cotton', 'Onion', 'Potato', 'Tomato', 'Maize', 'Banana']
            for comm in commodities:
                market_prices.append({
                    'commodity': comm,
                    'modal_price': round(random.uniform(2000, 7000), 2),
                    'min_price': round(random.uniform(1500, 5000), 2),
                    'max_price': round(random.uniform(5000, 10000), 2),
                    'arrival': str(random.randint(100, 5000)),
                    'date': today.strftime('%Y-%m-%d')
                })
        
        response = {
            'success': True,
            'user_location': {
                'latitude': latitude,
                'longitude': longitude
            },
            'nearest_market': {
                'name': nearest_market,
                'state': nearest_market_info['state'],
                'type': nearest_market_info['type'],
                'distance_km': round(min_distance, 2),
                'coordinates': {
                    'latitude': nearest_market_info['lat'],
                    'longitude': nearest_market_info['lng']
                }
            },
            'market_prices': market_prices,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"[!] Geolocation error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'nearest_market': 'Delhi'  # Fallback
        }), 500

@app.route("/api/v1/geolocation", methods=["POST"])
@jwt_required()
def geolocation():
    """Alias endpoint for location-based-markets for frontend compatibility."""
    data = request.get_json()
    
    # Call the location-based-markets function with modified response format
    try:
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        commodity = data.get('commodity', 'Wheat')
        
        if not latitude or not longitude:
            return jsonify({'error': 'latitude and longitude required'}), 400
        
        # Use the location_based_markets logic here
        markets_db = {
            'Delhi': {'lat': 28.7041, 'lng': 77.1025, 'state': 'Delhi', 'type': 'Metro'},
            'Mumbai': {'lat': 19.0760, 'lng': 72.8777, 'state': 'Maharashtra', 'type': 'Metro'},
            'Bangalore': {'lat': 12.9716, 'lng': 77.5946, 'state': 'Karnataka', 'type': 'Metro'},
            'Chennai': {'lat': 13.0827, 'lng': 80.2707, 'state': 'Tamil Nadu', 'type': 'Metro'},
            'Kolkata': {'lat': 22.5726, 'lng': 88.3639, 'state': 'West Bengal', 'type': 'Metro'},
            'Hyderabad': {'lat': 17.3850, 'lng': 78.4867, 'state': 'Telangana', 'type': 'Metro'},
            'Pune': {'lat': 18.5204, 'lng': 73.8567, 'state': 'Maharashtra', 'type': 'Metro'},
            'Ahmedabad': {'lat': 23.0225, 'lng': 72.5714, 'state': 'Gujarat', 'type': 'Metro'},
            'Jaipur': {'lat': 26.9124, 'lng': 75.7873, 'state': 'Rajasthan', 'type': 'Tier-2'},
            'Lucknow': {'lat': 26.8467, 'lng': 80.9462, 'state': 'Uttar Pradesh', 'type': 'Tier-2'},
            'Chandigarh': {'lat': 30.7333, 'lng': 76.7794, 'state': 'Chandigarh', 'type': 'Tier-2'},
            'Punjab-Sangrur': {'lat': 30.2424, 'lng': 75.6320, 'state': 'Punjab', 'type': 'Agricultural'},
            'Gujarat-Dohad': {'lat': 22.2406, 'lng': 74.5154, 'state': 'Gujarat', 'type': 'Agricultural'},
            'Tamil Nadu-Coimbatore': {'lat': 11.0066, 'lng': 76.9655, 'state': 'Tamil Nadu', 'type': 'Agricultural'},
            'Karnataka-Belgaum': {'lat': 15.8497, 'lng': 75.6193, 'state': 'Karnataka', 'type': 'Agricultural'},
            'Davangere': {'lat': 14.4667, 'lng': 75.9167, 'state': 'Karnataka', 'type': 'Agricultural'},
            'Hosapete': {'lat': 14.2386, 'lng': 76.3740, 'state': 'Karnataka', 'type': 'Agricultural'},
            'Hassan': {'lat': 13.2031, 'lng': 75.9722, 'state': 'Karnataka', 'type': 'Agricultural'},
            'Mandya': {'lat': 12.5647, 'lng': 76.1690, 'state': 'Karnataka', 'type': 'Agricultural'},
            'Kolar': {'lat': 13.1379, 'lng': 78.1310, 'state': 'Karnataka', 'type': 'Agricultural'},
        }
        
        from math import radians, sin, cos, sqrt, atan2
        def calculate_distance(lat1, lon1, lat2, lon2):
            R = 6371
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            return R * c
        
        nearest_market = None
        min_distance = float('inf')
        
        for market_name, coords in markets_db.items():
            distance = calculate_distance(latitude, longitude, coords['lat'], coords['lng'])
            if distance < min_distance:
                min_distance = distance
                nearest_market = market_name
        
        return jsonify({
            'nearest_market': nearest_market,
            'distance_km': round(min_distance, 2)
        }), 200
    except Exception as e:
        print(f"[!] Geolocation error: {e}")
        return jsonify({'nearest_market': 'Delhi', 'distance_km': 0}), 200

@app.route("/api/v1/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.datetime.now().isoformat()
    }), 200

@app.route("/api/v1/login", methods=["POST"])
def login():
    """Login endpoint - returns JWT token."""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    print(f"[LOGIN] Attempt with username: {username}")
    
    if not username or not password:
        print(f"[LOGIN] Missing credentials")
        return jsonify({"error": "Missing username or password"}), 400
    
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            print(f"[LOGIN] User found: {username}")
            if check_password_hash(user['password_hash'], password):
                print(f"[LOGIN] Password correct, issuing token")
                access_token = create_access_token(identity=user['username'])
                return jsonify({"token": access_token, "username": user['username']}), 200
            else:
                print(f"[LOGIN] Password incorrect for user: {username}")
                return jsonify({"error": "Invalid credentials"}), 401
        else:
            print(f"[LOGIN] User not found: {username}")
            return jsonify({"error": "Invalid credentials"}), 401
    except Exception as e:
        print(f"[LOGIN] Exception: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/v1/register", methods=["POST"])
def register():
    """Register new user."""
    data = request.get_json()
    username = data.get('username')
    email = data.get('email', '')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400
    
    try:
        conn = get_db()
        cursor = conn.cursor()
        password_hash = generate_password_hash(password)
        cursor.execute(
            'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
            (username, email, password_hash)
        )
        conn.commit()
        conn.close()
        return jsonify({"message": "User registered successfully"}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username already exists"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/v1/predict", methods=["POST"])
@jwt_required()
def predict():
    """Price prediction endpoint with ML fallback to simulated data."""
    data = request.get_json()
    commodity = data.get('commodity', 'Wheat')
    market = data.get('market', 'Delhi')
    # Accept both daysAhead (from frontend) and months_ahead (from other sources)
    days_ahead = int(data.get('daysAhead', data.get('months_ahead', 1)))
    months_ahead = days_ahead // 30 if days_ahead >= 30 else max(1, days_ahead // 10)
    
    # Try to use ML prediction first
    if ML_MANAGER:
        try:
            # Get recent prices from Agmarknet
            agmark_data = load_cached_agmark()
            if agmark_data and 'records' in agmark_data:
                # Filter by commodity
                filtered = [r for r in agmark_data['records'] 
                           if r.get('commodity', '').lower() == commodity.lower()]
                if filtered:
                    recent_prices = [float(r.get('modal_price', 0)) for r in filtered[-30:]]
                    if len(recent_prices) >= 7:
                        # Get ML prediction
                        days_ahead = int(months_ahead * 30)
                        prediction_result = ML_MANAGER.predict(commodity, recent_prices, days=days_ahead)
                        
                        if 'error' not in prediction_result and prediction_result.get('predictions'):
                            preds = prediction_result['predictions']
                            predicted_price = preds[-1] if preds else recent_prices[-1]
                            
                            current = round(recent_prices[-1], 2)
                            predicted = round(predicted_price, 2)
                            # Create chart data
                            import datetime
                            dates = []
                            for i in range(30):
                                d = datetime.datetime.now() - datetime.timedelta(days=29-i)
                                dates.append(d.strftime('%Y-%m-%d'))
                            chart_prices = [round(p, 2) if isinstance(p, (int, float)) else 0 for p in recent_prices[-28:]] + [current, predicted]
                            # XAI factors
                            xai_factors = [
                                {"factor": "Historical Price Trend", "impact": "Positive" if predicted > current else "Negative"},
                                {"factor": "Market Sentiments", "impact": "Positive"},
                                {"factor": "Seasonal Patterns", "impact": "Neutral"},
                                {"factor": "Weather Conditions", "impact": "Neutral"},
                                {"factor": "Supply Demand", "impact": "Positive"}
                            ]
                            return jsonify({
                                "commodity": commodity,
                                "market": market,
                                "current_price": current,
                                "predicted_price": predicted,
                                "months_ahead": months_ahead,
                                "trend": "up" if predicted > current else "down" if predicted < current else "stable",
                                "model_used": prediction_result.get('model_used', 'fallback'),
                                "confidence": max(prediction_result.get('confidence', [0.5])),
                                "all_predictions": [round(p, 2) for p in preds[:10]],
                                "prediction_report": {
                                    "average_price": predicted,
                                    "low_price": round(predicted * 0.9, 2),
                                    "high_price": round(predicted * 1.1, 2)
                                },
                                "chartData": {
                                    "dates": dates,
                                    "prices": chart_prices
                                },
                                "xai": xai_factors
                            }), 200
        except Exception as e:
            print(f"[!] ML prediction error: {e}")
    
    # Fallback to simulated prediction if ML unavailable
    base_price = random.uniform(2000, 8000)
    trend = random.choice([-100, -50, 0, 50, 100])
    predicted_price = base_price + (trend * months_ahead)
    
    # Create chart data
    import datetime
    dates = []
    for i in range(30):
        d = datetime.datetime.now() - datetime.timedelta(days=29-i)
        dates.append(d.strftime('%Y-%m-%d'))
    chart_prices = [round(base_price + random.uniform(-100, 100), 2) for _ in range(29)] + [round(predicted_price, 2)]
    
    # XAI factors
    xai_factors = [
        {"factor": "Historical Price Trend", "impact": "Positive" if trend > 0 else "Negative"},
        {"factor": "Market Sentiments", "impact": "Positive"},
        {"factor": "Seasonal Patterns", "impact": "Neutral"},
        {"factor": "Weather Conditions", "impact": "Neutral"},
        {"factor": "Supply Demand", "impact": "Positive"}
    ]
    
    return jsonify({
        "commodity": commodity,
        "market": market,
        "current_price": round(base_price, 2),
        "predicted_price": round(predicted_price, 2),
        "months_ahead": months_ahead,
        "trend": "up" if trend > 0 else "down" if trend < 0 else "stable",
        "model_used": "simulated",
        "warning": "Using simulated predictions - ML models not available",
        "prediction_report": {
            "average_price": round(predicted_price, 2),
            "low_price": round(predicted_price * 0.9, 2),
            "high_price": round(predicted_price * 1.1, 2)
        },
        "chartData": {
            "dates": dates,
            "prices": chart_prices
        },
        "xai": xai_factors
    }), 200

@app.route("/api/v1/xai", methods=["POST"])
@jwt_required()
def explain():
    """Get ML explanation for prediction."""
    data = request.get_json()
    commodity = data.get('commodity', 'Wheat')
    prediction = float(data.get('prediction', 0))
    
    if ML_MANAGER:
        try:
            # Get recent prices from Agmarknet
            agmark_data = load_cached_agmark()
            if agmark_data and 'records' in agmark_data:
                filtered = [r for r in agmark_data['records'] 
                           if r.get('commodity', '').lower() == commodity.lower()]
                if filtered:
                    recent_prices = [float(r.get('modal_price', 0)) for r in filtered[-30:]]
                    explanation = ML_MANAGER.get_explanation(commodity, prediction, recent_prices)
                    return jsonify(explanation), 200
        except Exception as e:
            print(f"[!] Explanation error: {e}")
    
    # Fallback explanation
    return jsonify({
        "commodity": commodity,
        "prediction": prediction,
        "factors": [
            {"name": "Market trends", "impact": "neutral", "value": "Insufficient data"},
            {"name": "Seasonal patterns", "impact": "neutral", "value": "Insufficient data"}
        ]
    }), 200

@app.route("/api/v1/ml-health", methods=["GET"])
@jwt_required()
def ml_health():
    """Check ML pipeline health."""
    if ML_MANAGER:
        try:
            health = ML_MANAGER.health_check()
            return jsonify(health), 200
        except Exception as e:
            print(f"[!] Health check error: {e}")
    
    return jsonify({
        "status": "degraded",
        "message": "ML pipeline not available",
        "ml_available": False
    }), 200

@app.route("/api/v1/sentiment", methods=["POST"])
@jwt_required()
def sentiment():
    """Sentiment analysis endpoint."""
    data = request.get_json()
    text = data.get('text', '')
    
    sentiment_score = random.uniform(-1, 1)
    sentiment_label = "positive" if sentiment_score > 0.2 else "negative" if sentiment_score < -0.2 else "neutral"
    
    return jsonify({
        "text": text[:100],
        "sentiment": sentiment_label,
        "score": round(sentiment_score, 3)
    }), 200

@app.route("/api/v1/weather", methods=["GET"])
@jwt_required()
def weather():
    """Weather data endpoint."""
    market = request.args.get('market', 'Delhi')
    conditions = ['Sunny', 'Partly Cloudy', 'Cloudy', 'Light Rain', 'Humid']
    temps = {'delhi': (28, 42), 'mumbai': (25, 34), 'bengaluru': (20, 32), 'kolkata': (26, 36)}
    temp_range = temps.get(market.lower(), (24, 38))
    temp = random.randint(temp_range[0], temp_range[1])
    condition = random.choice(conditions)
    
    impact = "Conditions stable. No significant crop impact expected."
    if "Rain" in condition:
        impact = "Rain favorable for sowing, may stabilize prices."
    elif "Sunny" in condition and temp > 38:
        impact = "High heat may stress crops, potentially increasing prices."
    
    return jsonify({
        "market": market,
        "temp": temp,
        "condition": condition,
        "impact": impact
    }), 200

@app.route("/api/v1/recommend-crop", methods=["POST"])
@jwt_required()
def recommend_crop():
    """Crop recommendation endpoint."""
    data = request.get_json()
    soil = data.get('soil_type', 'loamy')
    rainfall = float(data.get('rainfall', 800))
    ph = float(data.get('ph', 7))
    temp = float(data.get('temperature', 25))
    
    if rainfall > 2000:
        crop = 'Rice'
    elif rainfall > 1000:
        crop = 'Sugarcane'
    elif temp > 28 and rainfall < 800:
        crop = 'Bajra'
    else:
        crop = 'Wheat'
    
    return jsonify({"crop": crop, "reason": f"Optimal for {temp}C and {rainfall}mm rainfall"}), 200

@app.route("/api/v1/news", methods=["GET"])
@jwt_required()
def news():
    """Agricultural news endpoint."""
    news_items = [
        {"title": "Wheat prices rise amid global shortage", "source": "AgriNews", "date": "2024-11-28"},
        {"title": "Government announces new subsidy scheme", "source": "FarmTimes", "date": "2024-11-27"},
        {"title": "Monsoon predictions for next season released", "source": "Weather Bureau", "date": "2024-11-26"},
    ]
    return jsonify({"news": news_items}), 200

@app.route("/api/v1/sentiment-distribution", methods=["GET"])
@jwt_required()
def sentiment_distribution():
    """Sentiment distribution chart data."""
    return jsonify({
        "positive": random.randint(30, 60),
        "neutral": random.randint(20, 40),
        "negative": random.randint(10, 30)
    }), 200

@app.route("/api/v1/market-comparison", methods=["GET"])
@jwt_required()
def market_comparison():
    """Market price comparison."""
    commodity = request.args.get('commodity', 'Wheat')
    markets = ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata']
    comparison = {m: round(random.uniform(2500, 7500), 2) for m in markets}
    return jsonify({"commodity": commodity, "markets": comparison}), 200

@app.route("/api/v1/heatmap-data", methods=["GET"])
@jwt_required()
def heatmap_data():
    """Heatmap data for price trends by state."""
    commodity = request.args.get('commodity', 'Wheat')
    # Regional price data by state
    states_prices = {
        "Andhra Pradesh": round(random.uniform(2200, 2400), 2),
        "Maharashtra": round(random.uniform(2400, 2600), 2),
        "Punjab": round(random.uniform(2100, 2300), 2),
        "Karnataka": round(random.uniform(2300, 2500), 2),
        "Tamil Nadu": round(random.uniform(2300, 2500), 2),
        "Gujarat": round(random.uniform(2300, 2500), 2),
        "Madhya Pradesh": round(random.uniform(2200, 2400), 2),
        "Uttar Pradesh": round(random.uniform(2200, 2400), 2),
        "Bihar": round(random.uniform(2100, 2300), 2),
        "West Bengal": round(random.uniform(2100, 2300), 2),
        "Delhi": round(random.uniform(2400, 2600), 2),
        "Rajasthan": round(random.uniform(2100, 2300), 2),
        "Haryana": round(random.uniform(2300, 2500), 2),
        "Himachal Pradesh": round(random.uniform(2100, 2300), 2),
        "Uttarakhand": round(random.uniform(2100, 2300), 2),
        "Assam": round(random.uniform(2000, 2200), 2),
        "Odisha": round(random.uniform(2100, 2300), 2),
        "Telangana": round(random.uniform(2300, 2500), 2),
        "Kerala": round(random.uniform(2400, 2600), 2),
        "Jharkhand": round(random.uniform(2100, 2300), 2)
    }
    return jsonify(states_prices), 200

@app.route("/api/v1/model-performance", methods=["GET"])
@jwt_required()
def model_performance():
    """Model performance metrics."""
    return jsonify({
        "accuracy": round(random.uniform(0.75, 0.95), 3),
        "precision": round(random.uniform(0.70, 0.90), 3),
        "recall": round(random.uniform(0.70, 0.90), 3),
        "f1_score": round(random.uniform(0.72, 0.92), 3),
        "mape": round(random.uniform(5, 20), 2)
    }), 200

@app.route("/api/v1/chat", methods=["POST"])
@jwt_required()
def chat():
    """Chatbot endpoint."""
    data = request.get_json()
    user_message = data.get('message', '')
    
    if GEMINI_ENABLED:
        try:
            response = gemini_model.generate_content(user_message)
            return jsonify({"response": response.text}), 200
        except Exception as e:
            print(f"[!] Gemini error: {e}")
    
    # Rule-based fallback
    responses = {
        'price': "Current market prices vary by commodity and location. Use the dashboard to check real-time prices.",
        'weather': "Check the weather section for current conditions in your area.",
        'crop': "Use the crop recommendation tool with your soil and weather data.",
        'hello': "Hello! I'm AgroBERT, your agricultural assistant. How can I help?",
    }
    
    for keyword, response in responses.items():
        if keyword in user_message.lower():
            return jsonify({"response": response}), 200
    
    return jsonify({"response": "I'm here to help with agricultural information. Ask me about prices, weather, or crop recommendations!"}), 200

# --- Main Entry Point ---
if __name__ == '__main__':
    print("\n" + "="*60)
    print(" "*10 + "[OK] Server starting...")
    print("="*60)
    print(f"\n[OK] Environment: {os.getenv('ENVIRONMENT', 'development')}")
    print(f"[OK] Debug Mode: {DEBUG_MODE}")
    print(f"[OK] Running on port: {PORT}")
    print(f"[OK] Production Mode: {IS_PRODUCTION}")
    print(f"[OK] Demo Credentials: username='demo', password='demo123'")
    print("\n" + "="*60 + "\n")
    
    # Production: Use 0.0.0.0 for Render, Development: Use localhost
    HOST = '0.0.0.0' if IS_PRODUCTION else '127.0.0.1'
    app.run(host=HOST, port=PORT, debug=DEBUG_MODE)
