"""
AgroBERT Simplified Server - Fixed Authentication with Persistent Storage
"""
import json
import math
import random
import datetime
from flask import Flask, send_file, jsonify, request
from flask_cors import CORS
import os

# Get current directory
base_dir = os.path.dirname(os.path.abspath(__file__))

def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two points on Earth (kilometres)"""
    R = 6371.0  # Earth radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

app = Flask(__name__)
CORS(app)

# --- USER STORAGE LOGIC START ---
USER_DB_FILE = os.path.join(base_dir, 'users.json')

def load_users():
    """Load users from JSON file so they persist after restart"""
    if os.path.exists(USER_DB_FILE):
        try:
            with open(USER_DB_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    # Default users if file doesn't exist or error occurs
    return {'demo': 'demo123', 'test': 'test123'}

def save_users(users):
    """Save users to JSON file"""
    try:
        with open(USER_DB_FILE, 'w') as f:
            json.dump(users, f, indent=4)
    except Exception as e:
        print(f"Error saving users: {e}")

# Load users into memory on startup
USERS = load_users()
# --- USER STORAGE LOGIC END ---

def generate_mock_price_history(base_price=2200, days=30):
    """Generates mock historical price data"""
    dates = [(datetime.datetime.now() - datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]
    dates.reverse()
    prices = [round(base_price * (0.9 + random.random() * 0.2), 2) for _ in range(days)]
    return {"dates": dates, "prices": prices}

# FRONTEND ROUTES
@app.route('/')
def index():
    """Serve the main HTML file"""
    try:
        return send_file(os.path.join(base_dir, 'index.html'), mimetype='text/html')
    except Exception as e:
        return jsonify({"error": str(e)}), 404

@app.route('/<path:path>')
def static_files(path):
    """Serve static files"""
    try:
        file_path = os.path.join(base_dir, path)
        if os.path.isfile(file_path):
            return send_file(file_path)
    except Exception:
        pass
    return index()

# API ENDPOINTS
@app.route('/api/login', methods=['POST'])
@app.route('/api/v1/login', methods=['POST'])
def login():
    """Simple login endpoint"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        print(f"[LOGIN] Attempt: {username}")
        
        if not username or not password:
            return jsonify({'msg': 'Username and password required'}), 400
        
        # Check credentials
        if username in USERS and USERS[username] == password:
            print(f"[LOGIN] Success: {username}")
            return jsonify({
                'access_token': f'token_{username}_{int(datetime.datetime.now().timestamp())}',
                'user': {'username': username}
            }), 200
        
        print(f"[LOGIN] Failed: Invalid credentials")
        return jsonify({'msg': 'Invalid credentials'}), 401
    except Exception as e:
        print(f"[LOGIN] Error: {str(e)}")
        return jsonify({'msg': str(e)}), 500

@app.route('/api/register', methods=['POST'])
@app.route('/api/v1/register', methods=['POST'])
def register():
    """Simple register endpoint"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        email = data.get('email', '').strip()
        mobile = data.get('mobile', '').strip()
        
        if not all([username, password, email, mobile]):
            return jsonify({'msg': 'All fields required'}), 400
        
        if len(password) < 6:
            return jsonify({'msg': 'Password must be 6+ characters'}), 400
        
        if username in USERS:
            return jsonify({'msg': 'Username already exists'}), 409
        
        # Add new user
        USERS[username] = password
        
        # SAVE users to file immediately
        save_users(USERS)
        
        print(f"[REGISTER] New user: {username}")
        return jsonify({'msg': 'Account created! You can now login.'}), 201
    except Exception as e:
        return jsonify({'msg': str(e)}), 500

@app.route('/api/send-otp', methods=['POST'])
@app.route('/api/v1/send-otp', methods=['POST'])
def send_otp():
    """Send OTP endpoint"""
    identifier = request.get_json().get('identifier', '')
    otp = ''.join([str(random.randint(0, 9)) for _ in range(6)])
    print(f"[OTP] Generated for {identifier}: {otp}")
    return jsonify({'msg': 'OTP sent (check console)'}), 200

@app.route('/api/reset-password', methods=['POST'])
@app.route('/api/v1/reset-password', methods=['POST'])
def reset_password():
    """Reset password endpoint"""
    data = request.get_json()
    username = data.get('username', '').strip()
    new_password = data.get('new_password', '').strip()
    
    if username in USERS:
        USERS[username] = new_password
        
        # SAVE updated password to file
        save_users(USERS)
        
        return jsonify({'msg': 'Password reset successful'}), 200
    
    return jsonify({'msg': 'User not found'}), 404

@app.route('/api/predict', methods=['POST'])
@app.route('/api/v1/predict', methods=['POST'])
def predict_price():
    """Improved Price prediction endpoint with mock metrics and XAI"""
    data = request.get_json() or {}
    commodity = data.get('commodity', 'Wheat')
    market = data.get('market', 'Bengaluru')
    days_ahead = int(data.get('daysAhead', 7))
    
    base_prices = {'Wheat': 2200, 'Rice': 3000, 'Cotton': 6000, 'Onion': 1500, 'Tomato': 1200, 'Maize': 1800}
    base = base_prices.get(commodity, 2000)
    
    # Generate mock historical series
    history_days = 30
    history_dates = [(datetime.datetime.now() - datetime.timedelta(days=(history_days - i))).strftime("%Y-%m-%d") for i in range(history_days)]
    history_prices = [round(base * (0.9 + random.random() * 0.2), 2) for _ in range(history_days)]
    
    # Predict simple continuation + small noise
    predicted_base = history_prices[-1] * (1 + random.uniform(-0.03, 0.04))
    future_dates = [(datetime.datetime.now() + datetime.timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(days_ahead)]
    future_prices = [round(predicted_base * (1 + random.uniform(-0.02, 0.02)), 2) for _ in range(days_ahead)]
    
    # Mock XAI: numeric importances and sign
    xai = [
        {"factor": "Monsoon Rainfall", "impact": "Positive", "importance": round(random.uniform(0.2, 0.4), 3)},
        {"factor": "Mandi Arrivals", "impact": "Negative", "importance": round(random.uniform(0.05, 0.2), 3)},
        {"factor": "Historical Price Trend", "impact": "Positive", "importance": round(random.uniform(0.15, 0.35), 3)},
        {"factor": "Fuel Cost", "impact": "Negative", "importance": round(random.uniform(0.01, 0.1), 3)},
        {"factor": "News Sentiment", "impact": "Positive", "importance": round(random.uniform(0.01, 0.08), 3)}
    ]
    
    # Mock model performance metrics
    baseline_rmse = round(random.uniform(120, 180), 2)
    agrobert_rmse = round(random.uniform(30, 70), 2)
    mae = round(random.uniform(20, 80), 2)
    r2 = round(random.uniform(0.5, 0.9), 3)
    
    prediction_report = {
        "average_price": round(sum(future_prices) / len(future_prices), 2),
        "low_price": min(future_prices),
        "high_price": max(future_prices)
    }
    
    return jsonify({
        "prediction": {
            "predicted_modal_price_INR": round(predicted_base, 2),
            "trend": "Positive" if predicted_base > history_prices[-1] else "Negative"
        },
        "prediction_report": prediction_report,
        "chartData": {
            "dates": history_dates + future_dates,
            "prices": history_prices + future_prices
        },
        "xai": xai,
        "model_performance": {
            "baseline_rmse": baseline_rmse,
            "agrobert_rmse": agrobert_rmse,
            "mae": mae,
            "r2": r2
        }
    }), 200

@app.route('/api/weather', methods=['GET'])
@app.route('/api/v1/weather', methods=['GET'])
def get_weather():
    """Weather endpoint"""
    market = request.args.get('market', 'Delhi')
    conditions = ["Sunny", "Cloudy", "Rainy", "Clear", "Partly Cloudy"]
    return jsonify({
        "market": market,
        "temp": random.randint(20, 35),
        "condition": random.choice(conditions),
        "impact": "Favorable for harvest"
    }), 200

@app.route('/api/heatmap-data', methods=['GET'])
@app.route('/api/v1/heatmap-data', methods=['GET'])
def get_heatmap_data():
    """Heatmap data endpoint"""
    commodity = request.args.get('commodity', 'Wheat')
    states = [
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", 
        "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", 
        "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", 
        "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", 
        "Uttarakhand", "West Bengal", "Delhi", "Jammu and Kashmir"
    ]
    
    base_prices = {'Wheat': 2200, 'Rice': 3000, 'Cotton': 6000, 'Onion': 1500, 'Tomato': 1200}
    base_price = base_prices.get(commodity, 2000)
    
    heatmap_data = {}
    for state in states:
        variance = random.uniform(0.85, 1.15)
        heatmap_data[state] = round(base_price * variance, 2)
    
    return jsonify(heatmap_data), 200

@app.route('/api/analyze-sentiment', methods=['POST'])
@app.route('/api/v1/analyze-sentiment', methods=['POST'])
def analyze_sentiment():
    """Sentiment analysis endpoint"""
    score = random.random()
    sentiment = "Positive" if score > 0.6 else "Negative" if score < 0.4 else "Neutral"
    return jsonify({"sentiment": sentiment, "score": score}), 200

@app.route('/api/recommend-crop', methods=['POST'])
@app.route('/api/v1/recommend-crop', methods=['POST'])
def recommend_crop():
    """Crop recommendation endpoint"""
    data = request.get_json()
    soil = data.get('soil', '').split()[0] if data.get('soil') else 'Black'
    
    crop_map = {'Black': 'Cotton', 'Alluvial': 'Wheat', 'Red': 'Groundnut', 'Laterite': 'Tea'}
    crop = crop_map.get(soil, 'Maize')
    
    return jsonify({
        "crop": crop,
        "reason": f"Best suited for {soil} soil based on NPK values."
    }), 200

@app.route('/api/model-performance', methods=['GET'])
@app.route('/api/v1/model-performance', methods=['GET'])
def model_performance():
    """Model performance endpoint"""
    return jsonify({
        "baseline_rmse": "150.50",
        "agrobert_rmse": "45.20"
    }), 200

@app.route('/api/geolocation', methods=['POST'])
@app.route('/api/location-based-markets', methods=['POST'])
@app.route('/api/v1/geolocation', methods=['POST'])
@app.route('/api/v1/location-based-markets', methods=['POST'])
def get_geolocation():
    """Geolocation endpoint - returns nearest market object with distance (km) using haversine"""
    data = request.get_json() or {}
    latitude = float(data.get('latitude', 28.7041))
    longitude = float(data.get('longitude', 77.1025))
    commodity = data.get('commodity', 'Wheat')
    
    # Markets list with coordinates - expanded with more markets including regional ones
    markets_locations = {
        'Delhi': (28.7041, 77.1025),
        'Mumbai': (19.0760, 72.8777),
        'Bengaluru': (12.9716, 77.5946),
        'Pune': (18.5204, 73.8567),
        'Lucknow': (26.8467, 80.9462),
        'Mysuru': (12.2958, 76.6394),
        'Mangalore': (12.9141, 74.8560),
        'Hubli': (15.3647, 75.1237),
        'Mandya': (12.5234, 76.8956),
        'Hassan': (13.2033, 75.9165),
        'Davangere': (14.2386, 76.3740),
        'Hosapete': (15.2631, 76.3871),
        'Kolar': (13.1457, 78.1289),
    }
    
    nearest = None
    min_dist = float('inf')
    for mname, (mlat, mlon) in markets_locations.items():
        d = haversine_km(latitude, longitude, mlat, mlon)
        if d < min_dist:
            min_dist = d
            nearest = {'name': mname, 'lat': mlat, 'lng': mlon, 'distance_km': round(d, 3)}
    
    return jsonify({
        "user_location": {"latitude": latitude, "longitude": longitude},
        "nearest_market": nearest,
        "commodity": commodity
    }), 200

@app.route('/api/news', methods=['GET'])
@app.route('/api/v1/news', methods=['GET'])
def get_news():
    """News endpoint"""
    return jsonify([
        "Government raises MSP for Rabi crops.",
        "Heavy rains expected in Southern India.",
        "Export ban on onions lifted."
    ]), 200

@app.route('/api/chat', methods=['POST'])
@app.route('/api/v1/chat', methods=['POST'])
def chat():
    """Chatbot endpoint"""
    data = request.get_json()
    query = data.get('query', '')
    return jsonify({
        "response": f"I understand you asked about '{query}'. Agricultural prices are stable."
    }), 200

if __name__ == '__main__':
    print("\n" + "="*60)
    print("AgroBERT Simplified Server Starting")
    print("="*60)
    print("Frontend: http://127.0.0.1:3000")
    print("Test Credentials:")
    print("  Username: demo")
    print("  Password: demo123")
    print("="*60 + "\n")
    
    # Check if we have any saved users loaded
    if len(USERS) > 2:
        print(f"Loaded {len(USERS) - 2} additional user account(s) from storage.")
    
    app.run(host='127.0.0.1', port=3000, debug=True, use_reloader=False)
