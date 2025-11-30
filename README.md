# AgroBERT: Advanced Agri-Product Price Prediction Using ML

![Project Status](https://img.shields.io/badge/status-active-brightgreen)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

**AgroBERT** is an intelligent agricultural decision support system designed to forecast commodity prices using advanced deep learning and data analytics. The platform leverages diverse data sources including historical market prices, real-time weather information, global agricultural statistics, and market sentiment analysis to provide accurate, actionable insights for farmers, traders, policymakers, and agribusiness stakeholders.

The system integrates:
- **Hybrid Deep Learning Models** (DistilBERT-LSTM, Prophet)
- **Fine-tuned Large Language Models** (Falcon-7B, Mistral)
- **Explainable AI (XAI)** using SHAP for transparency
- **Multilingual Support** (English, Hindi, Kannada)
- **Voice Query Capabilities** via Web Speech Recognition API
- **Interactive Dashboard** with real-time visualizations

---

## Key Features

### Core Functionality
- **Real-Time Price Prediction**: Hybrid deep learning models for accurate crop price forecasting
- **Sentiment Analysis**: NLP-powered analysis of agricultural news and market sentiment
- **Explainable AI (XAI)**: SHAP-based visualizations showing key price drivers
- **Weather Impact Analysis**: Integration of weather data to understand market dynamics
- **Regional Analysis**: Geospatial heatmaps and regional price comparisons
- **Crop Recommendations**: ML-based suggestions based on soil, climate, and location
- **Automated Alerts**: Real-time notifications for significant price changes

### User Interface
- **Multilingual Dashboard**: English, Hindi, and Kannada support
- **Voice Input**: Hands-free query capability via speech recognition
- **Interactive Visualizations**: 
  - Price trend charts with confidence intervals
  - Feature importance charts
  - Sentiment distribution pie charts
  - Regional price heatmaps
- **Chat Interface**: AgriBERT multilingual chatbot for queries

### Data Integration
- **Agmarknet API**: Historical APMC mandi prices
- **OpenWeather API**: Real-time and forecast weather data
- **FAOSTAT**: Global agricultural production and trade data
- **Web Scraping**: News sentiment and market intelligence
- **Government Portals**: State agriculture department data

---

## System Architecture

<div align="center">
  <img src="./assets/system_architecture.png" alt="System Architecture" width="800" />
</div>

---

## Data Sources

Our platform integrates multiple data sources to ensure comprehensive market intelligence:

<div align="center">
  <img src="./assets/data_sources.png" alt="Data Sources" width="800" />
</div>


---

## Requirements

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: Minimum 8GB (16GB recommended for LLM fine-tuning)
- **Storage**: 10GB for models and datasets
- **OS**: Windows, macOS, or Linux

### Python Dependencies
```
Backend & ML:
- Flask==2.3.0
- pandas==1.5.3
- numpy==1.24.3
- scikit-learn==1.3.0
- TensorFlow==2.13.0
- PyTorch==2.0.0
- transformers==4.30.0
- prophet==1.1.4
- shap==0.42.0

Frontend:
- plotly
- D3.js
- Chart.js
- Tailwind CSS

APIs & Utilities:
- requests==2.31.0
- beautifulsoup4==4.12.0
- scrapy==2.9.0
- python-dotenv==1.0.0
```

See `requirements.txt` and `backend/models_ml/ml_requirements.txt` for complete dependency lists.

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/AgroBERT.git
cd AgroBERT
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Core requirements
pip install -r requirements.txt

# ML-specific requirements
pip install -r backend/models_ml/ml_requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the project root:
```env
# API Keys
OPENWEATHER_API_KEY=your_api_key_here
FAOSTAT_API_KEY=your_api_key_here

# Database
DATABASE_URL=postgresql://user:password@localhost/agrobert_db

# Flask Configuration
FLASK_ENV=development
SECRET_KEY=your_secret_key_here

# Server
PORT=3000
DEBUG=True
```

### 5. Database Setup
```bash
# Initialize database (if applicable)
python backend/scripts/init_db.py
```

### 6. Run the Application

#### Option A: Using start_all.bat (Windows)
```bash
start_all.bat
```

#### Option B: Manual Startup
```bash
# Terminal 1: Start Flask Backend
cd backend
python app_flask.py

# Terminal 2: Start Frontend Server
cd frontend
python serve.py
```

#### Access the Application
- **Frontend**: http://127.0.0.1:3000
- **Test Credentials**: 
  - Username: `demo`
  - Password: `demo123`

---

## Features Guide

### Price Prediction
1. Navigate to **Dashboard** tab
2. Select **Commodity** (Wheat, Rice, Tomato, etc.)
3. Choose **Market/Region**
4. Set **Days Ahead** for forecast
5. Click **Predict Price**
6. View predictions with confidence intervals and key drivers

### Sentiment Analysis
1. Go to **Dashboard** → **Sentiment Analysis**
2. Paste or type agricultural news text
3. Click **Analyze Sentiment**
4. View sentiment classification (Positive/Neutral/Negative) with confidence score

### Crop Recommendation
1. Navigate to **Dashboard** → **Crop Recommendation System**
2. Enter your **Soil Type**, **Rainfall**, **pH Level**, and **Temperature**
3. Click **Get Recommendation**
4. Receive personalized crop suggestions with rationale

### Analytics & Reports
1. **Analytics Tab**: View performance metrics, feature importance, and sentiment distribution
2. **Regional Analysis**: Compare prices across major APMC markets
3. **Reports**: Generate custom PDF reports based on predictions

### Multilingual Interface
- **Language Selector**: Top-right dropdown
- **Supported Languages**: English, Hindi (हिंदी), Kannada (ಕನ್ನಡ)
- **Voice Input**: Click microphone icon to speak your query

### Chatbot (AgriBERT)
- Ask questions in natural language
- Responses intelligently routed based on query type
- Multilingual responses in selected language
- Topics: prices, weather, soil, markets, sentiment, crops, recommendations

---

## Machine Learning Models

### Time-Series Forecasting
- **Hybrid DistilBERT-LSTM**: Combines textual embeddings with temporal patterns
- **Prophet**: Robust baseline with strong seasonality handling
- **ARIMA**: Classical statistical approach for validation

### Natural Language Processing
- **DistilBERT**: For agricultural news sentiment extraction
- **Falcon-7B / Mistral**: Fine-tuned for Indian agricultural context
- **Multilingual BERT**: For Hindi and Kannada query processing

### Explainability
- **SHAP (SHapley Additive exPlanations)**: 
  - Feature importance for predictions
  - Local explanation of individual predictions
  - Force plots and decision plots

### Model Performance Metrics
- **R² (Coefficient of Determination)**: Target > 0.8
- **MAE (Mean Absolute Error)**: < 5% of average price
- **RMSE (Root Mean Squared Error)**: Minimized across all commodities

---

## Project Structure

```
AgroBERT/
├── frontend/
│   ├── index.html              # Main UI application
│   ├── serve.py                # Frontend server
│   └── users.json              # User credentials
├── backend/
│   ├── app_flask.py            # Flask backend application
│   ├── models_ml/
│   │   ├── config.py           # Model configuration
│   │   ├── models.py           # Model architectures
│   │   ├── preprocessing.py    # Data preprocessing
│   │   ├── inference.py        # Model inference
│   │   ├── ml_integration.py   # ML pipeline integration
│   │   ├── explainer.py        # XAI/SHAP integration
│   │   ├── train.py            # Model training
│   │   └── ml_requirements.txt # ML dependencies
│   ├── scripts/
│   │   ├── collect_data.py     # Data collection
│   │   ├── health_check.py     # System health monitoring
│   │   └── init_db.py          # Database initialization
│   ├── db/                     # Database files
│   └── logs/                   # Application logs
├── db/                         # Standalone database storage
├── requirements.txt            # Core dependencies
├── Procfile                    # Deployment configuration
├── start_all.bat               # Windows startup script
├── cleanup.ps1                 # Windows cleanup script
└── README.md                   # This file
```

---

## Security & Privacy

### Authentication
- Secure credential storage with hashed passwords
- Session-based authentication tokens
- API key management via environment variables

### Data Protection
- User data encrypted at rest and in transit
- API keys never exposed in frontend code
- Rate limiting on sensitive endpoints

### Best Practices
- Regular dependency updates
- Input validation and sanitization
- CORS configuration for cross-origin requests
- Comprehensive error handling


## API Endpoints

### Authentication
```
POST /api/v1/login
GET  /api/v1/logout
GET  /api/v1/verify_otp
```

### Predictions
```
POST /api/v1/predict_price
POST /api/v1/analyze_sentiment
POST /api/v1/recommend_crop
GET  /api/v1/historical_prices
```

### Analytics
```
GET  /api/v1/market_trends
GET  /api/v1/regional_analysis
GET  /api/v1/weather_impact
GET  /api/v1/feature_importance
```

### Chat & Reports
```
POST /api/v1/chat
POST /api/v1/generate_report
GET  /api/v1/alerts
```

---

## Testing

### Run Unit Tests
```bash
pytest backend/tests/ -v
```

### Integration Tests
```bash
pytest backend/tests/integration/ -v
```

### Performance Tests
```bash
python backend/scripts/performance_test.py
```

---

## Data Sources

| Source | Type | Update Frequency | Coverage |
|--------|------|------------------|----------|
| Agmarknet | Historical Prices | Daily | All India APMC Markets |
| OpenWeather API | Weather Data | Real-time | Global Coverage |
| FAOSTAT | Agricultural Stats | Monthly | Global Data |
| Web Scraping | News/Sentiment | Real-time | Agricultural News |
| State Agriculture | Local Prices | Daily | State-level Data |

---

## About the Project

**AgroBERT** is developed as a Phase I Project in partial fulfillment of the Bachelor of Engineering degree in Artificial Intelligence & Machine Learning from Visvesvaraya Technological University (VTU).

### Project Details
- **Institution**: Bapuji Institute of Engineering and Technology (BIET), Davanagere
- **Department**: Artificial Intelligence & Machine Learning
- **Program**: B.E. AI & ML (4-year)
- **Academic Year**: 2024-25
- **Project Code**: BAI685

### Development Team
This is a collaborative effort by final-year engineering students in the AI & ML program. The team brings together diverse expertise in machine learning, software engineering, data analysis, and full-stack development.

---

## Technologies & Tools

### Machine Learning & Data Science
- TensorFlow/Keras, PyTorch
- Scikit-learn, XGBoost
- Pandas, NumPy
- SHAP, Plotly

### Backend
- Python 3.8+
- Flask
- PostgreSQL/MongoDB

### Frontend
- HTML5, CSS3, JavaScript
- Tailwind CSS
- Chart.js, D3.js
- Web Speech Recognition API

### DevOps & Deployment
- Git/GitHub
- AWS/Cloud Platforms
- CI/CD Pipelines

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Links & Resources

- **Project Repository**: [GitHub Link]
- **Documentation**: [Doc Link]
- **Live Demo**: [Demo Link]
- **Issue Tracker**: [GitHub Issues]

### References
- TensorFlow Documentation: https://www.tensorflow.org/
- Hugging Face Transformers: https://huggingface.co/transformers/
- SHAP Documentation: https://shap.readthedocs.io/
- Flask Documentation: https://flask.palletsprojects.com/

---

## Support & Contact

For questions, issues, or suggestions:
- **GitHub Issues**: Open an issue on the repository
- **Email**: Submit inquiry through official channels
- **Documentation**: Refer to the comprehensive docs folder

---

## Changelog

### Version 1.0.0 (Current)
- Core price prediction system
- Multilingual UI (EN, HI, KN)
- Sentiment analysis from news
- Explainable AI with SHAP
- Interactive dashboard with visualizations
- Voice query support
- Multilingual chatbot
- Crop recommendation system
- Regional analysis and heatmaps
- Automated alerts and reports

### Planned Features (Future)
- [ ] Mobile app (iOS/Android)
- [ ] Advanced IoT sensor integration
- [ ] Enhanced LLM fine-tuning
- [ ] Real-time price streaming
- [ ] Blockchain-based price verification
- [ ] Extended language support

---

## Acknowledgments

We extend our heartfelt thanks to:
- **Bapuji Institute of Engineering and Technology** for providing resources and facilities
- **Department of AI & ML** for academic guidance
- **All mentors and guides** for their invaluable support
- **Open-source community** for excellent tools and libraries

---

**Built with by the AgroBERT Development Team**

*Last Updated: November 2025*
