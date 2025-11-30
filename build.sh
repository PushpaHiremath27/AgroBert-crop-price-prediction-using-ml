#!/bin/bash
# Build script for Render deployment

# Exit on error
set -e

echo "[OK] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[OK] Database initialization..."
python -c "from backend.app_flask import initialize_db; initialize_db()"

echo "[OK] Build completed successfully!"
