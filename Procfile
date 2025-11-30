web: gunicorn -w 4 -b 0.0.0.0:$PORT backend.app_flask:app
release: python -c "from backend.app_flask import initialize_db; initialize_db()"
