"""WSGI entry point for production deployments (Render, etc.).

Usage:
    gunicorn wsgi:app
"""

from backend.app import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)