"""
Skillora AI Career Advisor - Application Launcher
"""

import os
from app import app
from config import config

def create_app(config_name=None):
    """Create Flask application with configuration"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    app.config.from_object(config[config_name])
    
    return app

if __name__ == '__main__':
    # Get configuration from environment
    config_name = os.environ.get('FLASK_ENV', 'development')
    
    # Create app
    flask_app = create_app(config_name)
    
    print("🚀 Starting Skillora AI Career Advisor")
    print(f"📊 Configuration: {config_name}")
    print(f"🌐 Server: http://localhost:5000")
    print("💡 Phase 1: Rule-based system ready")
    
    # Run the application
    flask_app.run(
        debug=True if config_name == 'development' else False,
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000))
    )
