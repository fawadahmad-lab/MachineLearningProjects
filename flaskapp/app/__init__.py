from flask import Flask

def create_app():
    app = Flask(__name__)
    
    # Initialize your routes, models, and other app components
    from .routes import main
    app.register_blueprint(main)

    return app
