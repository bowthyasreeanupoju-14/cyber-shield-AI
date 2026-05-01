from flask import Flask, render_template
from flask_cors import CORS
from config import Config
from services import model_loader
from services.auth_service import init_db
from routes.predict   import predict_bp
from routes.alerts    import alerts_bp
from routes.analytics import analytics_bp
from routes.auth      import auth_bp
from routes.history   import history_bp

def create_app():
    app = Flask(__name__)
    app.secret_key = Config.SECRET_KEY
    CORS(app, supports_credentials=True)

    # Initialize SQLite database
    init_db()

    # Load all ML models once at startup
    model_loader.load_all_models()

    # Register blueprints
    app.register_blueprint(predict_bp)
    app.register_blueprint(alerts_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(history_bp)

    # Health check
    @app.route('/api/health', methods=['GET'])
    def health():
        from datetime import datetime
        meta = model_loader.get('metadata')
        return {
            'status':    'online',
            'model':     meta['model_type'],
            'accuracy':  f"{meta['accuracy']*100:.2f}%",
            'auc':       meta['auc'],
            'timestamp': datetime.now().isoformat()
        }

    # Serve main app
    @app.route('/')
    def index():
        return render_template('dashboard.html')

    return app


if __name__ == '__main__':
    app = create_app()
    print("\n🚀 Server running at http://localhost:5000")
    print("📊 Dashboard: http://localhost:5000")
    print("🔗 Health:    http://localhost:5000/api/health\n")
    app.run(debug=Config.DEBUG, host='0.0.0.0', port=Config.PORT)
