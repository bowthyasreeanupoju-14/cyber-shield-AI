import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask
    DEBUG      = os.getenv('FLASK_DEBUG', 'True') == 'True'
    PORT       = int(os.getenv('PORT', 5000))
    SECRET_KEY = os.getenv('SECRET_KEY', 'cybershield2026secretkey')

    # Model paths
    MODEL_DIR          = os.path.join(os.path.dirname(__file__), 'models')
    FRAUD_MODEL_PATH   = os.path.join(MODEL_DIR, 'fraud_model.pkl')
    FEATURES_PATH      = os.path.join(MODEL_DIR, 'feature_names.pkl')
    CONFIG_PATH        = os.path.join(MODEL_DIR, 'config.pkl')
    SCALER_PATH        = os.path.join(MODEL_DIR, 'scaler.pkl')
    ENCODERS_PATH      = os.path.join(MODEL_DIR, 'encoders.pkl')
    METADATA_PATH      = os.path.join(MODEL_DIR, 'metadata.pkl')
    SHAP_PATH          = os.path.join(MODEL_DIR, 'shap_explainer.pkl')

    # Alert log
    ALERT_LOG_FILE = os.path.join(os.path.dirname(__file__), 'alert_history.json')

    # Email
    EMAIL_SENDER       = os.getenv('EMAIL_SENDER')
    EMAIL_APP_PASSWORD = os.getenv('EMAIL_APP_PASSWORD')
    EMAIL_RECEIVER     = os.getenv('EMAIL_RECEIVER')

    # Twilio
    TWILIO_SID    = os.getenv('TWILIO_ACCOUNT_SID')
    TWILIO_TOKEN  = os.getenv('TWILIO_AUTH_TOKEN')
    TWILIO_FROM   = os.getenv('TWILIO_FROM_NUMBER')
    TWILIO_TO     = os.getenv('TWILIO_TO_NUMBER')

    # OTP
    OTP_EXPIRY_MINUTES = int(os.getenv('OTP_EXPIRY_MINUTES', 10))
