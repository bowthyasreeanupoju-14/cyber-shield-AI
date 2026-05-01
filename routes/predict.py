from flask import Blueprint, request, jsonify
from datetime import datetime
from services.risk_calculator import calculate_risk
from services.alert_sender import dispatch_alert
from utils.validators import validate_predict_input

predict_bp = Blueprint('predict', __name__)


@predict_bp.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Validate input
        valid, error = validate_predict_input(data)
        if not valid:
            return jsonify({'success': False, 'error': error}), 400

        # Calculate risk
        result = calculate_risk(data)

        # Dispatch alerts based on risk level
        transaction_id = data.get('transaction_id', f"TXN-{datetime.now().strftime('%H%M%S')}")
        alert_result   = dispatch_alert(
            transaction_id = transaction_id,
            risk_score     = result['risk_score'],
            risk_level     = result['risk_level'],
            contributions  = result['contributions']
        )

        return jsonify({
            'success':          True,
            'transaction_id':   transaction_id,
            'risk_score':       result['risk_score'],
            'risk_level':       result['risk_level'],
            'fraud_probability':result['fraud_probability'],
            'anomaly_score':    result['anomaly_score'],
            'model_score':      result['model_score'],
            'anomaly_values':   result['anomaly_values'],
            'contributions':    result['contributions'],
            'shap_explanation': result['shap_explanation'],
            'alerts_sent':      alert_result,
            'timestamp':        datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
