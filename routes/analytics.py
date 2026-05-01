from flask import Blueprint, jsonify
from services import model_loader
from services.alert_sender import get_alert_history

analytics_bp = Blueprint('analytics', __name__)


@analytics_bp.route('/api/analytics/model', methods=['GET'])
def model_info():
    metadata = model_loader.get('metadata')
    config   = model_loader.get('config')
    return jsonify({
        'success':    True,
        'model_type': metadata['model_type'],
        'accuracy':   round(metadata['accuracy'] * 100, 2),
        'auc':        round(metadata['auc'], 4),
        'features':   metadata['features'],
        'fraud_rate': round(metadata['fraud_rate'] * 100, 2),
        'thresholds': config['thresholds'],
        'feature_weights': config['normalized_weights'],
    })


@analytics_bp.route('/api/analytics/summary', methods=['GET'])
def summary():
    alerts   = get_alert_history()
    total    = len(alerts)
    critical = sum(1 for a in alerts if a['risk_level'] == 'CRITICAL')
    high     = sum(1 for a in alerts if a['risk_level'] == 'HIGH')
    medium   = sum(1 for a in alerts if a['risk_level'] == 'MEDIUM')
    low      = sum(1 for a in alerts if a['risk_level'] == 'LOW')

    avg_score = (
        round(sum(a['risk_score'] for a in alerts) / total, 2)
        if total > 0 else 0
    )

    return jsonify({
        'success':   True,
        'total_checked': total,
        'by_level': {
            'CRITICAL': critical,
            'HIGH':     high,
            'MEDIUM':   medium,
            'LOW':      low,
        },
        'avg_risk_score': avg_score,
    })
