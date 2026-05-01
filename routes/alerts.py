from flask import Blueprint, jsonify
from services.alert_sender import get_alert_history

alerts_bp = Blueprint('alerts', __name__)


@alerts_bp.route('/api/alerts/history', methods=['GET'])
def history():
    alerts = get_alert_history()
    return jsonify({'success': True, 'count': len(alerts), 'alerts': alerts})


@alerts_bp.route('/api/alerts/stats', methods=['GET'])
def stats():
    alerts = get_alert_history()
    total     = len(alerts)
    critical  = sum(1 for a in alerts if a['risk_level'] == 'CRITICAL')
    high      = sum(1 for a in alerts if a['risk_level'] == 'HIGH')
    medium    = sum(1 for a in alerts if a['risk_level'] == 'MEDIUM')
    emails    = sum(1 for a in alerts if a.get('email_sent'))
    sms_count = sum(1 for a in alerts if a.get('sms_sent'))

    return jsonify({
        'success':  True,
        'total':    total,
        'critical': critical,
        'high':     high,
        'medium':   medium,
        'emails_sent': emails,
        'sms_sent':    sms_count,
    })
