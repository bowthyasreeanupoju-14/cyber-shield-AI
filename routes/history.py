from flask import Blueprint, request, jsonify
from services.auth_service import (
    validate_session_token, save_check,
    update_actions, get_history, get_history_stats
)

history_bp = Blueprint('history', __name__)


def get_current_user(req):
    token = req.cookies.get('cs_token') or req.headers.get('X-Auth-Token')
    return validate_session_token(token)


# ── Save a completed check ───────────────────────────────────────
@history_bp.route('/api/history/save', methods=['POST'])
def save():
    user = get_current_user(request)
    if not user:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401

    data = request.get_json()
    hid  = save_check(
        email          = user['email'],
        crime_type     = data.get('crime_type', 'Unknown'),
        risk_score     = data.get('risk_score', 0),
        risk_level     = data.get('risk_level', 'LOW'),
        anomaly_values = data.get('anomaly_values', {}),
        shap_summary   = data.get('shap_summary', {}),
        transaction_id = data.get('transaction_id', '')
    )
    return jsonify({'success': True, 'check_id': hid})


# ── Update actions taken ─────────────────────────────────────────
@history_bp.route('/api/history/actions', methods=['POST'])
def actions():
    user = get_current_user(request)
    if not user:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401

    data = request.get_json()
    update_actions(data.get('check_id'), data.get('actions', []))
    return jsonify({'success': True})


# ── Get history for logged-in user ──────────────────────────────
@history_bp.route('/api/history', methods=['GET'])
def fetch_history():
    user = get_current_user(request)
    if not user:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401

    history = get_history(user['email'])
    stats   = get_history_stats(user['email'])
    return jsonify({
        'success': True,
        'user':    user,
        'history': history,
        'stats':   stats
    })
