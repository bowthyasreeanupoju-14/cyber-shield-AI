import numpy as np
from services import model_loader


def build_feature_vector(data: dict) -> np.ndarray:
    features   = model_loader.get('features')
    encoders   = model_loader.get('encoders')

    amt        = float(data.get('amount', 100))
    amt_mean   = 100.0
    amt_std    = 50.0
    amt_zscore = (amt - amt_mean) / amt_std

    card1_raw  = str(data.get('card1', '0'))
    card1_enc  = 0
    if encoders and 'card1_encoder' in encoders:
        try:
            card1_enc = int(encoders['card1_encoder'].transform([card1_raw])[0])
        except Exception:
            card1_enc = 0

    card2_raw    = data.get('card2', None)
    card2_median = encoders.get('card2_median', 361.0) if encoders else 361.0
    card2_filled = float(card2_raw) if card2_raw is not None else card2_median

    values = {
        'credential_anomaly': int(data.get('credential_anomaly', 0)),
        'device_anomaly':     int(data.get('device_anomaly', 0)),
        'temporal_anomaly':   int(data.get('temporal_anomaly', 0)),
        'behavioral_anomaly': int(data.get('behavioral_anomaly', 0)),
        'geospatial_anomaly': int(data.get('geospatial_anomaly', 0)),
        'TransactionAmt':     amt,
        'amt_zscore':         amt_zscore,
        'card1_enc':          card1_enc,
        'card2_filled':       card2_filled,
    }

    return np.array([[values[f] for f in features]])


def calculate_risk(data: dict) -> dict:
    model    = model_loader.get('model')
    config   = model_loader.get('config')
    shap_exp = model_loader.get('shap')

    normalized_weights = config['normalized_weights']
    max_scores         = config['max_scores']
    thresholds         = config['thresholds']
    anomaly_features   = config['anomaly_features']

    feature_vector = build_feature_vector(data)

    fraud_prob  = float(model.predict_proba(feature_vector)[0][1])
    model_score = fraud_prob * 100

    anomaly_vals = {
        'credential_anomaly': int(data.get('credential_anomaly', 0)),
        'device_anomaly':     int(data.get('device_anomaly', 0)),
        'temporal_anomaly':   int(data.get('temporal_anomaly', 0)),
        'behavioral_anomaly': int(data.get('behavioral_anomaly', 0)),
        'geospatial_anomaly': int(data.get('geospatial_anomaly', 0)),
    }

    max_possible = sum([
        max_scores[feat] * normalized_weights[feat]
        for feat in anomaly_features
    ])
    weighted_sum = sum([
        anomaly_vals[feat] * normalized_weights[feat]
        for feat in anomaly_features
    ])
    anomaly_score = (weighted_sum / max_possible * 100) if max_possible > 0 else 0

    # Raw hybrid score used for threshold comparison
    final_risk = (0.6 * model_score) + (0.4 * anomaly_score)

    # ── Determine risk level using raw thresholds from training ──
    if final_risk >= thresholds['critical']:
        raw_level = 'CRITICAL'
    elif final_risk >= thresholds['high']:
        raw_level = 'HIGH'
    elif final_risk >= thresholds['medium']:
        raw_level = 'MEDIUM'
    else:
        raw_level = 'LOW'

    # ── Map raw level to display score range ─────────────────────
    # LOW      →  5 to 20
    # MEDIUM   → 20 to 50
    # HIGH     → 50 to 80
    # CRITICAL → 80 to 100
    #
    # Within each band, position is proportional to where the raw
    # score sits between that band's raw thresholds.

    raw_medium   = thresholds['medium']    # ~13.22
    raw_high     = thresholds['high']      # ~18.83
    raw_critical = thresholds['critical']  # ~29.30
    raw_max      = 50.0                    # practical max raw score

    if raw_level == 'LOW':
        # raw range: 0 → raw_medium  maps to display 5 → 20
        ratio         = final_risk / raw_medium if raw_medium > 0 else 0
        display_score = 5 + ratio * (20 - 5)

    elif raw_level == 'MEDIUM':
        # raw range: raw_medium → raw_high  maps to display 20 → 50
        ratio         = (final_risk - raw_medium) / (raw_high - raw_medium)
        display_score = 20 + ratio * (50 - 20)

    elif raw_level == 'HIGH':
        # raw range: raw_high → raw_critical  maps to display 50 → 80
        ratio         = (final_risk - raw_high) / (raw_critical - raw_high)
        display_score = 50 + ratio * (80 - 50)

    else:  # CRITICAL
        # raw range: raw_critical → raw_max  maps to display 80 → 100
        ratio         = (final_risk - raw_critical) / (raw_max - raw_critical)
        display_score = 80 + ratio * (100 - 80)

    display_score = round(min(max(display_score, 1), 100), 1)
    risk_level    = raw_level

    contributions = {}
    if weighted_sum > 0:
        for feat in anomaly_features:
            short = feat.replace('_anomaly', '')
            contributions[short] = round(
                (anomaly_vals[feat] * normalized_weights[feat] / weighted_sum) * 100, 2
            )
    else:
        for feat in anomaly_features:
            contributions[feat.replace('_anomaly', '')] = 0.0

    shap_explanation = _get_shap_explanation(feature_vector, shap_exp)

    return {
        'risk_score':        display_score,
        'raw_risk_score':    round(final_risk, 2),
        'risk_level':        risk_level,
        'fraud_probability': round(fraud_prob * 100, 2),
        'anomaly_score':     round(anomaly_score, 2),
        'model_score':       round(model_score, 2),
        'anomaly_values':    anomaly_vals,
        'contributions':     contributions,
        'shap_explanation':  shap_explanation,
        'thresholds':        thresholds,
    }


def _get_shap_explanation(feature_vector: np.ndarray, explainer) -> dict:
    try:
        features  = model_loader.get('features')
        shap_vals = explainer.shap_values(feature_vector)

        if isinstance(shap_vals, list):
            fraud_shap = shap_vals[1][0]
        else:
            fraud_shap = shap_vals[0]

        paired = sorted(zip(features, fraud_shap), key=lambda x: x[1], reverse=True)

        pushing_fraud = [
            {'feature': n.replace('_', ' ').title(), 'impact': round(float(v) * 100, 2)}
            for n, v in paired if v > 0
        ][:3]

        pushing_legit = [
            {'feature': n.replace('_', ' ').title(), 'impact': round(float(v) * 100, 2)}
            for n, v in paired if v < 0
        ][:1]

        return {
            'pushing_to_fraud': pushing_fraud,
            'pushing_to_legit': pushing_legit
        }
    except Exception as e:
        return {'error': str(e), 'pushing_to_fraud': [], 'pushing_to_legit': []}
