def validate_predict_input(data: dict) -> tuple[bool, str]:
    """
    Validates incoming prediction request.
    Returns (is_valid, error_message)
    """
    if not data:
        return False, "No data provided"

    # Anomaly scores must be integers 0-3 (temporal max is 2)
    anomaly_limits = {
        'credential_anomaly': 3,
        'device_anomaly':     3,
        'temporal_anomaly':   2,
        'behavioral_anomaly': 2,
        'geospatial_anomaly': 3,
    }

    for field, max_val in anomaly_limits.items():
        val = data.get(field)
        if val is not None:
            try:
                val = int(val)
            except (TypeError, ValueError):
                return False, f"{field} must be an integer"
            if val < 0 or val > max_val:
                return False, f"{field} must be between 0 and {max_val}"

    # Amount must be positive if provided
    amount = data.get('amount')
    if amount is not None:
        try:
            amount = float(amount)
            if amount < 0:
                return False, "amount must be a positive number"
        except (TypeError, ValueError):
            return False, "amount must be a number"

    return True, ""
