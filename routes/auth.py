import smtplib
import random
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import Blueprint, request, jsonify, make_response
from config import Config
from services.auth_service import (
    generate_otp, store_otp, verify_otp,
    get_or_create_user, create_session_token,
    validate_session_token, delete_session
)

auth_bp = Blueprint('auth', __name__)


# ── Helper: send OTP email ───────────────────────────────────────
def send_otp_email(to_email, otp, name):
    try:
        html = f"""
        <html><body style='font-family:Arial;background:#f4f4f4;padding:30px'>
        <div style='max-width:480px;margin:auto;background:white;border-radius:12px;
                    padding:32px;border-top:4px solid #0D7377'>
            <h2 style='color:#0D7377;margin:0 0 8px'>CyberShield</h2>
            <p style='color:#666;margin:0 0 24px;font-size:14px'>Your security companion</p>
            <p style='color:#2C3E50;font-size:16px'>Hi {name},</p>
            <p style='color:#2C3E50;font-size:15px'>Your one-time password (OTP) is:</p>
            <div style='text-align:center;margin:24px 0'>
                <span style='font-size:42px;font-weight:bold;letter-spacing:12px;
                             color:#0D7377;background:#f0fafa;padding:16px 28px;
                             border-radius:8px;display:inline-block'>{otp}</span>
            </div>
            <p style='color:#888;font-size:13px;text-align:center'>
                Valid for 10 minutes. Do not share this with anyone.
            </p>
            <hr style='border:none;border-top:1px solid #eee;margin:24px 0'>
            <p style='color:#bbb;font-size:12px;text-align:center'>
                CyberShield — Protecting you from cybercrime
            </p>
        </div></body></html>"""

        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"CyberShield OTP: {otp}"
        msg['From']    = Config.EMAIL_SENDER
        msg['To']      = to_email
        msg.attach(MIMEText(html, 'html'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(Config.EMAIL_SENDER, Config.EMAIL_APP_PASSWORD)
            server.sendmail(Config.EMAIL_SENDER, to_email, msg.as_string())

        return True
    except Exception as e:
        print(f"❌ OTP email failed: {e}")
        return False


# ── Route: Send OTP ──────────────────────────────────────────────
@auth_bp.route('/api/auth/send-otp', methods=['POST'])
def send_otp():
    data  = request.get_json()
    email = data.get('email', '').strip().lower()
    name  = data.get('name', '').strip()

    if not email or '@' not in email:
        return jsonify({'success': False, 'error': 'Valid email required'}), 400
    if not name:
        return jsonify({'success': False, 'error': 'Name is required'}), 400

    otp  = generate_otp()
    store_otp(email, otp)
    sent = send_otp_email(email, otp, name)

    if sent:
        return jsonify({'success': True, 'message': f'OTP sent to {email}'})
    else:
        # For demo: return OTP in response if email fails
        return jsonify({
            'success':  True,
            'message':  'OTP generated (email unavailable in demo)',
            'demo_otp': otp
        })


# ── Route: Verify OTP & Login ────────────────────────────────────
@auth_bp.route('/api/auth/verify-otp', methods=['POST'])
def verify_otp_route():
    data  = request.get_json()
    email = data.get('email', '').strip().lower()
    name  = data.get('name', '').strip()
    otp   = data.get('otp', '').strip()

    if not email or not otp:
        return jsonify({'success': False, 'error': 'Email and OTP required'}), 400

    valid, msg = verify_otp(email, otp)
    if not valid:
        return jsonify({'success': False, 'error': msg}), 401

    user  = get_or_create_user(email, name)
    token = create_session_token(email)

    response = make_response(jsonify({
        'success': True,
        'user':    user,
        'token':   token
    }))
    response.set_cookie(
        'cs_token', token,
        max_age=86400 * 7,  # 7 days
        httponly=False,
        samesite='Lax'
    )
    return response


# ── Route: Get current user ──────────────────────────────────────
@auth_bp.route('/api/auth/me', methods=['GET'])
def me():
    token = request.cookies.get('cs_token') or request.headers.get('X-Auth-Token')
    user  = validate_session_token(token)
    if not user:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    return jsonify({'success': True, 'user': user})


# ── Route: Logout ────────────────────────────────────────────────
@auth_bp.route('/api/auth/logout', methods=['POST'])
def logout():
    token = request.cookies.get('cs_token')
    if token:
        delete_session(token)
    response = make_response(jsonify({'success': True}))
    response.delete_cookie('cs_token')
    return response
