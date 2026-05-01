import smtplib
import json
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from config import Config

# ── Twilio (optional) ────────────────────────────────────────────
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════
# ALERT LOG (JSON file) 
# ══════════════════════════════════════════════════════════════════

def log_alert(transaction_id, risk_score, risk_level, email_sent, sms_sent):
    try:
        with open(Config.ALERT_LOG_FILE, 'r') as f:
            history = json.load(f)
    except Exception:
        history = []

    history.insert(0, {
        'timestamp':      datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'transaction_id': transaction_id,
        'risk_score':     risk_score,
        'risk_level':     risk_level,
        'email_sent':     email_sent,
        'sms_sent':       sms_sent,
    })

    with open(Config.ALERT_LOG_FILE, 'w') as f:
        json.dump(history[:100], f, indent=2)


def get_alert_history():
    try:
        with open(Config.ALERT_LOG_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════
# EMAIL ALERT
# ══════════════════════════════════════════════════════════════════

def send_email_alert(transaction_id, risk_score, risk_level, contributions):
    try:
        level_color = '#ff4444' if risk_level == 'CRITICAL' else '#ffa500'

        rows = ''.join([
            f"<tr><td style='padding:8px;border:1px solid #333'>{k.title()}</td>"
            f"<td style='padding:8px;border:1px solid #333;color:#00d4ff'>{v:.2f}%</td></tr>"
            for k, v in contributions.items()
        ])

        html = f"""
        <html><body style='background:#0a0e1a;color:white;font-family:Arial;padding:20px'>
        <div style='max-width:600px;margin:auto;background:#1a1d2e;border-radius:10px;
                    padding:30px;border:2px solid {level_color}'>
            <h2 style='color:{level_color};text-align:center'>
                {'🚨 CRITICAL ALERT' if risk_level == 'CRITICAL' else '⚠️ HIGH RISK ALERT'}
            </h2>
            <table style='width:100%;border-collapse:collapse;margin:15px 0'>
                <tr><td style='padding:8px;color:#aaa'>Transaction ID</td>
                    <td style='padding:8px;font-weight:bold'>{transaction_id}</td></tr>
                <tr><td style='padding:8px;color:#aaa'>Risk Score</td>
                    <td style='padding:8px;font-size:24px;font-weight:bold;color:{level_color}'>
                    {risk_score}/100</td></tr>
                <tr><td style='padding:8px;color:#aaa'>Risk Level</td>
                    <td style='padding:8px;font-weight:bold;color:{level_color}'>{risk_level}</td></tr>
                <tr><td style='padding:8px;color:#aaa'>Time</td>
                    <td style='padding:8px'>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
            </table>
            <h3 style='color:#00d4ff'>Anomaly Breakdown</h3>
            <table style='width:100%;border-collapse:collapse'>
                <tr style='background:#2a2d3e'>
                    <th style='padding:8px;border:1px solid #333;text-align:left'>Feature</th>
                    <th style='padding:8px;border:1px solid #333;text-align:left'>Contribution</th>
                </tr>{rows}
            </table>
            <div style='background:#2a2d3e;padding:15px;border-radius:8px;margin-top:15px'>
                <strong style='color:{level_color}'>Recommended Action:</strong><br>
                {'🔴 Block transaction immediately and contact cardholder'
                 if risk_level == 'CRITICAL'
                 else '🟡 Hold transaction for manual review'}
            </div>
        </div></body></html>"""

        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"[{risk_level}] Fraud Alert — TXN {transaction_id} | Score: {risk_score}/100"
        msg['From']    = Config.EMAIL_SENDER
        msg['To']      = Config.EMAIL_RECEIVER
        msg.attach(MIMEText(html, 'html'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(Config.EMAIL_SENDER, Config.EMAIL_APP_PASSWORD)
            server.sendmail(Config.EMAIL_SENDER, Config.EMAIL_RECEIVER, msg.as_string())

        print(f"✅ Email sent for TXN {transaction_id}")
        return True
    except Exception as e:
        print(f"❌ Email failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════════
# SMS ALERT
# ══════════════════════════════════════════════════════════════════

def send_sms_alert(transaction_id, risk_score, risk_level):
    if not TWILIO_AVAILABLE:
        print("⚠️ Twilio not installed — skipping SMS")
        return False
    try:
        client  = TwilioClient(Config.TWILIO_SID, Config.TWILIO_TOKEN)
        body    = (
            f"🚨 FRAUD ALERT\n"
            f"TXN: {transaction_id}\n"
            f"Score: {risk_score}/100\n"
            f"Level: {risk_level}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}\n"
            f"ACTION: Block transaction immediately"
        )
        msg = client.messages.create(body=body, from_=Config.TWILIO_FROM, to=Config.TWILIO_TO)
        print(f"✅ SMS sent — SID: {msg.sid}")
        return True
    except Exception as e:
        print(f"❌ SMS failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════════
# MASTER DISPATCHER
# ══════════════════════════════════════════════════════════════════

def dispatch_alert(transaction_id, risk_score, risk_level, contributions):
    """
    CRITICAL → Email + SMS
    HIGH     → Email only
    MEDIUM   → Log only
    LOW      → Nothing
    """
    email_sent = False
    sms_sent   = False

    if risk_level == 'CRITICAL':
        email_sent = send_email_alert(transaction_id, risk_score, risk_level, contributions)
        sms_sent   = send_sms_alert(transaction_id, risk_score, risk_level)
    elif risk_level == 'HIGH':
        email_sent = send_email_alert(transaction_id, risk_score, risk_level, contributions)
    elif risk_level == 'MEDIUM':
        print(f"📋 MEDIUM logged: TXN {transaction_id} | Score {risk_score}")

    log_alert(transaction_id, risk_score, risk_level, email_sent, sms_sent)

    return {'email_sent': email_sent, 'sms_sent': sms_sent}
