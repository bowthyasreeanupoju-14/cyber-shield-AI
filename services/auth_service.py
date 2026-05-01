import sqlite3
import random
import string
import hashlib
import os
from datetime import datetime, timedelta
from config import Config

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database.db')

# ── Initialize database ──────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            email      TEXT    UNIQUE NOT NULL,
            name       TEXT,
            created_at TEXT    DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS otp_store (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            email      TEXT NOT NULL,
            otp        TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            used       INTEGER DEFAULT 0
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            email      TEXT NOT NULL,
            token      TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS check_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            email           TEXT    NOT NULL,
            timestamp       TEXT    DEFAULT CURRENT_TIMESTAMP,
            crime_type      TEXT,
            risk_score      REAL,
            risk_level      TEXT,
            anomaly_values  TEXT,
            shap_summary    TEXT,
            actions_taken   TEXT,
            transaction_id  TEXT
        )
    ''')

    conn.commit()
    conn.close()
    print("✅ Database initialized")


# ── OTP functions ────────────────────────────────────────────────
def generate_otp():
    return ''.join(random.choices(string.digits, k=6))


def store_otp(email, otp):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    # Delete old OTPs for this email
    c.execute("DELETE FROM otp_store WHERE email = ?", (email,))
    expires = (datetime.now() + timedelta(minutes=10)).isoformat()
    c.execute(
        "INSERT INTO otp_store (email, otp, expires_at) VALUES (?, ?, ?)",
        (email, otp, expires)
    )
    conn.commit()
    conn.close()


def verify_otp(email, otp):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute(
        "SELECT id, expires_at, used FROM otp_store WHERE email=? AND otp=? ORDER BY id DESC LIMIT 1",
        (email, otp)
    )
    row = c.fetchone()
    if not row:
        conn.close()
        return False, "Invalid OTP"
    otp_id, expires_at, used = row
    if used:
        conn.close()
        return False, "OTP already used"
    if datetime.now() > datetime.fromisoformat(expires_at):
        conn.close()
        return False, "OTP expired"
    # Mark as used
    c.execute("UPDATE otp_store SET used=1 WHERE id=?", (otp_id,))
    conn.commit()
    conn.close()
    return True, "OK"


# ── User functions ───────────────────────────────────────────────
def get_or_create_user(email, name=None):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT id, name FROM users WHERE email=?", (email,))
    row = c.fetchone()
    if row:
        conn.close()
        return {'id': row[0], 'email': email, 'name': row[1]}
    c.execute(
        "INSERT INTO users (email, name) VALUES (?, ?)",
        (email, name or email.split('@')[0])
    )
    conn.commit()
    uid = c.lastrowid
    conn.close()
    return {'id': uid, 'email': email, 'name': name or email.split('@')[0]}


def get_user_by_email(email):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT id, email, name FROM users WHERE email=?", (email,))
    row = c.fetchone()
    conn.close()
    if row:
        return {'id': row[0], 'email': row[1], 'name': row[2]}
    return None


# ── Session token ────────────────────────────────────────────────
def create_session_token(email):
    token = hashlib.sha256(
        f"{email}{datetime.now().isoformat()}{random.random()}".encode()
    ).hexdigest()
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("DELETE FROM sessions WHERE email=?", (email,))
    c.execute(
        "INSERT INTO sessions (email, token) VALUES (?, ?)",
        (email, token)
    )
    conn.commit()
    conn.close()
    return token


def validate_session_token(token):
    if not token:
        return None
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT email FROM sessions WHERE token=?", (token,))
    row = c.fetchone()
    conn.close()
    if row:
        return get_user_by_email(row[0])
    return None


def delete_session(token):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("DELETE FROM sessions WHERE token=?", (token,))
    conn.commit()
    conn.close()


# ── History functions ────────────────────────────────────────────
def save_check(email, crime_type, risk_score, risk_level,
               anomaly_values, shap_summary, transaction_id):
    import json
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute('''
        INSERT INTO check_history
        (email, crime_type, risk_score, risk_level,
         anomaly_values, shap_summary, transaction_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        email, crime_type, risk_score, risk_level,
        json.dumps(anomaly_values),
        json.dumps(shap_summary),
        transaction_id
    ))
    conn.commit()
    hid = c.lastrowid
    conn.close()
    return hid


def update_actions(check_id, actions):
    import json
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute(
        "UPDATE check_history SET actions_taken=? WHERE id=?",
        (json.dumps(actions), check_id)
    )
    conn.commit()
    conn.close()


def get_history(email, limit=50):
    import json
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute('''
        SELECT id, timestamp, crime_type, risk_score, risk_level,
               anomaly_values, shap_summary, actions_taken, transaction_id
        FROM check_history
        WHERE email=?
        ORDER BY id DESC LIMIT ?
    ''', (email, limit))
    rows = c.fetchall()
    conn.close()

    result = []
    for row in rows:
        result.append({
            'id':             row[0],
            'timestamp':      row[1],
            'crime_type':     row[2],
            'risk_score':     row[3],
            'risk_level':     row[4],
            'anomaly_values': json.loads(row[5]) if row[5] else {},
            'shap_summary':   json.loads(row[6]) if row[6] else {},
            'actions_taken':  json.loads(row[7]) if row[7] else [],
            'transaction_id': row[8]
        })
    return result


def get_history_stats(email):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT COUNT(*) FROM check_history WHERE email=?", (email,))
    total = c.fetchone()[0]
    c.execute(
        "SELECT COUNT(*) FROM check_history WHERE email=? AND risk_level='CRITICAL'",
        (email,)
    )
    critical = c.fetchone()[0]
    c.execute(
        "SELECT COUNT(*) FROM check_history WHERE email=? AND risk_level='HIGH'",
        (email,)
    )
    high = c.fetchone()[0]
    conn.close()
    return {'total': total, 'critical': critical, 'high': high}
