"""
Simple email/password authentication with Neon PostgreSQL backend.
Replaces Clerk OAuth with direct database authentication.
"""

import hashlib
import os
from database.neon_db import get_connection
from datetime import datetime

def hash_password(password: str) -> str:
    """Hash password using SHA256 with salt."""
    salt = 'deceptionai_salt_2025'
    return hashlib.sha256(
        f'{salt}{password}{salt}'.encode()
    ).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash."""
    return hash_password(password) == hashed

def register_user(email: str, name: str, password: str) -> dict:
    """
    Register a new user.
    Returns: {'success': bool, 'user': {...}} or {'success': False, 'error': '...'}
    """
    try:
        import sys
        sys.path.insert(0, '.')
        from database.neon_db import get_connection, init_database
        init_database()
        
        conn    = get_connection()
        cur     = conn.cursor()
        pw_hash = hash_password(password)
        
        # Check if email already exists
        cur.execute(
            'SELECT id, name FROM users WHERE LOWER(email) = LOWER(%s)',
            (email.lower().strip(),)
        )
        existing = cur.fetchone()
        
        if existing:
            cur.close()
            conn.close()
            return {
                'success': False,
                'error': 'An account with this email already exists. Please login instead.'
            }
        
        # Insert new user
        cur.execute("""
            INSERT INTO users (email, name, password_hash, auth_provider, created_at, last_login)
            VALUES (%s, %s, %s, 'email', NOW(), NOW())
            RETURNING id, email, name
        """, (email.lower().strip(), name.strip(), pw_hash))
        
        row = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()
        
        if row:
            print(f'✅ Registered: {row[1]}')
            return {
                'success': True,
                'user': {
                    'id'   : row[0],
                    'email': row[1],
                    'name' : row[2]
                }
            }
        return {'success': False, 'error': 'Registration failed silently.'}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def login_user(email: str, password: str) -> dict:
    """
    Authenticate user with email and password.
    Returns: {'success': bool, 'user': {...}} or {'success': False, 'error': '...'}
    """
    try:
        import sys
        sys.path.insert(0, '.')
        from database.neon_db import get_connection
        
        conn    = get_connection()
        cur     = conn.cursor()
        pw_hash = hash_password(password)
        
        cur.execute("""
            SELECT id, email, name
            FROM users
            WHERE LOWER(email) = LOWER(%s)
              AND password_hash = %s
        """, (email.lower().strip(), pw_hash))
        
        row = cur.fetchone()
        
        if not row:
            # Check if email exists but wrong password
            cur.execute(
                'SELECT id FROM users WHERE LOWER(email) = LOWER(%s)',
                (email.lower().strip(),)
            )
            exists = cur.fetchone()
            cur.close()
            conn.close()
            if exists:
                return {'success': False, 'error': 'Incorrect password. Please try again.'}
            return {'success': False, 'error': 'No account found. Please register first.'}
        
        # Update last_login
        cur.execute('UPDATE users SET last_login = NOW() WHERE id = %s', (row[0],))
        conn.commit()
        cur.close()
        conn.close()
        
        print(f'✅ Login: {row[1]}')
        return {
            'success': True,
            'user': {'id': row[0], 'email': row[1], 'name': row[2]}
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def save_analysis(user_id: int, filename: str, results: dict) -> bool:
    """
    Save analysis results to database.
    Returns: True if successful, False otherwise.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO analysis_history 
            (user_id, filename, verdict, lie_probability, 
             confidence, stress_index, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            user_id,
            filename,
            results.get('verdict', 'UNKNOWN'),
            float(results.get('lie_probability', 0.5)),
            float(results.get('confidence', 0.5)),
            int(results.get('stress_index', 50)),
            datetime.now()
        ))
        
        conn.commit()
        cur.close()
        conn.close()
        return True
    
    except Exception as e:
        print(f"Save analysis error: {e}")
        return False

def get_user_history(user_id: int) -> list:
    """
    Get analysis history for a user (last 20 analyses).
    Returns: List of analysis records.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT filename, verdict, lie_probability,
                   confidence, stress_index, created_at
            FROM analysis_history
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 20
        """, (user_id,))
        
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        return [
            {
                'filename': r[0],
                'verdict': r[1],
                'lie_probability': r[2],
                'confidence': r[3],
                'stress_index': r[4],
                'created_at': str(r[5])
            }
            for r in rows
        ]
    
    except Exception as e:
        print(f"History error: {e}")
        return []

def get_user_stats(user_id: int) -> dict:
    """
    Get statistics for a user.
    Returns: {'total': int, 'deceptive': int, 'truthful': int}
    """
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN verdict = 'DECEPTIVE' THEN 1 ELSE 0 END) as deceptive,
                   SUM(CASE WHEN verdict = 'TRUTHFUL' THEN 1 ELSE 0 END) as truthful
            FROM analysis_history
            WHERE user_id = %s
        """, (user_id,))
        
        row = cur.fetchone()
        cur.close()
        conn.close()
        
        if row:
            return {
                'total': row[0] or 0,
                'deceptive': row[1] or 0,
                'truthful': row[2] or 0
            }
        return {'total': 0, 'deceptive': 0, 'truthful': 0}
    
    except Exception as e:
        print(f"Stats error: {e}")
        return {'total': 0, 'deceptive': 0, 'truthful': 0}
