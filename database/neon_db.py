"""
Neon PostgreSQL database module for simple email/password authentication.
Handles users, analysis history, and statistics storage.
"""

import os
import psycopg2
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    """
    Get a Neon PostgreSQL connection with SSL support.
    Supports both DATABASE_URL and NEON_DATABASE_URL environment variables.
    """
    database_url = os.getenv('DATABASE_URL') or os.getenv('NEON_DATABASE_URL')
    
    if not database_url:
        raise ValueError("DATABASE_URL or NEON_DATABASE_URL not set in .env file")
    
    try:
        conn = psycopg2.connect(
            database_url,
            sslmode='require',
            connect_timeout=10
        )
        return conn
    except psycopg2.Error as e:
        raise Exception(f"Failed to connect to Neon database: {e}")

def init_database():
    try:
        conn = get_connection()
        cur  = conn.cursor()

        # Check if users table has password_hash column
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'users'
        """)
        existing_cols = [r[0] for r in cur.fetchall()]

        # Create users table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            SERIAL PRIMARY KEY,
                email         VARCHAR(255) UNIQUE NOT NULL,
                name          VARCHAR(255) NOT NULL DEFAULT '',
                password_hash VARCHAR(128) NOT NULL DEFAULT '',
                avatar_url    VARCHAR(500) DEFAULT '',
                auth_provider VARCHAR(50)  DEFAULT 'email',
                clerk_id      VARCHAR(255) DEFAULT '',
                created_at    TIMESTAMP   DEFAULT NOW(),
                last_login    TIMESTAMP   DEFAULT NOW()
            )
        """)

        # Add missing columns if table exists but is missing columns
        required_cols = {
            'password_hash': 'VARCHAR(128) NOT NULL DEFAULT \'\'',
            'avatar_url'   : 'VARCHAR(500) DEFAULT \'\'',
            'auth_provider': 'VARCHAR(50)  DEFAULT \'email\'',
            'clerk_id'     : 'VARCHAR(255) DEFAULT \'\'',
            'last_login'   : 'TIMESTAMP   DEFAULT NOW()',
            'name'         : 'VARCHAR(255) NOT NULL DEFAULT \'\'',
        }
        for col, col_type in required_cols.items():
            if col not in existing_cols:
                try:
                    cur.execute(
                        f"ALTER TABLE users ADD COLUMN {col} {col_type}"
                    )
                    print(f"  Added column: {col}")
                except Exception:
                    pass

        # Create analysis_history table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id              SERIAL PRIMARY KEY,
                user_id         INTEGER REFERENCES users(id)
                                ON DELETE CASCADE,
                filename        VARCHAR(500) DEFAULT '',
                verdict         VARCHAR(20)  DEFAULT 'UNKNOWN',
                lie_probability FLOAT        DEFAULT 0.5,
                confidence      FLOAT        DEFAULT 0.5,
                stress_index    INTEGER      DEFAULT 50,
                video_duration  FLOAT        DEFAULT 0,
                resolution      VARCHAR(50)  DEFAULT '',
                analysis_mode   VARCHAR(50)  DEFAULT 'Instant',
                created_at      TIMESTAMP    DEFAULT NOW()
            )
        """)

        conn.commit()
        cur.close()
        conn.close()
        print("✅ Database ready")
        return True

    except Exception as e:
        print(f"❌ DB init error: {e}")
        return False

def upsert_user(email: str, name: str, password_hash: str) -> dict:
    """
    Placeholder for backward compatibility with old dashboard code.
    Not used in new simple_auth flow.
    """
    return {
        "db_id": 0,
        "total_analyses": 0,
        "member_since": "N/A"
    }

def save_analysis_legacy(clerk_user_id: str, results: dict) -> int:
    """
    Placeholder for backward compatibility with old dashboard code.
    Not used in new simple_auth flow.
    """
    return 0

def get_user_history_legacy(clerk_user_id: str, limit: int = 10) -> list:
    """
    Placeholder for backward compatibility with old dashboard code.
    Not used in new simple_auth flow.
    """
    return []

def get_user_stats_legacy(clerk_user_id: str) -> dict:
    """
    Placeholder for backward compatibility with old dashboard code.
    Not used in new simple_auth flow.
    """
    return {
        "total": 0,
        "deceptive": 0,
        "truthful": 0,
        "avg_conf": 0.0,
        "avg_lie": 0.0,
    }