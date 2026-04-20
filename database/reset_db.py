"""
Reset and recreate Neon PostgreSQL database tables with correct schema.
Run this once to fix schema issues.
"""

import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

def reset_database():
    DATABASE_URL = os.getenv('DATABASE_URL') or os.getenv('NEON_DATABASE_URL')
    if not DATABASE_URL:
        print("❌ No DATABASE_URL in .env")
        return False

    try:
        conn = psycopg2.connect(DATABASE_URL, sslmode='require')
        cur  = conn.cursor()

        print("Dropping old tables...")
        cur.execute("DROP TABLE IF EXISTS analysis_history CASCADE;")
        cur.execute("DROP TABLE IF EXISTS users CASCADE;")

        print("Creating users table...")
        cur.execute("""
            CREATE TABLE users (
                id            SERIAL PRIMARY KEY,
                email         VARCHAR(255) UNIQUE NOT NULL,
                name          VARCHAR(255) NOT NULL DEFAULT '',
                password_hash VARCHAR(128) NOT NULL DEFAULT '',
                avatar_url    VARCHAR(500) DEFAULT '',
                auth_provider VARCHAR(50)  DEFAULT 'email',
                clerk_id      VARCHAR(255) DEFAULT '',
                created_at    TIMESTAMP   DEFAULT NOW(),
                last_login    TIMESTAMP   DEFAULT NOW()
            );
        """)

        print("Creating analysis_history table...")
        cur.execute("""
            CREATE TABLE analysis_history (
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
            );
        """)

        conn.commit()
        cur.close()
        conn.close()
        print("✅ Database reset complete!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == '__main__':
    reset_database()