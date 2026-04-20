import json, os, hashlib, secrets
from datetime import datetime, timedelta
from pathlib import Path
import streamlit as st

USERS_FILE = Path('auth/users.json')
SESSION_DURATION_HOURS = 24

class AuthManager:
    """Handles user registration, login, and session management."""

    def __init__(self):
        USERS_FILE.parent.mkdir(exist_ok=True)
        if not USERS_FILE.exists():
            USERS_FILE.write_text(json.dumps({"users": {}}))
        self._create_demo_account()

    def _create_demo_account(self):
        """Create demo account if it doesn't exist."""
        data = self._load_users()
        if 'demo' not in data['users']:
            self.register(
                username  = 'demo',
                email     = 'demo@deceptionai.com',
                password  = 'demo123',
                full_name = 'Demo User'
            )

    def _load_users(self) -> dict:
        try:
            return json.loads(USERS_FILE.read_text())
        except:
            return {"users": {}}

    def _save_users(self, data: dict):
        USERS_FILE.write_text(json.dumps(data, indent=2))

    def _hash_password(self, password: str, salt: str = None) -> tuple:
        if salt is None:
            salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        ).hex()
        return hashed, salt

    def register(self, username: str, email: str,
                 password: str, full_name: str = "") -> dict:
        """Register a new user. Returns {success, message}."""
        if len(username) < 3:
            return {"success": False,
                    "message": "Username must be at least 3 characters"}
        if len(password) < 6:
            return {"success": False,
                    "message": "Password must be at least 6 characters"}
        if "@" not in email:
            return {"success": False,
                    "message": "Invalid email address"}

        data = self._load_users()
        if username in data["users"]:
            return {"success": False,
                    "message": "Username already exists"}

        emails = [u["email"] for u in data["users"].values()]
        if email in emails:
            return {"success": False,
                    "message": "Email already registered"}

        hashed, salt = self._hash_password(password)
        data["users"][username] = {
            "username"  : username,
            "email"     : email,
            "full_name" : full_name,
            "password"  : hashed,
            "salt"      : salt,
            "created_at": datetime.now().isoformat(),
            "analyses"  : 0,
        }
        self._save_users(data)
        return {"success": True,
                "message": "Account created successfully"}

    def login(self, username: str, password: str) -> dict:
        """Authenticate user. Returns {success, message, user}."""
        data = self._load_users()
        if username not in data["users"]:
            return {"success": False,
                    "message": "Invalid username or password"}

        user = data["users"][username]
        hashed, _ = self._hash_password(password, user["salt"])
        if hashed != user["password"]:
            return {"success": False,
                    "message": "Invalid username or password"}

        token = secrets.token_hex(32)
        return {
            "success": True,
            "message": "Login successful",
            "user": {
                "username" : user["username"],
                "email"    : user["email"],
                "full_name": user["full_name"],
                "token"    : token,
            }
        }

    def increment_analyses(self, username: str):
        data = self._load_users()
        if username in data["users"]:
            data["users"][username]["analyses"] = \
                data["users"][username].get("analyses", 0) + 1
            self._save_users(data)

    def get_user_stats(self, username: str) -> dict:
        data = self._load_users()
        user = data["users"].get(username, {})
        return {
            "analyses" : user.get("analyses", 0),
            "member_since": user.get("created_at", "")[:10],
        }