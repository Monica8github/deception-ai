"""
Clerk Google OAuth authentication module.
Handles Google sign-in via Clerk API.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

CLERK_SECRET_KEY = os.getenv('CLERK_SECRET_KEY', '')
CLERK_FRONTEND_API = os.getenv('CLERK_FRONTEND_API', '')
CLERK_PUBLISHABLE_KEY = os.getenv('CLERK_PUBLISHABLE_KEY', '')

def get_google_oauth_url(redirect_url: str = 'http://localhost:8501') -> str:
    """Generate Clerk Google OAuth URL."""
    frontend_api = CLERK_FRONTEND_API.rstrip('/')
    return (
        f"{frontend_api}/v1/oauth_callback?"
        f"strategy=oauth_google"
        f"&redirect_url={redirect_url}"
        f"&action_complete_redirect_url={redirect_url}"
    )

def verify_session_token(session_token: str) -> dict:
    """Verify a Clerk session token and return user info."""
    try:
        headers = {
            'Authorization': f'Bearer {CLERK_SECRET_KEY}',
            'Content-Type': 'application/json'
        }
        resp = requests.get(
            'https://api.clerk.com/v1/sessions/' + session_token,
            headers=headers, timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            user_id = data.get('user_id')
            if user_id:
                user_resp = requests.get(
                    f'https://api.clerk.com/v1/users/{user_id}',
                    headers=headers, timeout=10
                )
                if user_resp.status_code == 200:
                    u = user_resp.json()
                    emails = u.get('email_addresses', [])
                    email = emails[0]['email_address'] if emails else ''
                    name = f"{u.get('first_name','')} {u.get('last_name','')}".strip()
                    return {
                        'success': True,
                        'user': {
                            'id': user_id,
                            'email': email,
                            'name': name or email,
                            'image': u.get('image_url', '')
                        }
                    }
        return {'success': False, 'error': 'Invalid session'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def is_clerk_configured() -> bool:
    """Check if Clerk API keys are properly configured."""
    return bool(CLERK_SECRET_KEY and CLERK_PUBLISHABLE_KEY
                and 'xxxx' not in CLERK_SECRET_KEY)