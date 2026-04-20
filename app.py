"""
DeceptionAI — Main Entry Point
Routes users to signin or dashboard based on authentication state.
"""

import streamlit as st

# Session state defaults
for key, val in {
    'authenticated': False,
    'user': None,
    'analysed': False,
    'results': {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Route based on auth state
if not st.session_state.get('authenticated'):
    st.switch_page('pages/signin.py')
else:
    st.switch_page('pages/dashboard.py')
