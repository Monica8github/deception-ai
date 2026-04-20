import streamlit as st
from auth.auth_manager import AuthManager

def render_signin():
    auth = AuthManager()

    st.markdown("""
<style>
.auth-card {
    background: #0d0d14;
    border: 1px solid #1a1a2e;
    border-radius: 20px;
    padding: 48px 40px;
    max-width: 440px;
    margin: 60px auto;
}
.stTextInput > div > div > input {
    background: #111118 !important;
    border: 1px solid #1a1a2e !important;
    border-radius: 10px !important;
    color: #f0f0f5 !important;
    padding: 12px 16px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #e63946 !important;
    box-shadow: 0 0 0 3px rgba(230,57,70,0.15) !important;
}
label { color: #8888aa !important;
        font-size: 13px !important;
        font-weight: 500 !important; }
</style>
""", unsafe_allow_html=True)

    # Back button
    if st.button("← Back to Home", key="signin_back"):
        st.session_state['page'] = 'landing'
        st.rerun()

    # Card
    st.markdown("""
<div class="auth-card">
    <div style="text-align:center; margin-bottom:36px">
        <div style="font-size:36px; margin-bottom:12px">🔍</div>
        <div style="font-size:26px; font-weight:800;
                    letter-spacing:-0.5px; margin-bottom:8px">
            Welcome back
        </div>
        <div style="color:#8888aa; font-size:14px">
            Sign in to your DeceptionAI account
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    # Center the form
    _, form_col, _ = st.columns([1, 2, 1])
    with form_col:
        st.markdown("<div style='height:8px'></div>",
                    unsafe_allow_html=True)

        username = st.text_input(
            "Username",
            placeholder="Enter your username",
            key="signin_username"
        )
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            key="signin_password"
        )

        st.markdown("<div style='height:8px'></div>",
                    unsafe_allow_html=True)

        # Error/success state
        if 'signin_error' in st.session_state:
            st.markdown(f"""
<div style="background:rgba(230,57,70,0.1);
            border:1px solid rgba(230,57,70,0.3);
            border-radius:10px; padding:12px 16px;
            color:#e63946; font-size:13px; margin-bottom:12px">
    ❌ {st.session_state['signin_error']}
</div>
""", unsafe_allow_html=True)
            del st.session_state['signin_error']

        # Sign In button
        st.markdown("""
<style>
.signin-btn .stButton > button {
    background:linear-gradient(135deg,#e63946,#c1121f) !important;
    color:white !important; border:none !important;
    font-size:15px !important; padding:14px !important;
    width:100% !important;
    box-shadow:0 4px 16px rgba(230,57,70,0.3) !important;
}
</style>""", unsafe_allow_html=True)

        if st.button("Sign In", key="signin_submit",
                     use_container_width=True):
            if not username or not password:
                st.session_state['signin_error'] = \
                    "Please fill in all fields"
                st.rerun()
            else:
                result = auth.login(username, password)
                if result['success']:
                    st.session_state['authenticated'] = True
                    st.session_state['user'] = result['user']
                    st.session_state['page'] = 'dashboard'
                    st.session_state.pop('analysed', None)
                    st.session_state.pop('results', None)
                    st.rerun()
                else:
                    st.session_state['signin_error'] = \
                        result['message']
                    st.rerun()

        st.markdown("""
<div style="text-align:center; margin-top:24px;
            color:#8888aa; font-size:13px">
    Don't have an account?
</div>
""", unsafe_allow_html=True)

        if st.button("Create Account →",
                     key="goto_signup",
                     use_container_width=True):
            st.session_state['page'] = 'signup'
            st.rerun()

        # Demo account hint
        st.markdown("""
<div style="background:#0a0a14; border:1px solid #1a1a2e;
            border-radius:10px; padding:14px 16px;
            margin-top:20px; font-size:12px; color:#555570;
            text-align:center; line-height:1.8">
    <strong style="color:#8888aa">Demo account</strong><br>
    Username: <code style="color:#2ec4b6">demo</code> &nbsp;
    Password: <code style="color:#2ec4b6">demo123</code>
</div>
""", unsafe_allow_html=True)