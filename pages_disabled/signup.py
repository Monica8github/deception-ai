import streamlit as st
from auth.auth_manager import AuthManager

def render_signup():
    auth = AuthManager()

    st.markdown("""
<style>
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

    if st.button("← Back to Home", key="signup_back"):
        st.session_state['page'] = 'landing'
        st.rerun()

    st.markdown("""
<div style="text-align:center; padding:40px 0 32px">
    <div style="font-size:36px; margin-bottom:12px">🔍</div>
    <div style="font-size:28px; font-weight:800;
                letter-spacing:-0.5px; margin-bottom:8px">
        Create your account
    </div>
    <div style="color:#8888aa; font-size:14px">
        Join DeceptionAI — free forensic analysis platform
    </div>
</div>
""", unsafe_allow_html=True)

    _, fc, _ = st.columns([1, 2, 1])
    with fc:
        if 'signup_error' in st.session_state:
            st.markdown(f"""
<div style="background:rgba(230,57,70,0.1);
            border:1px solid rgba(230,57,70,0.3);
            border-radius:10px; padding:12px 16px;
            color:#e63946; font-size:13px; margin-bottom:12px">
    ❌ {st.session_state['signup_error']}
</div>
""", unsafe_allow_html=True)
            del st.session_state['signup_error']

        if st.session_state.get('signup_success'):
            st.markdown("""
<div style="background:rgba(46,196,182,0.1);
            border:1px solid rgba(46,196,182,0.3);
            border-radius:10px; padding:16px;
            color:#2ec4b6; font-size:13px; margin-bottom:12px;
            text-align:center">
    ✅ Account created! Redirecting to sign in...
</div>
""", unsafe_allow_html=True)
            del st.session_state['signup_success']
            import time; time.sleep(1.5)
            st.session_state['page'] = 'signin'
            st.rerun()

        full_name = st.text_input(
            "Full Name",
            placeholder="Your full name",
            key="su_name"
        )
        username = st.text_input(
            "Username",
            placeholder="Choose a username (min 3 chars)",
            key="su_user"
        )
        email = st.text_input(
            "Email Address",
            placeholder="your@email.com",
            key="su_email"
        )

        pc1, pc2 = st.columns(2)
        with pc1:
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Min 6 characters",
                key="su_pass"
            )
        with pc2:
            confirm = st.text_input(
                "Confirm Password",
                type="password",
                placeholder="Repeat password",
                key="su_confirm"
            )

        st.markdown("<div style='height:8px'></div>",
                    unsafe_allow_html=True)

        # Password strength indicator
        if password:
            strength = (len(password) >= 6) + \
                       any(c.isupper() for c in password) + \
                       any(c.isdigit() for c in password)
            colors = ['#e63946','#f4a261','#2ec4b6']
            labels = ['Weak','Fair','Strong']
            bar_w  = [33, 66, 100]
            st.markdown(f"""
<div style="margin-bottom:12px">
    <div style="background:#1a1a2e;border-radius:99px;
                height:5px;overflow:hidden;margin-bottom:4px">
        <div style="width:{bar_w[strength-1]}%;height:100%;
                    background:{colors[strength-1]};
                    border-radius:99px;
                    transition:width 0.3s"></div>
    </div>
    <div style="font-size:11px;color:{colors[strength-1]}">
        Password strength: {labels[strength-1]}
    </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<style>
div[data-testid="column"] .stButton > button {
    background:linear-gradient(135deg,#e63946,#7c3aed) !important;
    color:white !important; border:none !important;
    font-size:15px !important; padding:14px !important;
    box-shadow:0 4px 16px rgba(230,57,70,0.3) !important;
}
</style>""", unsafe_allow_html=True)

        if st.button("Create Account",
                     key="signup_submit",
                     use_container_width=True):
            if not all([full_name, username, email,
                        password, confirm]):
                st.session_state['signup_error'] = \
                    "Please fill in all fields"
                st.rerun()
            elif password != confirm:
                st.session_state['signup_error'] = \
                    "Passwords do not match"
                st.rerun()
            else:
                result = auth.register(
                    username, email, password, full_name
                )
                if result['success']:
                    st.session_state['signup_success'] = True
                    st.rerun()
                else:
                    st.session_state['signup_error'] = \
                        result['message']
                    st.rerun()

        st.markdown("""
<div style="text-align:center;margin-top:24px;
            color:#8888aa;font-size:13px">
    Already have an account?
</div>
""", unsafe_allow_html=True)
        if st.button("Sign In →", key="goto_signin",
                     use_container_width=True):
            st.session_state['page'] = 'signin'
            st.rerun()