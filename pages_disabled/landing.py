import streamlit as st

def render_landing():
    st.markdown("""
<style>
.hero-title {
    font-size: 62px;
    font-weight: 900;
    letter-spacing: -2px;
    line-height: 1.05;
    margin-bottom: 24px;
}
.hero-title .accent {
    background: linear-gradient(135deg, #e63946, #a855f7 60%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.nav { display:flex; align-items:center;
       justify-content:space-between; padding:20px 48px;
       border-bottom:1px solid #1a1a2e;
       background:rgba(5,5,8,0.97); position:sticky;
       top:0; z-index:100; }
.logo { font-size:20px; font-weight:800; letter-spacing:0.5px;
        background:linear-gradient(135deg,#e63946,#a855f7);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;
        background-clip:text; }
.feat-card { background:#0d0d14; border:1px solid #1a1a2e;
             border-radius:16px; padding:28px 24px;
             transition:border-color 0.2s; }
.feat-card:hover { border-color:#252540; }
.stat-num { font-family:'JetBrains Mono',monospace; font-size:38px;
            font-weight:700;
            background:linear-gradient(135deg,#e63946,#a855f7);
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;
            background-clip:text; display:block; margin-bottom:6px; }
@keyframes fadeUp {
    from { opacity:0; transform:translateY(30px); }
    to   { opacity:1; transform:translateY(0); }
}
.animate { animation: fadeUp 0.7s ease forwards; }
</style>
""", unsafe_allow_html=True)

    # ── NAV ────────────────────────────────────────────────────
    st.markdown("""
<nav class="nav">
    <div class="logo">🔍 DeceptionAI</div>
    <div style="display:flex; gap:32px; align-items:center">
        <a style="color:#8888aa;text-decoration:none;
                  font-size:13px;font-weight:500">Features</a>
        <a style="color:#8888aa;text-decoration:none;
                  font-size:13px;font-weight:500">How It Works</a>
        <a style="color:#8888aa;text-decoration:none;
                  font-size:13px;font-weight:500">Research</a>
    </div>
    <div style="display:flex;gap:12px"></div>
</nav>
""", unsafe_allow_html=True)

    # Nav buttons
    ncol1, ncol2, ncol3 = st.columns([6, 1, 1])
    with ncol2:
        if st.button("Sign In", key="nav_signin",
                     use_container_width=True):
            st.session_state['page'] = 'signin'
            st.rerun()
    with ncol3:
        st.markdown("""
<style>
div[data-testid="column"]:last-child .stButton > button {
    background: linear-gradient(135deg,#e63946,#c1121f) !important;
    color: white !important; border: none !important;
}
</style>""", unsafe_allow_html=True)
        if st.button("Get Started", key="nav_signup",
                     use_container_width=True):
            st.session_state['page'] = 'signup'
            st.rerun()

    # ── HERO ──────────────────────────────────────────────────
    st.markdown("<div style='height:60px'></div>",
                unsafe_allow_html=True)
    st.markdown("""
<div style="text-align:center; padding:0 40px" class="animate">
    <div style="display:inline-flex; align-items:center; gap:8px;
                background:rgba(230,57,70,0.1);
                border:1px solid rgba(230,57,70,0.25);
                color:#e63946; padding:6px 18px; border-radius:99px;
                font-size:12px; font-weight:700; letter-spacing:1px;
                margin-bottom:32px">
        <span style="width:7px;height:7px;background:#e63946;
                     border-radius:50%;animation:blink 1.5s infinite">
        </span>
        AI-POWERED FORENSIC ANALYSIS PLATFORM
    </div>
    <div class="hero-title">
        Detect deception with<br>
        <span class="accent">multimodal AI intelligence</span>
    </div>
    <p style="color:#8888aa; font-size:18px; line-height:1.8;
              max-width:580px; margin:0 auto 44px; font-weight:400">
        Combines facial micro-expressions, vocal stress patterns,
        and linguistic cues to identify deceptive behaviour
        with clinical precision.
    </p>
</div>
""", unsafe_allow_html=True)

    # CTA buttons
    hc1, hc2, hc3 = st.columns([2, 1, 1])
    with hc2:
        st.markdown("""
<style>
div[data-testid="column"]:nth-child(2) .stButton > button {
    background:linear-gradient(135deg,#e63946,#7c3aed) !important;
    color:white !important; border:none !important;
    font-size:16px !important; padding:16px 32px !important;
    box-shadow:0 4px 24px rgba(230,57,70,0.3) !important;
}
</style>""", unsafe_allow_html=True)
        if st.button("🚀 Start Free Analysis",
                     key="hero_cta", use_container_width=True):
            st.session_state['page'] = 'signup'
            st.rerun()
    with hc3:
        if st.button("Sign In →", key="hero_signin",
                     use_container_width=True):
            st.session_state['page'] = 'signin'
            st.rerun()

    # ── STATS STRIP ───────────────────────────────────────────
    st.markdown("<div style='height:72px'></div>",
                unsafe_allow_html=True)
    st.markdown("""
<div style="background:#0a0a10; border-top:1px solid #1a1a2e;
            border-bottom:1px solid #1a1a2e; padding:44px 0;
            display:flex; justify-content:center; gap:0">
    <div style="text-align:center; padding:0 64px;
                border-right:1px solid #1a1a2e">
        <span class="stat-num">121</span>
        <span style="color:#555570;font-size:13px;font-weight:500">
            Training Videos
        </span>
    </div>
    <div style="text-align:center; padding:0 64px;
                border-right:1px solid #1a1a2e">
        <span class="stat-num">3</span>
        <span style="color:#555570;font-size:13px;font-weight:500">
            AI Modalities
        </span>
    </div>
    <div style="text-align:center; padding:0 64px;
                border-right:1px solid #1a1a2e">
        <span class="stat-num">F1 0.69</span>
        <span style="color:#555570;font-size:13px;font-weight:500">
            Model Accuracy
        </span>
    </div>
    <div style="text-align:center; padding:0 64px">
        <span class="stat-num">&lt; 3s</span>
        <span style="color:#555570;font-size:13px;font-weight:500">
            Analysis Time
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

    # ── FEATURES GRID ────────────────────────────────────────
    st.markdown("<div style='height:72px'></div>",
                unsafe_allow_html=True)
    st.markdown("""
<div style="text-align:center; margin-bottom:48px">
    <div style="color:#e63946;font-size:11px;font-weight:700;
                letter-spacing:2px;margin-bottom:12px">
        CORE CAPABILITIES
    </div>
    <div style="font-size:36px;font-weight:800;
                letter-spacing:-0.5px">
        Three modalities. One verdict.
    </div>
</div>
""", unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    features = [
        ("🎥","#e63946","Visual Analysis",
         "12 facial features per frame — eye aspect ratio, brow raise, "
         "lip compression, head pose, micro-expression intensity "
         "via MediaPipe FaceMesh."),
        ("🔊","#2ec4b6","Voice Stress",
         "71 acoustic features — MFCCs, pitch f0, RMS energy, "
         "jitter, shimmer, and vocal stress score Sa(t) = "
         "|f0'(t)| + |E'(t)|."),
        ("📝","#a855f7","NLP Intelligence",
         "OpenAI Whisper transcription + DistilBERT encoding "
         "extracts attention-weighted suspicious linguistic "
         "markers from speech."),
    ]
    for col, (icon, color, title, desc) in zip(
            [fc1, fc2, fc3], features):
        with col:
            st.markdown(f"""
<div class="feat-card">
    <div style="width:44px;height:44px;border-radius:10px;
                background:rgba({
                    '230,57,70' if color=='#e63946' else
                    '46,196,182' if color=='#2ec4b6' else
                    '168,85,247'},0.12);
                display:flex;align-items:center;
                justify-content:center;font-size:20px;
                margin-bottom:16px">{icon}</div>
    <div style="font-size:16px;font-weight:700;
                margin-bottom:8px;color:#f0f0f5">{title}</div>
    <div style="font-size:13px;color:#8888aa;
                line-height:1.7">{desc}</div>
</div>
""", unsafe_allow_html=True)

    # ── HOW IT WORKS ─────────────────────────────────────────
    st.markdown("<div style='height:80px'></div>",
                unsafe_allow_html=True)
    st.markdown("""
<div style="text-align:center; margin-bottom:48px">
    <div style="color:#e63946;font-size:11px;font-weight:700;
                letter-spacing:2px;margin-bottom:12px">
        WORKFLOW
    </div>
    <div style="font-size:36px;font-weight:800;
                letter-spacing:-0.5px">
        How it works
    </div>
</div>
""", unsafe_allow_html=True)

    steps = [
        ("01","Upload Video",
         "Upload any MP4, AVI, or MOV file. "
         "Supports up to 500MB.","#e63946"),
        ("02","AI Extracts Features",
         "Three parallel pipelines analyse "
         "face, voice, and speech simultaneously.","#f4a261"),
        ("03","Attention Fusion",
         "Channel-wise attention learns which modality "
         "matters most for this specific subject.","#a855f7"),
        ("04","Forensic Report",
         "Instant verdict with confidence score, "
         "temporal map, and explainability breakdown.","#2ec4b6"),
    ]
    s1, s2, s3, s4 = st.columns(4)
    for col, (num, title, desc, color) in zip(
            [s1, s2, s3, s4], steps):
        with col:
            st.markdown(f"""
<div style="text-align:center; padding:24px 16px">
    <div style="font-family:'JetBrains Mono',monospace;
                font-size:42px; font-weight:700; color:{color};
                opacity:0.3; margin-bottom:12px">{num}</div>
    <div style="font-size:15px; font-weight:700;
                margin-bottom:8px; color:#f0f0f5">{title}</div>
    <div style="font-size:13px; color:#8888aa;
                line-height:1.7">{desc}</div>
</div>
""", unsafe_allow_html=True)

    # ── CTA SECTION ──────────────────────────────────────────
    st.markdown("<div style='height:80px'></div>",
                unsafe_allow_html=True)
    st.markdown("""
<div style="background:linear-gradient(135deg,#0d0509,#0d0514);
            border:1px solid #1a1a2e; border-radius:24px;
            padding:72px 40px; text-align:center; margin:0 0 80px">
    <div style="font-size:42px;font-weight:900;
                letter-spacing:-0.8px;margin-bottom:16px">
        Ready to analyse?
    </div>
    <p style="color:#8888aa;font-size:16px;margin-bottom:40px">
        Create a free account and run your first forensic
        analysis in under 60 seconds.
    </p>
</div>
""", unsafe_allow_html=True)

    cc1, cc2, cc3 = st.columns([2, 1, 2])
    with cc2:
        st.markdown("""
<style>
.cta-btn .stButton > button {
    background:linear-gradient(135deg,#e63946,#7c3aed) !important;
    color:white !important; border:none !important;
    font-size:16px !important; padding:16px !important;
    box-shadow:0 4px 32px rgba(230,57,70,0.3) !important;
}
</style>""", unsafe_allow_html=True)
        if st.button("Create Free Account",
                     key="cta_final", use_container_width=True):
            st.session_state['page'] = 'signup'
            st.rerun()

    # ── FOOTER ────────────────────────────────────────────────
    st.markdown("""
<div style="border-top:1px solid #1a1a2e; padding:32px 40px;
            display:flex; align-items:center;
            justify-content:space-between; flex-wrap:wrap; gap:16px">
    <div>
        <div style="font-size:15px;font-weight:700;
                    background:linear-gradient(135deg,#e63946,#a855f7);
                    -webkit-background-clip:text;
                    -webkit-text-fill-color:transparent;
                    background-clip:text">
            DeceptionAI
        </div>
        <div style="color:#555570;font-size:11px;margin-top:4px">
            St. Joseph's College of Engineering ——
            AI & Data Science
        </div>
    </div>
    <div style="color:#555570;font-size:11px;
                font-family:'JetBrains Mono',monospace">
        v2.1 · PyTorch · DistilBERT · Streamlit
    </div>
</div>
""", unsafe_allow_html=True)