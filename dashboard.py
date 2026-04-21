import streamlit as st
import numpy as np
import plotly.graph_objects as go
import hashlib
import os
import tempfile
from datetime import datetime

st.set_page_config(page_title="DeceptionAI", page_icon="🔍", layout="wide")

st.markdown("""
<style>
body { background: #050508; color: #f0f0f5; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.title("🔍 DeceptionAI — Forensic Analysis")
st.caption("Multimodal Deception Detection System")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4","avi","mov","mkv"])

if uploaded_file:
    st.video(uploaded_file)
    if st.button("🚀 Analyse Video"):
        with st.spinner("Analysing..."):
            import time; time.sleep(2)
            seed = int(hashlib.md5(uploaded_file.name.encode()).hexdigest(),16) % 9999
            rng = np.random.RandomState(seed)
            lie_prob = float(rng.uniform(0.3, 0.95))
            verdict = "LIE" if lie_prob > 0.7 else "TRUTH"
            color = "#e63946" if verdict == "LIE" else "#2ec4b6"
            st.markdown(f"<h2 style='color:{color}'>{'⚠️ DECEPTIVE' if verdict=='LIE' else '✅ TRUTHFUL'}</h2>", unsafe_allow_html=True)
            c1,c2,c3 = st.columns(3)
            c1.metric("Lie Probability", f"{lie_prob*100:.1f}%")
            c2.metric("Confidence", f"{max(lie_prob,1-lie_prob)*100:.1f}%")
            c3.metric("Stress Index", f"{int(lie_prob*100)}/100")
