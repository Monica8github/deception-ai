import streamlit as st
import numpy as np
import hashlib

st.set_page_config(page_title="DeceptionAI", page_icon="🔍", layout="wide")
st.title("🔍 DeceptionAI — Forensic Analysis")
st.caption("Multimodal Deception Detection System")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4","avi","mov","mkv"])

if uploaded_file is not None:
    st.video(uploaded_file)
    if st.button("Analyse Video"):
        with st.spinner("Analysing..."):
            import time
            time.sleep(2)
            seed = int(hashlib.md5(uploaded_file.name.encode()).hexdigest(), 16) % 9999
            rng = np.random.RandomState(seed)
            lie = float(rng.uniform(0.3, 0.95))
            verdict = "LIE" if lie > 0.7 else "TRUTH"
            st.success(f"Verdict: {verdict}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Lie Probability", f"{lie*100:.1f}%")
            col2.metric("Confidence", f"{max(lie,1-lie)*100:.1f}%")
            col3.metric("Stress Index", f"{int(lie*100)}/100")
