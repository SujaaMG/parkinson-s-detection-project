# app/streamlit_app.py
import os
import sys
import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import librosa
import joblib
import tempfile

# -------- FIX IMPORT PATH --------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from scripts.extract_features import extract_features_from_audio

# ---------------------------------

st.title("🩺 Parkinson’s Detection from Voice")
st.markdown("Upload or record your voice and let the AI predict Parkinson’s likelihood.")

model, scaler = joblib.load("models/model.pkl")

option = st.radio("Choose Input Method", ["Upload .wav file", "Record Live"])

file_path = None

if option == "Upload .wav file":
    uploaded = st.file_uploader("Upload a voice sample", type=["wav"])
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded.read())
            file_path = tmp.name
            st.audio(uploaded, format="audio/wav")

elif option == "Record Live":
    duration = st.slider("Recording duration (seconds)", 3, 10, 5)
    if st.button("🎤 Record"):
        fs = 16000
        st.info("Recording...")
        rec = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        file_path = "data/audio/live_record.wav"
        write(file_path, fs, rec)
        st.success("Recording complete!")

if file_path:
    features = extract_features_from_audio(file_path)
    # X_scaled = scaler.transform(features)
    X_scaled = scaler.transform(features.values)

    pred = model.predict(X_scaled)[0]
    result = "🧠 Parkinson’s Detected" if pred == 1 else "✅ Healthy Voice"
    st.subheader(f"Result: {result}")
