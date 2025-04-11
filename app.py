import streamlit as st
import numpy as np
import librosa
import joblib
import tensorflow as tf
import os
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------ CONFIGURAÇÃO INICIAL ------------------------ #

MODEL_PATH = 'notebooks/models/emotion_recognition_model.h5'
SCALER_PATH = 'notebooks/models/scaler.save'

EMOTIONS = ["angry", "calm", "disgust", "fear",
            "happy", "neutral", "sad", "surprise"]

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ------------------------ FUNÇÃO DE EXTRAÇÃO ------------------------ #
def extract_features(audio_path):
    data, sr = librosa.load(audio_path, sr=16000, mono=True)
    features = []

    features.extend(np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0))
    features.extend(np.mean(librosa.feature.chroma_stft(y=data, sr=sr).T, axis=0))
    features.extend(np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0))
    features.extend(np.mean(librosa.feature.rms(y=data).T, axis=0))
    features.extend(np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0))

    target_length = 155
    if len(features) < target_length:
        features.extend([0] * (target_length - len(features)))
    else:
        features = features[:target_length]

    return np.array(features).reshape(1, -1)

# ------------------------ STREAMLIT APP ------------------------ #

st.title("🎧 TRILHA: Detecção de Emoções em Áudio")

st.markdown("""
Bem-vindo ao sistema de detecção de emoções baseado em áudio!

Faça upload de um arquivo `.wav`, `.mp3` ou `.ogg` e descubra qual emoção está sendo expressada na fala.

Um projeto feito por Nathan David um aluno do trilha. **DIVIRTA-SE!**
""")

uploaded_file = st.file_uploader(
    "🎙️ Envie seu áudio aqui:", 
    type=["wav", "mp3", "ogg"]
)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_audio_path = tmp_file.name

    st.audio(temp_audio_path, format='audio/wav')

    features = extract_features(temp_audio_path)

    features_scaled = scaler.transform(features)
    features_scaled = features_scaled.reshape(features_scaled.shape[0], features_scaled.shape[1], 1)

    prediction = model.predict(features_scaled)
    predicted_label = EMOTIONS[np.argmax(prediction)]

    st.markdown(f"### 🧠 Emoção Predita: **{predicted_label.upper()}**")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=EMOTIONS, y=prediction[0], palette="viridis", ax=ax)
    ax.set_ylabel("Probabilidade")
    ax.set_title("Distribuição de Probabilidades por Emoção")
    ax.set_ylim(0, 1)
    st.pyplot(fig)
    os.remove(temp_audio_path)
