import streamlit as st
from PIL import Image
import io
import torch
import soundfile as sf
import librosa
import torch.nn.functional as F
import numpy as np
from transformers import (
    AutoProcessor, AutoModelForImageClassification,
    AutoModelForAudioClassification, AutoTokenizer,
    AutoModelForSequenceClassification
)
from deepface import DeepFace

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Models
audio_model_name = "superb/wav2vec2-base-superb-er"
text_model_name = "bhadresh-savani/bert-base-go-emotion"

audio_extractor = AutoProcessor.from_pretrained(audio_model_name)
audio_model = AutoModelForAudioClassification.from_pretrained(audio_model_name).to(device)
audio_labels = {int(k): v for k,v in audio_model.config.id2label.items()}

text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_model = AutoModelForSequenceClassification.from_pretrained(text_model_name).to(device)
text_labels = {int(k): v for k,v in text_model.config.id2label.items()}

COMMON = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Predict Functions
def predict_face(img_path):
    preds = DeepFace.analyze(img_path, actions=['emotion'], enforce_detection=False)
    emotions = preds[0]['emotion']
    return {k.lower(): v/100.0 for k, v in emotions.items()}

def predict_audio(audio_path, target_sr=16000):
    speech, sr = sf.read(audio_path)
    if speech.ndim > 1:
        speech = speech.mean(axis=1)
    if sr != target_sr:
        speech = librosa.resample(speech, orig_sr=sr, target_sr=target_sr)
    inputs = audio_extractor(speech, sampling_rate=target_sr, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = audio_model(**inputs).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

    mapped = {}
    for i, p in enumerate(probs):
        lab = audio_labels[i].lower()
        if "neu" in lab: mapped["neutral"] = float(p)
        elif "hap" in lab: mapped["happy"] = float(p)
        elif "ang" in lab: mapped["anger"] = float(p)
        elif "sad" in lab: mapped["sad"] = float(p)
    return mapped

def predict_text(text):
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = text_model(**inputs).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

    mapped = {}
    for i, p in enumerate(probs):
        lab = text_labels[i].lower()
        if "anger" in lab: mapped["anger"] = mapped.get("anger", 0) + float(p)
        elif "sad" in lab: mapped["sad"] = mapped.get("sad", 0) + float(p)
        elif "happy" in lab or "amusement" in lab or "joy" in lab: mapped["happy"] = mapped.get("happy", 0) + float(p)
        elif "neutral" in lab: mapped["neutral"] = mapped.get("neutral", 0) + float(p)
        elif "fear" in lab: mapped["fear"] = mapped.get("fear", 0) + float(p)
        elif "surprise" in lab: mapped["surprise"] = mapped.get("surprise", 0) + float(p)
        elif "disgust" in lab: mapped["disgust"] = mapped.get("disgust", 0) + float(p)
    return mapped

def fuse_predictions(face_map, audio_map, text_map, w_face=0.4, w_audio=0.3, w_text=0.3):
    fused = {c: 0.0 for c in COMMON}
    for c in COMMON:
        fused[c] += w_face * face_map.get(c, 0.0)
        fused[c] += w_audio * audio_map.get(c, 0.0)
        fused[c] += w_text * text_map.get(c, 0.0)
    s = sum(fused.values()) or 1.0
    for k in fused: fused[k] = fused[k] / s
    pred = max(fused, key=fused.get)
    return pred, fused

# === Streamlit UI ===
st.title("🎭 Multimodal Emotion Recognition")

st.write("Upload a face image, an audio clip (.wav), and enter a sentence. The model will predict the emotion based on all three.")

face_file = st.file_uploader("📷 Upload Face Image (jpg/jpeg)", type=["jpg", "jpeg"])
audio_file = st.file_uploader("🔊 Upload Audio File (.wav)", type=["wav"])
text_input = st.text_area("✏️ Enter a sentence here:")

if st.button("🔍 Predict"):
    if not face_file or not audio_file or not text_input:
        st.error("Please upload all three inputs.")
    else:
        # Save uploaded files
        face_path = "temp_face.jpg"
        audio_path = "temp_audio.wav"
        with open(face_path, "wb") as f:
            f.write(face_file.read())
        with open(audio_path, "wb") as f:
            f.write(audio_file.read())

        # Run predictions
        face_map = predict_face(face_path)
        audio_map = predict_audio(audio_path)
        text_map = predict_text(text_input)
        pred, fused = fuse_predictions(face_map, audio_map, text_map)

        st.success(f"🎯 Predicted Emotion: {pred.upper()}")
        st.write("Confidence Scores:")
        for k, v in sorted(fused.items(), key=lambda x: -x[1]):
            st.write(f"{k.capitalize()}: {v:.3f}")
        st.image(face_path, caption="Uploaded Face Image")
