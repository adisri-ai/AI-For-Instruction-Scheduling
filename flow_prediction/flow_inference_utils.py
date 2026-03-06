import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle
import re
import ast
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from training import build_cfg
#Prediction helpers

def load_artifacts(prefix="cfg_lstm"):
    model = load_model(f"{prefix}_model.keras")
    with open(f"{prefix}_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open(f"{prefix}_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return model, tokenizer, meta


def predict_decision(input_context: str, model, tokenizer, meta, threshold=0.5):
    seq = tokenizer.texts_to_sequences([input_context])
    padded = pad_sequences(seq, maxlen=meta.get("maxlen", 40), padding="pre")
    prob = float(model.predict(padded, verbose=0)[0, 0])

    return {"probability": prob, "decision": "ENTER" if prob >= threshold else "SKIP"}
