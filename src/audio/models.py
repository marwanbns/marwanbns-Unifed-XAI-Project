from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

CLASS_NAMES = ["real", "fake"]


def load_audio_model(model_path: str | Path): # Methode pour charger le modèle
    return tf.keras.models.load_model(str(model_path))


def predict_audio(model: tf.keras.Model, x_tensor: np.ndarray):
    probs = model.predict(x_tensor)
    if probs.shape[-1] == 1:
        # On a un cas sigmoid binaire donc 1 seule valeur
        p_fake = float(probs[0][0])
        p_real = 1.0 - p_fake
        probs_vec = np.array([p_real, p_fake], dtype=np.float32)
        label_idx = int(np.argmax(probs_vec))
        return label_idx, probs_vec

    probs_vec = probs[0]
    label_idx = int(np.argmax(probs_vec))
    return label_idx, probs_vec