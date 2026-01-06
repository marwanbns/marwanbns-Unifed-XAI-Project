from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image


def wav_to_melspec_png(
    wav_path: str | Path,
    out_png_path: str | Path,
    target_size: Tuple[int, int] = (224, 224),
): # Pour convertir un fichier wav en spectrogramme
    wav_path = Path(wav_path)
    out_png_path = Path(out_png_path)
    out_png_path.parent.mkdir(parents=True, exist_ok=True)

    # On charge l'audio
    y, sr = librosa.load(str(wav_path), sr=None)

    # Le spectrogramme avec passage en db
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    # Figure
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    librosa.display.specshow(log_ms, sr=sr, ax=ax)
    ax.axis("off")

    fig.savefig(str(out_png_path), bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    img = Image.open(out_png_path).convert("RGB")
    img = img.resize(target_size)
    return img


def pil_to_tensor(img: Image.Image):
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)