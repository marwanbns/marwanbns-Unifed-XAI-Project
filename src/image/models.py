from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

IMAGE_CLASS_NAMES = ["benign", "malignant"]  # ou ["negative", "positive"]

def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_alexnet_binary():
    m = models.alexnet(weights=None)
    in_features = m.classifier[6].in_features
    m.classifier[6] = nn.Linear(in_features, 1)
    return m

def build_densenet121_binary():
    m = models.densenet121(weights=None)
    in_features = m.classifier.in_features
    m.classifier = nn.Linear(in_features, 1)
    return m

def load_image_model(model_spec):
    ckpt = Path(model_spec.path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {ckpt}")

    if getattr(model_spec, "arch", None) == "alexnet":
        model = build_alexnet_binary()
    elif getattr(model_spec, "arch", None) in ["densenet", "densenet121"]:
        model = build_densenet121_binary()
    else:
        raise ValueError(f"Unknown image model arch: {getattr(model_spec, 'arch', None)}")

    state = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(state)
    model.to(_device())
    model.eval()
    return model

@torch.no_grad()
def predict_image(model: nn.Module, x_tensor: torch.Tensor):
    x = x_tensor.to(_device())
    logits = model(x)  # (1,1)
    p_malignant = torch.sigmoid(logits).item()
    p_benign = 1.0 - p_malignant
    probs = np.array([p_benign, p_malignant], dtype=np.float32)
    label_idx = int(np.argmax(probs))
    return label_idx, probs