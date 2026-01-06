from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional
InputType = Literal["audio", "image"]


@dataclass(frozen=True) # Classe réservé aux modèles
class ModelSpec:
    id: str
    name: str
    input_type: InputType
    framework: str
    path: str
    arch: str | None = None


@dataclass(frozen=True)
class XaiSpec:
    id: str
    name: str
    compatible_with: List[InputType]


MODELS: List[ModelSpec] = [
    ModelSpec(
        id="mobilenet_audio",
        name="MobileNet (Audio Spectrogram)",
        input_type="audio",
        framework="tensorflow",
        path="models/audio/saved_model/model",
    ),
    # Image
    ModelSpec(
        id="densenet_xray",
        name="DenseNet (X-ray)",
        input_type="image",
        framework="pytorch",
        path="models/image/best_lunglesion_densenet121.pt",
        arch="densenet121",
    ),
    ModelSpec(
        id="alexnet_xray",
        name="AlexNet (X-ray)",
        input_type="image",
        framework="pytorch",
        path="models/image/best_lunglesion_alexnet.pt",
        arch="alexnet",
    ),
]

XAI_METHODS: List[XaiSpec] = [
    XaiSpec(id="lime", name="LIME", compatible_with=["audio"]),
    XaiSpec(id="shap", name="SHAP", compatible_with=["audio"]),
    XaiSpec(id="gradcam", name="Grad-CAM", compatible_with=["audio", "image"]),
]

# Helpers pour l'UI

def get_models_for_input(input_type: InputType):
    return [m for m in MODELS if m.input_type == input_type]


def get_xai_for_input(input_type: InputType):
    return [x for x in XAI_METHODS if input_type in x.compatible_with]


def get_model_by_id(model_id: str):
    for m in MODELS:
        if m.id == model_id:
            return m
    return None


def get_xai_by_id(xai_id: str):
    for x in XAI_METHODS:
        if x.id == xai_id:
            return x
    return None