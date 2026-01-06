from __future__ import annotations
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms

# On normalise comme aec ImageNet
VAL_TFMS = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

def load_xray_as_tensor(image_path: str | Path):
    image_path = Path(image_path)
    img = Image.open(image_path).convert("RGB")
    x = VAL_TFMS(img).unsqueeze(0)
    return x