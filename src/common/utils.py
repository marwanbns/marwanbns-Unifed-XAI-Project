from pathlib import Path

def is_audio_file(path: str | Path) -> bool:
    return str(path).lower().endswith(".wav")

def is_image_file(path: str | Path) -> bool:
    return str(path).lower().endswith((".png", ".jpg", ".jpeg"))