from __future__ import annotations
from pathlib import Path
import uuid


def ensure_dir(path: str | Path): # Crée le dossier s'il n'exitse pas
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_uploaded_file(uploaded_file, out_dir: str | Path):
    out_dir = ensure_dir(out_dir)

    suffix = Path(uploaded_file.name).suffix.lower()
    filename = f"{uuid.uuid4().hex}{suffix}"
    out_path = out_dir / filename

    out_path.write_bytes(uploaded_file.getbuffer())
    return out_path


def make_cache_path(input_path: str | Path, cache_dir: str | Path, new_suffix: str):
    cache_dir = ensure_dir(cache_dir)
    input_path = Path(input_path)
    return cache_dir / f"{input_path.stem}_{uuid.uuid4().hex}{new_suffix}"