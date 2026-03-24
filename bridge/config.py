"""Centralised path resolution for ncpu-bridge.

Resolution order for NCPU models:
1. NCPU_PATH env var
2. Sibling directory ../nCPU relative to this repo
3. ~/.ncpu/models (downloaded via scripts/download_models.sh)
4. Bundled exported_models/onnx inside this package (ONNX-only fallback)

For BRIDGE_PATH (this repo root):
1. BRIDGE_PATH env var
2. Auto-detected from this file's location
"""

import os
from pathlib import Path

_BRIDGE_ROOT = Path(__file__).resolve().parent.parent

def get_bridge_path() -> Path:
    """Return the ncpu-bridge repo/package root."""
    return Path(os.environ.get("BRIDGE_PATH", str(_BRIDGE_ROOT)))

def get_ncpu_path() -> Path:
    """Return the nCPU project root (contains models/, ncpu/ etc.)."""
    explicit = os.environ.get("NCPU_PATH")
    if explicit:
        p = Path(explicit)
        if p.is_dir():
            return p

    # Sibling checkout
    sibling = _BRIDGE_ROOT.parent / "nCPU"
    if sibling.is_dir():
        return sibling

    # Home directory download location
    home = Path.home() / ".ncpu"
    if home.is_dir():
        return home

    # Fallback: bundled ONNX models inside this repo
    bundled = _BRIDGE_ROOT / "exported_models"
    if bundled.is_dir():
        return _BRIDGE_ROOT  # caller will append /models or /exported_models

    raise FileNotFoundError(
        "Cannot find nCPU models. Set NCPU_PATH env var, clone nCPU as a "
        "sibling directory, or run: scripts/download_models.sh"
    )

def get_models_dir() -> str:
    """Return the models directory path as a string."""
    ncpu = get_ncpu_path()
    models = ncpu / "models"
    if models.is_dir():
        return str(models)
    # Maybe it's the home dir layout: ~/.ncpu/models
    return str(ncpu / "models") if (ncpu / "models").is_dir() else str(ncpu)

def get_clawd_data_path(filename: str) -> Path:
    """Return path for clawd data files, configurable via CLAWD_DATA env var."""
    data_dir = Path(os.environ.get("CLAWD_DATA", str(Path.home() / ".ncpu" / "data")))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / filename

# Convenience constants (lazy — use functions above in new code)
NCPU_PATH = None  # Set on first access via __getattr__
BRIDGE_PATH = None

def __getattr__(name):
    if name == "NCPU_PATH":
        return get_ncpu_path()
    if name == "BRIDGE_PATH":
        return get_bridge_path()
    raise AttributeError(name)
