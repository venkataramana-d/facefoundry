"""
Centralized style + aspect configuration (spec §13, §14).

Single source of truth for the Corporate Trainer Profile master style and the
output aspect ratios, imported by the local web app (`server.py`). The Kaggle
worker keeps its own executable copy of the prompt in `STYLE_PRESETS` (it runs
detached on Kaggle), but every *value* it needs is mirrored here so the app and
the worker never drift and the prompt is not duplicated across UI code.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# The new master style for the trainer-image project (spec §7, §14).
# ---------------------------------------------------------------------------
CORPORATE_TRAINER_PROFILE = {
    "key": "corporate_trainer_profile",
    "name": "Corporate Trainer Profile",
    "category": "Corporate",
    "aspect_ratio": "35:45",            # new aspect; the default for this style
    "background": "white to very light-gray studio",
    "clothing": "charcoal/dark-gray suit, crisp white shirt, light-gray polka-dot tie",
    "lighting": "soft professional studio, even facial illumination",
    "expression": "calm, confident, subtle natural smile",
    "identity_preservation": "maximum",
    "logo": False,
    "text": False,
    "default_resolution": "2048",
    "default_speed": "balanced",
    "face_enhance": True,               # gentle (blended), keeps skin natural
    "white_background": False,          # backdrop comes from the prompt, not rembg
}

# The style key the whole app defaults to for the trainer project (spec §7).
DEFAULT_STYLE = CORPORATE_TRAINER_PROFILE["key"]

# ---------------------------------------------------------------------------
# Aspect ratios. `gen` is the native SDXL render size (both dims divisible by
# 64 so the UNet is happy). `portrait_3545` renders 896x1152 == 35:45 exactly
# (896/1152 = 0.7778), so we get true framing instead of padding a square.
# ---------------------------------------------------------------------------
ASPECT_DIMS = {
    "square":        (1024, 1024),   # 1:1
    "portrait":      (1024, 1024),   # 4:5 achieved by padding in post (legacy)
    "portrait_3545": (896, 1152),    # 35:45 rendered natively
}


def aspect_dims(aspect: str) -> tuple[int, int]:
    """Native (width, height) render size for an aspect key. Unknown -> square."""
    return ASPECT_DIMS.get(aspect, ASPECT_DIMS["square"])
