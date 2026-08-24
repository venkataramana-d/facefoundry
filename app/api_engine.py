"""
Advanced generation engine: turns each uploaded photo into a professional
corporate headshot by calling a modern image model (Google Gemini 2.5 Flash
Image, aka "Nano Banana") over HTTPS.

This is a drop-in alternative to the Kaggle/SDXL+InstantID engine. It writes to
the exact same job-folder layout the web UI already reads:
    jobs/<id>/input/<stem>.<ext>              (source, for the "before" view)
    jobs/<id>/output/headshots_out/<stem>.jpg (result, for the "after" view)
    jobs/<id>/output/results.json             (summary the UI polls)

Selection is automatic: if an API key is configured (and the engine isn't
forced to "kaggle"), runner.py routes jobs here. Otherwise it uses Kaggle.

No third-party dependencies: uses urllib from the stdlib. PIL (already a project
dependency) is used only for final resize / JPEG encoding.
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

from .kaggle_client import JobError, JOBS_DIR, REPO

# ----------------------------------------------------------------------------
# Configuration (all via environment; no secrets in code)
# ----------------------------------------------------------------------------
# The key: any of these env vars works so the user can name it naturally.
_KEY_VARS = ("FACEFOUNDRY_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")
# GA image model. Override with FACEFOUNDRY_API_MODEL if Google renames it.
_DEFAULT_MODEL = "gemini-2.5-flash-image"
_ENDPOINT = ("https://generativelanguage.googleapis.com/v1beta/models/"
             "{model}:generateContent")

_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
         ".webp": "image/webp"}


def api_key() -> str | None:
    # 1) environment variables (best for hosted deploys)
    for v in _KEY_VARS:
        val = os.getenv(v)
        if val and val.strip():
            return val.strip()
    # 2) a local, git-ignored file so the user can drop the key in without
    #    restarting or exposing it in chat. Read fresh each call (no caching)
    #    so creating/editing the file takes effect on the next Generate.
    try:
        f = REPO / ".api_key"
        if f.is_file():
            txt = f.read_text(encoding="utf-8").strip()
            if txt:
                return txt
    except Exception:
        pass
    return None


def is_configured() -> bool:
    """True when the API engine should handle jobs: a key is present and the
    engine isn't explicitly forced to Kaggle."""
    if os.getenv("FACEFOUNDRY_ENGINE", "").strip().lower() == "kaggle":
        return False
    return api_key() is not None


def engine_name() -> str:
    return "api" if is_configured() else "kaggle"


# ----------------------------------------------------------------------------
# Prompts: instruction sent to the image editor for each style. The person's
# real photo is attached, so these are edit instructions, not text-to-image.
# ----------------------------------------------------------------------------
_IDENTITY = ("Keep the person's face, identity, likeness, bone structure, skin "
             "tone, age, hairstyle and any eyeglasses exactly as in the original "
             "photo. Do not change their facial features or make them look like a "
             "different person.")

_COMMON = ("Natural, true-to-life color and realistic skin texture (not plastic or "
           "waxy). Sharp focus on the eyes. Head-and-shoulders, centered, facing "
           "the camera, calm confident neutral expression. Ultra realistic "
           "photograph, high resolution. Do not add any logo, text, or watermark.")

# Per-category background + lighting scene (Gemini isn't token-limited, so these
# can be descriptive). Chosen to match each style's worker preset.
_STYLE_SCENE = {
    "edstellar_executive": "Place them on a pure white seamless studio background with bright, even softbox lighting.",
    "corporate": "Place them on a smooth neutral light grey studio background with soft, even softbox lighting.",
    "formal_executive": "Place them on a deep navy studio background with a subtle vignette and dramatic soft directional lighting.",
    "linkedin_classic": "Place them on a smooth light blue-grey gradient background with bright, even softbox lighting.",
    "modern_tech": "Place them in a softly blurred modern office background with gentle bokeh and bright natural window light.",
    "warm_friendly": "Place them on a soft cream beige background with warm, flattering light.",
    "startup_casual": "Place them on a bright, airy white background with natural daylight.",
    "healthcare": "Place them on a bright, clean clinical white background with soft, even lighting.",
    "academic": "Place them in a warm, softly blurred library background with soft directional light.",
}

_STYLE_WARDROBE = {
    "edstellar_executive": (
        "Dress the person in a tailored dark navy blue business suit jacket, a "
        "crisp light blue dress shirt, and a dark navy blue tie with small evenly "
        "spaced white polka dots."),
    "corporate": (
        "Dress the person in a charcoal grey business suit and a white dress shirt."),
    "formal_executive": (
        "Dress the person in a black formal business suit, white shirt and a "
        "conservative dark tie."),
    "linkedin_classic": (
        "Dress the person in a navy blazer over a light shirt, smart business "
        "casual."),
    "modern_tech": (
        "Dress the person in a smart dark blazer over a plain crew-neck top, "
        "modern professional look."),
    "warm_friendly": (
        "Dress the person in a soft blue shirt or light blazer, approachable "
        "professional look."),
    "startup_casual": (
        "Dress the person in a clean solid-color collared shirt, smart casual "
        "professional look."),
    "healthcare": (
        "Dress the person in a clean white medical coat over professional attire."),
    "academic": (
        "Dress the person in a tweed or dark blazer over a collared shirt, "
        "scholarly professional look."),
}


def _prompt_for(cfg: dict) -> str:
    style = cfg.get("style_preset", "corporate")
    wardrobe = _STYLE_WARDROBE.get(style, _STYLE_WARDROBE["corporate"])
    scene = _STYLE_SCENE.get(style, _STYLE_SCENE["corporate"])
    # A custom background from the user, or the white-background toggle, overrides
    # the style's default scene.
    bg = (cfg.get("background") or "").strip()
    if bg:
        scene = f"Place them on a {bg} background with soft, even lighting."
    elif cfg.get("white_background") and style != "edstellar_executive":
        scene = "Place them on a pure white seamless studio background with bright, even softbox lighting."
    extra = (cfg.get("custom_prompt") or "").strip()
    parts = [
        "Transform this photo into a professional corporate headshot portrait.",
        _IDENTITY, wardrobe, scene, _COMMON,
    ]
    if extra:
        parts.append(extra)
    return " ".join(parts)


# ----------------------------------------------------------------------------
# HTTP call to the image model
# ----------------------------------------------------------------------------
def _call_gemini(prompt: str, img_bytes: bytes, mime: str, key: str,
                 model: str, timeout: int = 120) -> bytes:
    """Return generated image bytes, or raise JobError with a readable message."""
    url = _ENDPOINT.format(model=model)
    body = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime,
                                 "data": base64.b64encode(img_bytes).decode()}},
            ],
        }],
        "generationConfig": {"responseModalities": ["IMAGE"], "temperature": 0.35},
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    # Key travels in a header, never in the URL/query string.
    req.add_header("x-goog-api-key", key)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = json.loads(e.read().decode()).get("error", {}).get("message", "")
        except Exception:
            pass
        if e.code in (401, 403):
            raise JobError("Image API rejected the key (401/403). Check the API "
                           "key is valid and the Generative Language API is enabled. "
                           f"{detail}")
        if e.code == 429:
            raise JobError("Image API rate/quota limit hit (429). Wait a moment or "
                           f"check your quota. {detail}")
        raise JobError(f"Image API HTTP {e.code}: {detail or e.reason}")
    except urllib.error.URLError as e:
        raise JobError(f"Could not reach the image API: {e.reason}")

    if "error" in payload:
        raise JobError(f"Image API error: {payload['error'].get('message', payload['error'])}")

    # Dig out the returned image part (REST v1beta uses camelCase inlineData).
    for cand in payload.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                return base64.b64decode(inline["data"])
    # No image came back: surface any text/refusal so the user sees why.
    text = ""
    for cand in payload.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            if part.get("text"):
                text += part["text"]
        fr = cand.get("finishReason")
        if fr and fr not in ("STOP", "MAX_TOKENS"):
            text = f"[{fr}] {text}"
    raise JobError(f"Image API returned no image. {text[:300]}".strip())


# ----------------------------------------------------------------------------
# Job runner (same signature/contract as kaggle_client.run_job)
# ----------------------------------------------------------------------------
def _emit(on_event, stage: str, message: str, **extra) -> None:
    if on_event:
        try:
            on_event(stage, message, extra)
        except Exception:
            pass


def run_api_job(images_dir: Path, cfg: dict, job_id: str, on_event=None) -> dict:
    """Generate headshots for every image in images_dir via the image API.
    Mirrors kaggle_client.run_job: returns a summary dict, raises JobError,
    emits (stage, message, extra) progress."""
    key = api_key()
    if not key:
        raise JobError("No image API key configured. Set FACEFOUNDRY_API_KEY "
                       "(a Google AI Studio key) and restart.")
    model = os.getenv("FACEFOUNDRY_API_MODEL", "").strip() or _DEFAULT_MODEL

    job_dir = JOBS_DIR / job_id
    in_dir = job_dir / "input"
    out_dir = job_dir / "output" / "headshots_out"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    _emit(on_event, "pack", "Preparing images")
    # Collect uploads -> input/ (so the "before" view resolves), preserving stem.
    sources: list[tuple[str, Path]] = []
    for p in sorted(Path(images_dir).iterdir()):
        if p.is_file() and p.suffix.lower() in _MIME:
            dest = in_dir / p.name
            if p.resolve() != dest.resolve():
                shutil.copy2(p, dest)
            sources.append((p.stem, dest))
    limit = cfg.get("limit")
    if isinstance(limit, int) and limit > 0:
        sources = sources[:limit]
    total = len(sources)
    if total == 0:
        raise JobError("No usable images found for this job.")

    prompt = _prompt_for(cfg)
    out_size = int(cfg.get("output_size") or 1024)

    _emit(on_event, "running", f"Generating {total} headshot(s) via image API",
          total=total, ok=0, failed=0)

    from PIL import Image  # local import; PIL is a project dependency

    results = []
    ok = failed = 0
    for i, (stem, src) in enumerate(sources, 1):
        try:
            raw = src.read_bytes()
            mime = _MIME.get(src.suffix.lower(), "image/jpeg")
            gen = _retry(lambda: _call_gemini(prompt, raw, mime, key, model),
                         attempts=2)
            img = Image.open(io.BytesIO(gen)).convert("RGB")
            # Upscale to requested output size if the model returned smaller.
            if out_size > max(img.size):
                img = img.resize((out_size, out_size), Image.LANCZOS)
            dst = out_dir / f"{stem}.jpg"
            img.save(dst, "JPEG", quality=95, subsampling=0)
            results.append({"stem": stem, "status": "ok", "output": f"{stem}.jpg",
                            "seed": cfg.get("seed_base", 0)})
            ok += 1
        except Exception as e:  # noqa: BLE001 - one bad image must not kill the batch
            results.append({"stem": stem, "status": "failed", "error": str(e)[:300]})
            failed += 1
        _emit(on_event, "running", f"Generated {i}/{total}",
              total=total, ok=ok, failed=failed)

    _emit(on_event, "download", "Finalizing")
    summary = {
        "job_id": job_id, "engine": "api", "model": model,
        "style_preset": cfg.get("style_preset"), "total": total, "ok": ok,
        "failed": failed, "config": cfg, "results": results,
    }
    (job_dir / "output" / "results.json").write_text(json.dumps(summary, indent=2))
    if ok == 0:
        # Nothing succeeded: raise so the UI shows the failure + reason.
        first_err = next((r.get("error") for r in results if r.get("error")), "unknown error")
        raise JobError(f"All {total} image(s) failed. First error: {first_err}")
    _emit(on_event, "done", f"job done - ok={ok} failed={failed}",
          ok=ok, failed=failed, total=total)
    return summary


def _retry(fn, attempts: int = 2, delay: float = 2.0):
    last = None
    for n in range(attempts):
        try:
            return fn()
        except JobError as e:
            last = e
            msg = str(e).lower()
            # Don't retry auth/refusal errors; only transient ones.
            if any(t in msg for t in ("rejected the key", "401", "403", "no image")):
                raise
            if n < attempts - 1:
                time.sleep(delay)
    raise last
