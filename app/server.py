"""
FaceFoundry web UI.

Local FastAPI app that turns the orchestrator into a click-and-go tool:
  New job  ->  live status  ->  review (approve / reject / re-roll)  ->  download.

Run:
    pip install -r requirements.txt
    python -m uvicorn app.server:app --reload --port 8000
    open http://localhost:8000

Nothing here touches the GPU - it drives the Kaggle orchestrator and shows
results. Job history lives in jobs/facefoundry.db.

Layout is an app shell: a fixed left sidebar (brand, nav, live job list) plus a
workspace that swaps content per route.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import secrets
import threading
import time
import zipfile
from collections import defaultdict
from html import escape
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               RedirectResponse, Response, StreamingResponse)

from . import db, runner
from .kaggle_client import JOBS_DIR, cancel_kernel, slugify_job_id

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:  # Pillow is optional at import time; upload path re-checks
    Image = None
    UnidentifiedImageError = Exception

# Upload safety limits (per POST to /jobs). Tuned for phone photos + typical batch.
MAX_FILE_BYTES = 25 * 1024 * 1024       # 25 MB per photo
MAX_FILES = 50                          # hard cap per job
MAX_TOTAL_BYTES = 400 * 1024 * 1024     # 400 MB total per job

_SAFE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9\-_]{0,63}$")
_SAFE_STEM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,127}$")


def _safe_job_id(job_id: str) -> str:
    if not _SAFE_ID_RE.match(job_id or ""):
        raise HTTPException(status_code=400, detail="invalid job id")
    return job_id


def _safe_stem(stem: str) -> str:
    if not _SAFE_STEM_RE.match(stem or ""):
        raise HTTPException(status_code=400, detail="invalid file id")
    return stem

app = FastAPI(title="FaceFoundry")

# Optional password gate for hosted deployments. Set FACEFOUNDRY_PASSWORD (and
# optionally FACEFOUNDRY_USER, default "team") as an env var on the host to
# require HTTP Basic auth. Unset -> no gate (local use is unaffected).
_SITE_PASSWORD = os.environ.get("FACEFOUNDRY_PASSWORD", "")
_SITE_USER = os.environ.get("FACEFOUNDRY_USER", "team")

# Strict auth is OPT-IN. Set FACEFOUNDRY_REQUIRE_AUTH=1 to force a password (the
# app then serves only /healthz until FACEFOUNDRY_PASSWORD is set). By default,
# leaving FACEFOUNDRY_PASSWORD unset simply runs the app open (no login) - which
# is what you want when you deliberately remove the password to share a link.
_HOSTED = bool(os.environ.get("FACEFOUNDRY_REQUIRE_AUTH"))

# Simple in-memory per-IP rate limit for state-changing requests (resets on
# restart). Blunts abuse/CSRF-spray on top of the concurrent-job cap.
_RATE_MAX = 120         # unsafe requests per window per IP (roomy for rapid review;
_RATE_WINDOW = 60       # automated abuse does far more) - seconds window
_rate_hits: dict[str, list] = defaultdict(list)
_rate_lock = threading.Lock()
_UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


@app.middleware("http")
async def _security(request, call_next):
    path = request.url.path
    if path == "/healthz":                       # always open for health checks
        return await call_next(request)

    # 1) Fail closed when hosted without a password configured.
    if _HOSTED and not _SITE_PASSWORD:
        return Response(
            "This deployment has no access password configured. Set the "
            "FACEFOUNDRY_PASSWORD environment variable to enable access.",
            status_code=503)

    # 2) HTTP Basic password gate.
    if _SITE_PASSWORD:
        ok = False
        header = request.headers.get("authorization", "")
        if header.startswith("Basic "):
            try:
                user, _, pw = base64.b64decode(header[6:]).decode().partition(":")
                ok = (secrets.compare_digest(user, _SITE_USER)
                      and secrets.compare_digest(pw, _SITE_PASSWORD))
            except Exception:
                ok = False
        if not ok:
            return Response("Authentication required", status_code=401,
                            headers={"WWW-Authenticate": 'Basic realm="FaceFoundry"'})

    # 3) State-changing requests: CSRF (same-origin) + rate limit.
    if request.method in _UNSAFE_METHODS:
        # CSRF: a cross-site form/fetch carries the attacker's Origin/Referer.
        # Accept any host the platform legitimately presents (proxies rewrite Host
        # / add X-Forwarded-Host) so real same-origin POSTs are never falsely blocked.
        src = request.headers.get("origin") or request.headers.get("referer") or ""
        if src:
            allowed = {h for h in (request.headers.get("host"),
                                   request.headers.get("x-forwarded-host"),
                                   request.url.netloc) if h}
            if urlparse(src).netloc not in allowed:
                return Response("cross-origin request blocked", status_code=403)
        # Rate limit per client IP.
        ip = request.client.host if request.client else "?"
        now = time.time()
        with _rate_lock:
            hits = [t for t in _rate_hits[ip] if now - t < _RATE_WINDOW]
            if len(hits) >= _RATE_MAX:
                _rate_hits[ip] = hits
                return Response("Too many requests - slow down.", status_code=429)
            hits.append(now)
            _rate_hits[ip] = hits

    return await call_next(request)

STYLES = [
    ("corporate", "Corporate", "Charcoal suit, neutral studio grey"),
    ("modern_tech", "Modern Tech", "Smart-casual, soft office bokeh"),
    ("warm_friendly", "Warm & Friendly", "Cozy tone, golden-hour light"),
    ("formal_executive", "Formal Executive", "Navy backdrop, editorial lighting"),
    ("linkedin_classic", "LinkedIn Classic", "Blue-grey gradient, business casual"),
    ("startup_casual", "Startup Casual", "Bright white, relaxed and modern"),
    ("healthcare", "Healthcare", "Clinical white, medical coat"),
    ("academic", "Academic", "Muted library, tweed and cardigan"),
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Output resolution options (final saved size; SDXL renders at 1024 then upscales).
RESOLUTIONS = [
    ("1024", "Standard", "1K · fastest"),
    ("2048", "High", "2K · sharp"),
    ("4096", "Ultra", "4K · best quality"),
]
# Render-speed presets -> diffusion steps (fewer = faster, more = higher fidelity).
SPEEDS = [
    ("fast", "Fast", "~20 steps"),
    ("balanced", "Balanced", "~30 steps"),
    ("best", "Best", "~45 steps"),
]
SPEED_STEPS = {"fast": 20, "balanced": 30, "best": 45}

# The status stepper (kept in sync with runner._STAGE_STEP).
STEPS = [
    ("Prepare images", "Packaging your photos"),
    ("Upload to Kaggle", "Sending a private dataset"),
    ("Launch GPU worker", "Starting the Tesla P100"),
    ("Generate headshots", "AI is rendering each face"),
    ("Download results", "Pulling finished headshots back"),
    ("Complete", "Ready to review"),
]

db.init()  # idempotent; ensures tables exist regardless of startup-event timing

# On (re)start, any job still marked running/queued lost its polling thread.
# Instead of killing it, RECONNECT to its Kaggle kernel and keep tracking to
# completion - so a restart never loses an in-flight render. (If the kernel
# already errored, the resumed poll detects it and marks the job failed.)
def _resume_orphans() -> None:
    for _j in db.list_jobs():
        # Only reconnect jobs that actually launched a kernel (a kernel dir was
        # written). Skip rows that never got that far - nothing to resume.
        if _j["status"] in ("running", "queued") and (JOBS_DIR / _j["id"] / "kernel").is_dir():
            try:
                runner.resume_running(_j["id"])
            except Exception:
                pass  # never let startup crash on a bad row


_resume_orphans()


# ============================================================================
# SVG icon set (inline, single stroke family - no emoji)
# ============================================================================
def _svg(path: str, size: int = 20, extra: str = "") -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
            f'stroke-linejoin="round" {extra}>{path}</svg>')


ICON = {
    "logo": _svg('<path d="M12 2 3 7v10l9 5 9-5V7z"/><circle cx="12" cy="11" r="3"/>'
                 '<path d="M8 19a4 4 0 0 1 8 0"/>', 22),
    "check": _svg('<path d="M20 6 9 17l-5-5"/>'),
    "x": _svg('<path d="M18 6 6 18M6 6l12 12"/>'),
    "plus": _svg('<path d="M12 5v14M5 12h14"/>'),
    "grid": _svg('<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/>'
                 '<rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>'),
    "download": _svg('<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
                     '<path d="M7 10l5 5 5-5"/><path d="M12 15V3"/>'),
    "reroll": _svg('<path d="M3 12a9 9 0 0 1 15-6.7L21 8"/><path d="M21 3v5h-5"/>'
                   '<path d="M21 12a9 9 0 0 1-15 6.7L3 16"/><path d="M3 21v-5h5"/>'),
    "back": _svg('<path d="M19 12H5"/><path d="M12 19l-7-7 7-7"/>'),
    "folder": _svg('<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>'),
    "spark": _svg('<path d="M12 3v4M12 17v4M3 12h4M17 12h4"/>'
                  '<path d="M12 8a4 4 0 0 0 4 4 4 4 0 0 0-4 4 4 4 0 0 0-4-4 4 4 0 0 0 4-4z"/>'),
    "alert": _svg('<path d="M12 9v4M12 17h.01"/>'
                  '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>'),
    "trash": _svg('<path d="M3 6h18"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>'
                  '<path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6M14 11v6"/>'),
    "edit": _svg('<path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>'
                 '<path d="M18.5 2.5a2.1 2.1 0 0 1 3 3L12 15l-4 1 1-4z"/>'),
}


# ============================================================================
# Helpers to locate a job's files on disk
# ============================================================================
def _input_path(job_id: str, stem: str) -> Path | None:
    _safe_job_id(job_id); _safe_stem(stem)
    d = JOBS_DIR / job_id / "input"
    if not d.is_dir():
        return None
    for ext in IMAGE_EXTS:
        p = d / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def _output_path(job_id: str, stem: str) -> Path | None:
    _safe_job_id(job_id); _safe_stem(stem)
    d = JOBS_DIR / job_id / "output" / "headshots_out"
    if d.is_dir():
        p = d / f"{stem}.jpg"
        if p.is_file():
            return p
    # Legacy fallback: some worker versions wrote elsewhere under output/.
    d2 = JOBS_DIR / job_id / "output"
    if d2.is_dir():
        target = f"{stem}.jpg"
        for p in d2.rglob("*.jpg"):
            if p.name == target:
                return p
    return None


def _results(job_id: str) -> dict | None:
    p = JOBS_DIR / job_id / "output" / "results.json"
    if p.is_file():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _elapsed(created_at: str | None) -> int:
    if not created_at:
        return 0
    try:
        return max(0, int(time.time() - time.mktime(time.strptime(created_at, "%Y-%m-%d %H:%M:%S"))))
    except Exception:
        return 0


def _fmt_elapsed(sec: int) -> str:
    if sec < 60:
        return f"{sec}s"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


# ============================================================================
# App shell (sidebar + workspace) - the structural frame for every page
# ============================================================================
def _sidebar(active: str) -> str:
    jobs = db.list_jobs(limit=12)
    items = ""
    for j in jobs:
        st = j["status"]
        dot = {"done": "d-ok", "failed": "d-bad", "running": "d-run", "queued": "d-run"}.get(st, "d-run")
        meta = (f'{j["ok"]}/{j["total"]}' if st == "done" and j["total"]
                else st if st in ("failed",) else f'step {j["step"] or 0}/6')
        items += (f'<a class="jitem" href="/jobs/{escape(j["id"])}">'
                  f'<span class="dot {dot}"></span>'
                  f'<span class="ji-name">{escape(j["id"])}</span>'
                  f'<span class="ji-meta">{escape(meta)}</span></a>')
    if not items:
        items = '<div class="jempty">No jobs yet. Create one to get started.</div>'

    return f"""
    <aside class=sidebar>
      <a class=brand href="/"><span class=mark>{ICON['logo']}</span><span>FaceFoundry</span></a>
      <a class="navbtn {'on' if active=='new' else ''}" href="/">{ICON['plus']} New job</a>
      <div class=navlabel>Recent jobs</div>
      <nav class=jlist>{items}</nav>
      <div class=sidefoot><span class=dot d-ok></span> Free Kaggle GPU · $0 per batch</div>
    </aside>"""


def _shell(title: str, active: str, topbar: str, content: str, extra_head: str = "") -> str:
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<title>{escape(title)}</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root {{
  --bg:#eef1f6; --panel:#ffffff; --panel-2:#f4f7fb; --line:#e2e8f1; --line-2:#d2dae6;
  --fg:#182338; --muted:#576174; --faint:#8592a6; --field:#f7f9fc;
  --accent:#1e56a0; --accent-2:#17457f; --accent-soft:rgba(30,86,160,.09);
  --grad:linear-gradient(180deg,#2461ac,#1e56a0);
  --ok:#177a4a; --ok-soft:rgba(23,122,74,.10); --bad:#b42318; --bad-soft:rgba(180,35,24,.08);
  --shadow:0 1px 2px rgba(20,33,61,.04),0 3px 10px rgba(20,33,61,.06);
  --r:12px;
}}
* {{ box-sizing:border-box; }}
html,body {{ height:100%; }}
body {{ margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }}
a {{ color:inherit; text-decoration:none; }}
svg {{ display:block; }}
.dot {{ width:8px; height:8px; border-radius:50%; flex:none; display:inline-block; }}
.d-ok {{ background:var(--ok); }} .d-bad {{ background:var(--bad); }}
.d-run {{ background:var(--accent); animation:blink 1.4s ease-in-out infinite; }}
@keyframes blink {{ 50% {{ opacity:.35; }} }}

/* ---- app shell ---- */
.app {{ display:flex; min-height:100dvh; }}
.sidebar {{ width:266px; flex:none; background:var(--panel); border-right:1px solid var(--line);
  padding:18px 14px; display:flex; flex-direction:column; gap:6px; position:sticky; top:0; height:100dvh; }}
.brand {{ display:flex; align-items:center; gap:11px; font-weight:700; font-size:17px; letter-spacing:-.3px; padding:6px 8px 16px; }}
.brand .mark {{ display:grid; place-items:center; width:34px; height:34px; border-radius:10px; background:var(--grad); color:#fff; }}
.navbtn {{ display:flex; align-items:center; gap:10px; font-weight:600; padding:11px 12px; border-radius:11px;
  background:var(--grad); color:#fff; box-shadow:0 6px 18px rgba(124,92,255,.3); }}
.navbtn svg {{ width:18px; height:18px; }}
.navbtn.ghost {{ background:var(--field); color:var(--fg); box-shadow:none; }}
.navlabel {{ font-size:11px; text-transform:uppercase; letter-spacing:.6px; color:var(--faint); padding:16px 10px 6px; }}
.jlist {{ display:flex; flex-direction:column; gap:2px; overflow-y:auto; flex:1; margin:0 -4px; padding:0 4px; }}
.jitem {{ display:grid; grid-template-columns:auto 1fr auto; align-items:center; gap:9px; padding:9px 10px; border-radius:9px; transition:background .12s; }}
.jitem:hover {{ background:var(--panel-2); }}
.ji-name {{ font-size:13.5px; font-weight:500; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.ji-meta {{ font-size:11px; color:var(--faint); font-variant-numeric:tabular-nums; }}
.jempty {{ color:var(--faint); font-size:13px; padding:10px; }}
.sidefoot {{ display:flex; align-items:center; gap:8px; font-size:12px; color:var(--muted); padding:12px 10px 4px; border-top:1px solid var(--line); margin-top:6px; }}

.workspace {{ flex:1; min-width:0; display:flex; flex-direction:column; }}
.topbar {{ position:sticky; top:0; z-index:10; display:flex; align-items:center; justify-content:space-between; gap:16px;
  height:66px; padding:0 30px; background:rgba(255,255,255,.82); backdrop-filter:blur(12px); border-bottom:1px solid var(--line); }}
.topbar .tt {{ font-size:17px; font-weight:650; letter-spacing:-.3px; }}
.topbar .tt small {{ display:block; font-size:12.5px; font-weight:400; color:var(--muted); letter-spacing:0; margin-top:1px; }}
.content {{ padding:30px; max-width:1000px; width:100%; }}

/* ---- primitives ---- */
h2 {{ font-size:12px; text-transform:uppercase; letter-spacing:.7px; color:var(--faint); font-weight:600; margin:0 0 14px; }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:var(--r); padding:22px; box-shadow:var(--shadow); }}
details.panel > summary {{ cursor:pointer; font-weight:600; font-size:14px; list-style:none; }}
details.panel > summary::-webkit-details-marker {{ display:none; }}
details.panel > summary::before {{ content:"+"; display:inline-block; width:18px; color:var(--accent); font-weight:700; }}
details.panel[open] > summary::before {{ content:"-"; }}
.hint {{ font-size:12.5px; color:var(--faint); line-height:1.5; }}
/* segmented control */
.seg {{ display:grid; grid-auto-flow:column; grid-auto-columns:1fr; gap:8px; }}
.segopt {{ position:relative; }}
.segopt input {{ position:absolute; inset:0; opacity:0; cursor:pointer; width:100%; height:100%; }}
.segbox {{ display:flex; flex-direction:column; gap:2px; text-align:center; border:1.5px solid var(--line-2);
  border-radius:10px; padding:10px 8px; transition:.15s; }}
.segopt input:checked + .segbox {{ border-color:var(--accent); background:var(--accent-soft); }}
.segopt input:focus-visible + .segbox {{ box-shadow:0 0 0 3px var(--accent-soft); }}
.segn {{ font-weight:600; font-size:14px; }} .segh {{ font-size:11.5px; color:var(--muted); }}
/* toggle switch */
.switch {{ display:flex; align-items:center; gap:10px; cursor:pointer; font-size:14px; color:var(--fg); margin:0; }}
.switch input {{ position:absolute; opacity:0; width:0; height:0; }}
.switch .track {{ position:relative; width:40px; height:22px; border-radius:999px; background:var(--line-2); flex:none; transition:.15s; }}
.switch .track::after {{ content:""; position:absolute; top:2px; left:2px; width:18px; height:18px; border-radius:50%; background:#fff; box-shadow:var(--shadow); transition:.15s; }}
.switch input:checked + .track {{ background:var(--accent); }}
.switch input:checked + .track::after {{ transform:translateX(18px); }}
.switch input:focus-visible + .track {{ box-shadow:0 0 0 3px var(--accent-soft); }}
/* toast */
#toast {{ position:fixed; left:50%; bottom:24px; transform:translateX(-50%) translateY(20px); opacity:0; pointer-events:none;
  background:var(--fg); color:var(--panel); font-size:13.5px; font-weight:500; padding:10px 16px; border-radius:10px; box-shadow:var(--shadow); transition:.2s; z-index:50; }}
#toast.show {{ opacity:1; transform:translateX(-50%) translateY(0); }}
/* per-image download button overlay */
.pair figure .dl {{ position:absolute; right:8px; top:8px; width:30px; height:30px; border-radius:8px; display:grid; place-items:center;
  background:rgba(255,255,255,.9); color:var(--fg); opacity:0; transition:.15s; }}
.pair figure:hover .dl {{ opacity:1; }} .pair figure .dl svg {{ width:16px; height:16px; }}
.stack > * + * {{ margin-top:20px; }}
label {{ display:block; font-size:13px; color:var(--muted); margin:0 0 7px; }}
input,select {{ width:100%; font:inherit; color:var(--fg); background:var(--field); border:1px solid var(--line-2);
  border-radius:10px; padding:11px 13px; transition:border-color .15s,box-shadow .15s; }}
input::placeholder {{ color:var(--faint); }}
input:focus,select:focus {{ outline:none; border-color:var(--accent); box-shadow:0 0 0 3px var(--accent-soft); }}
.field-row {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(110px,1fr)); gap:14px; }}
.btn {{ display:inline-flex; align-items:center; justify-content:center; gap:8px; font:inherit; font-weight:600; cursor:pointer;
  border:1px solid transparent; border-radius:11px; padding:12px 20px; min-height:46px; background:var(--grad); color:#fff;
  box-shadow:0 6px 18px rgba(124,92,255,.3); transition:transform .06s,filter .15s; }}
.btn:hover {{ filter:brightness(1.08); }} .btn:active {{ transform:translateY(1px); }}
.btn.ghost {{ background:var(--field); color:var(--fg); border-color:var(--line-2); box-shadow:none; font-weight:500; }}
.btn.ghost:hover {{ background:var(--panel-2); filter:none; }}
.btn.danger {{ color:var(--bad); }} .btn.danger:hover {{ background:var(--bad-soft); border-color:var(--bad); }}
.btn.ok {{ background:linear-gradient(135deg,#22b47a,#3ecf8e); box-shadow:0 6px 18px rgba(62,207,142,.28); }}
.btn.sm {{ padding:9px 14px; min-height:40px; font-size:14px; border-radius:9px; }}
.btn svg {{ width:18px; height:18px; }}
.actions {{ display:flex; gap:11px; flex-wrap:wrap; }}
.badge {{ display:inline-flex; align-items:center; gap:6px; font-size:12px; font-weight:600; padding:4px 11px; border-radius:999px; }}
.badge .dot {{ box-shadow:0 0 8px currentColor; }}
.b-done {{ color:var(--ok); background:var(--ok-soft); }} .b-failed {{ color:var(--bad); background:var(--bad-soft); }}
.b-running,.b-queued {{ color:var(--accent); background:var(--accent-soft); }}
.muted {{ color:var(--muted); }} .faint {{ color:var(--faint); }}
.empty {{ text-align:center; color:var(--muted); padding:44px 12px; }}

/* ---- new-job split layout ---- */
.split {{ display:grid; grid-template-columns:1.05fr .95fr; gap:20px; align-items:start; }}
.dropzone {{ position:relative; border:1.5px dashed var(--line-2); border-radius:var(--r); background:var(--field);
  min-height:280px; display:grid; place-items:center; text-align:center; padding:26px; transition:border-color .15s,background .15s; }}
.dropzone.hot {{ border-color:var(--accent); background:var(--accent-soft); }}
.dropzone input {{ position:absolute; inset:0; opacity:0; cursor:pointer; padding:0; }}
.dz-ic {{ width:56px; height:56px; border-radius:16px; background:var(--panel-2); display:grid; place-items:center; margin:0 auto 14px; color:var(--accent); }}
.dz-ic svg {{ width:26px; height:26px; }}
.dz-t {{ font-weight:600; font-size:16px; }} .dz-s {{ color:var(--muted); font-size:13.5px; margin-top:5px; }}
.dz-count {{ margin-top:12px; font-size:13px; color:var(--accent); font-weight:600; min-height:18px; }}
.styles {{ display:grid; gap:10px; }}
.srow {{ position:relative; }}
.srow input {{ position:absolute; inset:0; opacity:0; cursor:pointer; width:100%; height:100%; }}
.srow .box {{ display:flex; align-items:center; gap:13px; border:1.5px solid var(--line-2); border-radius:12px; padding:13px 15px; transition:.15s; }}
.srow input:checked + .box {{ border-color:var(--accent); background:var(--accent-soft); }}
.srow input:focus-visible + .box {{ box-shadow:0 0 0 3px var(--accent-soft); }}
.srow .swatch {{ width:34px; height:34px; border-radius:9px; flex:none; }}
.sw0 {{ background:linear-gradient(135deg,#4b5563,#9ca3af); }} .sw1 {{ background:linear-gradient(135deg,#2563eb,#38bdf8); }}
.sw2 {{ background:linear-gradient(135deg,#d97706,#fbbf24); }} .sw3 {{ background:linear-gradient(135deg,#1e3a8a,#6d28d9); }}
.sw4 {{ background:linear-gradient(135deg,#0a66c2,#7ea9d6); }} .sw5 {{ background:linear-gradient(135deg,#e2e8f0,#a3bffa); }}
.sw6 {{ background:linear-gradient(135deg,#0891b2,#e6f6fa); }} .sw7 {{ background:linear-gradient(135deg,#78350f,#b08968); }}
.srow .name {{ font-weight:600; }} .srow .desc {{ font-size:12px; color:var(--muted); }}

/* ---- status dashboard ---- */
.dash {{ display:grid; grid-template-columns:280px 1fr; gap:20px; align-items:start; }}
.ring {{ --p:0; width:150px; height:150px; border-radius:50%; margin:4px auto 0;
  background:radial-gradient(closest-side,var(--panel) 78%,transparent 79%),
            conic-gradient(from 0deg,#2461ac,#3b82c9 calc(var(--p)*1%),var(--line) 0);
  display:grid; place-items:center; }}
.ring b {{ font-size:34px; font-weight:700; letter-spacing:-1px; font-variant-numeric:tabular-nums; }}
.ring small {{ font-size:13px; color:var(--muted); }}
.nowbox {{ text-align:center; margin-top:16px; }}
.nowbox .lab {{ font-size:11px; text-transform:uppercase; letter-spacing:.6px; color:var(--faint); }}
.nowbox .big {{ font-size:19px; font-weight:650; margin:4px 0; }}
.nowbox .live {{ color:var(--muted); font-size:13px; }}
.metrow {{ display:flex; gap:20px; justify-content:center; margin-top:16px; padding-top:16px; border-top:1px solid var(--line); }}
.metrow .m .n {{ font-size:18px; font-weight:700; font-variant-numeric:tabular-nums; }} .metrow .m .l {{ font-size:11px; color:var(--faint); }}

.stepper {{ list-style:none; margin:0; padding:0; }}
.step {{ display:flex; gap:15px; padding-bottom:18px; position:relative; }}
.step:last-child {{ padding-bottom:0; }}
.step .node {{ flex:none; width:32px; height:32px; border-radius:50%; display:grid; place-items:center;
  background:var(--field); border:1.5px solid var(--line-2); color:var(--faint); z-index:1; }}
.step .node svg {{ width:17px; height:17px; }}
.step:not(:last-child) .node::after {{ content:""; position:absolute; left:16px; top:32px; bottom:0; width:2px; background:var(--line-2); }}
.step .t {{ font-weight:600; padding-top:5px; }}
.step .m {{ font-size:13px; color:var(--muted); margin-top:2px; }}
.step.done .node {{ background:linear-gradient(135deg,#22b47a,#3ecf8e); border-color:transparent; color:#04120b; }}
.step.done:not(:last-child) .node::after {{ background:var(--ok); }}
.step.active .node {{ background:var(--grad); border-color:transparent; color:#fff; box-shadow:0 0 0 5px var(--accent-soft); }}
.step.error .node {{ background:linear-gradient(135deg,#e11d48,#ff6b6b); border-color:transparent; color:#fff; }}
.step.pending .t {{ color:var(--faint); }}
.spin {{ animation:spin 1s linear infinite; }} @keyframes spin {{ to {{ transform:rotate(360deg); }} }}
@keyframes pulse {{ 50% {{ opacity:.5; }} }} .step.active .m {{ animation:pulse 1.6s ease-in-out infinite; }}
@media (prefers-reduced-motion:reduce) {{ .spin,.step.active .m,.d-run {{ animation:none; }} }}

/* ---- review gallery ---- */
.stats {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:20px; }}
.stat {{ flex:1; min-width:100px; background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px 18px; }}
.stat .n {{ font-size:26px; font-weight:700; letter-spacing:-.5px; font-variant-numeric:tabular-nums; }}
.stat .l {{ font-size:12px; color:var(--muted); }}
.gallery {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:18px; }}
.tile {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; overflow:hidden; transition:transform .12s,border-color .12s; }}
.tile:hover {{ transform:translateY(-3px); border-color:var(--line-2); }}
.tile .head {{ display:flex; align-items:center; justify-content:space-between; padding:11px 13px; }}
.pair {{ display:grid; grid-template-columns:1fr 1fr; gap:2px; background:var(--line); }}
.pair figure {{ margin:0; position:relative; background:var(--panel-2); }}
.pair img {{ width:100%; aspect-ratio:1; object-fit:cover; display:block; }}
.pair figcaption {{ position:absolute; left:8px; bottom:8px; font-size:10px; font-weight:600; letter-spacing:.4px; text-transform:uppercase;
  color:#fff; background:rgba(0,0,0,.6); padding:3px 8px; border-radius:6px; }}
.tile.approved {{ border-color:var(--ok); box-shadow:0 0 0 1.5px var(--ok); }}
.tile.rejected {{ opacity:.4; }}
.tile .ctl {{ display:flex; gap:9px; padding:12px 13px; }} .tile .ctl .btn {{ flex:1; }}
.failbox {{ display:grid; place-items:center; aspect-ratio:2/1; color:var(--bad); background:var(--bad-soft); font-size:13px; text-align:center; padding:12px; }}
pre.err {{ white-space:pre-wrap; word-break:break-word; background:#fbf1f0; border:1px solid #f0d5d2; border-radius:12px;
  padding:15px; color:var(--bad); font-size:13px; max-height:340px; overflow:auto; }}
/* image editor */
.editor {{ display:grid; grid-template-columns:1fr 330px; gap:20px; align-items:start; }}
.canvas-wrap {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:18px; display:grid; place-items:center; box-shadow:var(--shadow); min-height:440px; }}
#cv {{ max-width:100%; max-height:72vh; border-radius:8px; cursor:grab; touch-action:none; background:#fff; box-shadow:0 2px 12px rgba(20,33,61,.12); }}
#cv:active {{ cursor:grabbing; }}
.ed-sec {{ border-top:1px solid var(--line); padding:16px 0; }}
.ed-sec:first-child {{ border-top:0; padding-top:0; }}
.ed-sec h3 {{ font-size:11.5px; text-transform:uppercase; letter-spacing:.5px; color:var(--faint); margin:0 0 12px; }}
.slider {{ display:flex; align-items:center; gap:10px; margin:9px 0; font-size:13px; }}
.slider > label {{ flex:none; width:78px; margin:0; color:var(--muted); }}
.slider input[type=range] {{ flex:1; accent-color:var(--accent); }}
.slider input[type=color] {{ width:44px; height:30px; padding:2px; }}
.slider select {{ flex:1; }}
.chips {{ display:flex; gap:7px; flex-wrap:wrap; margin-bottom:6px; }}
.chip {{ padding:7px 13px; border:1px solid var(--line-2); border-radius:9px; font-size:13px; cursor:pointer; background:var(--field); user-select:none; }}
.chip.on {{ border-color:var(--accent); background:var(--accent-soft); color:var(--accent); font-weight:600; }}
.ed-status {{ font-size:12px; color:var(--muted); margin-top:8px; min-height:16px; }}
@media (max-width:900px) {{ .editor {{ grid-template-columns:1fr; }} }}

@media (max-width:900px) {{
  .app {{ flex-direction:column; }}
  .sidebar {{ width:auto; height:auto; position:static; flex-direction:row; flex-wrap:wrap; align-items:center; gap:8px; padding:12px 16px; }}
  .sidebar .navbtn {{ margin-left:auto; }}
  .jlist {{ display:none; }} .navlabel,.sidefoot {{ display:none; }} .brand {{ padding:2px 4px; }}
  .split,.dash {{ grid-template-columns:1fr; }}
  .content {{ padding:20px 16px; }}
  .topbar {{ height:auto; min-height:60px; padding:12px 16px; flex-wrap:wrap; gap:10px; }}
  .topbar .actions {{ width:100%; }}
  .gallery {{ grid-template-columns:repeat(auto-fill,minmax(240px,1fr)); }}
}}
@media (max-width:560px) {{
  .content {{ padding:16px 12px; }}
  .seg {{ grid-auto-flow:row; grid-auto-columns:auto; }}
  .stats {{ gap:8px; }} .stat {{ min-width:calc(50% - 8px); }}
  .gallery {{ grid-template-columns:1fr; }}
  .panel {{ padding:18px; }}
  h1 {{ font-size:20px; }}
  .btn {{ flex:1; }} .topbar .tt {{ font-size:16px; }}
}}
</style>{extra_head}</head><body>
<div class=app>
  {_sidebar(active)}
  <div class=workspace>
    <div class=topbar>{topbar}</div>
    <div class=content>{content}</div>
  </div>
</div></body></html>"""


def _badge(status: str) -> str:
    label = {"done": "Done", "failed": "Failed", "running": "Running", "queued": "Queued"}.get(status, status)
    return f'<span class="badge b-{escape(status)}"><span class=dot></span>{escape(label)}</span>'


def _delete_form(job_id: str) -> str:
    return (f'<form action="/jobs/{escape(job_id)}/delete" method=post style=margin:0 '
            f'onsubmit="return confirm(\'Delete job {escape(job_id)}? This removes its photos and results.\')">'
            f'<button class="btn ghost sm danger" type=submit>{ICON["trash"]} Delete</button></form>')


# ============================================================================
# Home: new job (split layout)
# ============================================================================
@app.get("/healthz")
def healthz() -> JSONResponse:
    """Unauthenticated health check for hosting platforms (Render, etc.)."""
    return JSONResponse({"status": "ok"})


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    style_rows = ""
    for i, (val, name, desc) in enumerate(STYLES):
        checked = "checked" if i == 0 else ""
        style_rows += (f'<label class=srow><input type=radio name=style value="{val}" {checked}>'
                       f'<span class=box><span class="swatch sw{i}"></span>'
                       f'<span><span class=name>{escape(name)}</span><br>'
                       f'<span class=desc>{escape(desc)}</span></span></span></label>')

    def _seg(field: str, opts: list, default_val: str) -> str:
        cells = ""
        for val, name, hint in opts:
            checked = "checked" if val == default_val else ""
            cells += (f'<label class=segopt><input type=radio name={field} value="{val}" {checked}>'
                      f'<span class=segbox><span class=segn>{escape(name)}</span>'
                      f'<span class=segh>{escape(hint)}</span></span></label>')
        return f'<div class=seg>{cells}</div>'

    topbar = ('<div class=tt>Create a new job'
              '<small>Turn ordinary photos into professional headshots. Follow the 5 steps, then click Generate.</small></div>'
              f'<button class=btn form=jobform type=submit>{ICON["spark"]} Generate headshots</button>')

    content = f"""
    <form id=jobform action="/jobs" method="post" enctype="multipart/form-data">
      <div class=split>
        <div class=panel>
          <h2>Step 1. Add your photos</h2>
          <p class=hint style="margin:-6px 0 12px">Add the faces you want turned into headshots.
            One clear, front-facing photo per person gives the best result. Choose how to add them:</p>
          <div class=upmode style="display:flex;gap:8px;margin-bottom:10px;flex-wrap:wrap">
            <button type=button class="btn sm on" id=mode-folder onclick="setMode('folder')">A whole folder</button>
            <button type=button class="btn sm" id=mode-files onclick="setMode('files')">One or more files</button>
            <button type=button class="btn sm" id=mode-camera onclick="setMode('camera')">Take a photo</button>
          </div>
          <div class=dropzone id=dz>
            <input id=images-folder type=file accept="image/*" multiple webkitdirectory
                   onchange="onPick(event)">
            <input id=images-files type=file accept="image/*" multiple
                   onchange="onPick(event)" style="display:none">
            <input id=images-camera type=file accept="image/*" capture="user"
                   onchange="onPick(event)" style="display:none">
            <div>
              <div class=dz-ic>{ICON['folder']}</div>
              <div class=dz-t id=dz-title>Drop a folder here, or click to choose one</div>
              <div class=dz-s id=dz-sub>Best results with clear, front-facing faces. JPG or PNG.</div>
              <div class=dz-count id=dzc></div>
            </div>
          </div>
          <div id=thumbs class=thumbs style="display:none;margin-top:12px;grid-template-columns:repeat(auto-fill,minmax(84px,1fr));gap:8px"></div>
          <div id=upstats class=hint style="margin-top:8px;display:none"></div>
          <p class=hint style="margin:14px 0 0"><b>What to expect:</b> the work runs on a free cloud GPU.
            The first job in a session takes about 10 to 15 minutes while the AI models download, then only
            a few seconds per photo. Later jobs are much faster. Your photos are resized and stripped of
            location data in your browser before upload, so they stay private and upload quickly.</p>
        </div>

        <div class="stack">
          <div class=panel>
            <h2>Step 2. Choose a style</h2>
            <p class=hint style="margin:-6px 0 12px">The style sets the outfit, background, and lighting of
              the finished headshot. Pick the look that suits your team ({len(STYLES)} to choose from).</p>
            <div class=styles>{style_rows}</div>
          </div>
          <div class=panel>
            <h2>Step 3. Output quality</h2>
            <p class=hint style="margin:-6px 0 12px">Higher resolution and slower speed give a sharper,
              more detailed headshot, but take a little longer to render.</p>
            <label>Resolution</label>{_seg("resolution", RESOLUTIONS, "2048")}
            <label style="margin-top:16px">Render speed</label>{_seg("speed", SPEEDS, "balanced")}
          </div>
          <div class=panel>
            <h2>Step 4. Finishing touches <span class=faint style="text-transform:none;letter-spacing:0;font-weight:400">(optional)</span></h2>
            <p class=hint style="margin:-6px 0 12px">Optional extras to polish the result. The defaults are
              a good starting point, so you can safely leave them as they are.</p>
            <label class=switch><input type=checkbox name=face_enhance value=1 checked>
              <span class=track></span><span>Sharpen the face (recommended) - cleaner eyes, skin, and detail</span></label>
            <label class=switch style="margin-top:12px"><input type=checkbox name=white_background value=1>
              <span class=track></span><span>Pure white background - the clean, formal corporate look</span></label>
            <label class=switch style="margin-top:12px"><input type=checkbox name=multi_reference value=1 checked>
              <span class=track></span><span>Use every photo of a person (recommended) - blends all their photos for a truer likeness</span></label>
            <label style="margin-top:14px">Custom background <span class=faint>(optional)</span></label>
            <input name=background placeholder="Describe a background, e.g. soft teal, warm grey, office blur">
            <label style="margin-top:14px">Extra details <span class=faint>(optional)</span></label>
            <input name=custom_prompt placeholder="Add anything to include, e.g. wearing glasses, warm smile">
          </div>
          <details class=panel>
            <summary>Step 5. Advanced settings <span class=faint>(optional - identity, guidance, seed)</span></summary>
            <div class=field-row style="margin-top:16px">
              <div><label>Limit</label><input name=limit type=number min=1 placeholder="all"></div>
              <div><label>Job name</label><input name=job_name placeholder="auto"></div>
              <div><label>Identity</label><input name=identity type=number step=0.05 value=0.8></div>
              <div><label>Adapter</label><input name=adapter type=number step=0.05 value=0.8></div>
              <div><label>Guidance</label><input name=guidance type=number step=0.5 value=5.0></div>
              <div><label>Seed</label><input name=seed type=number value=42></div>
            </div>
          </details>
        </div>
      </div>
    </form>
    <script>
      const dz = document.getElementById('dz');
      const inpFolder = document.getElementById('images-folder');
      const inpFiles = document.getElementById('images-files');
      const inpCamera = document.getElementById('images-camera');
      const bFolder = document.getElementById('mode-folder');
      const bFiles = document.getElementById('mode-files');
      const bCamera = document.getElementById('mode-camera');
      const dzTitle = document.getElementById('dz-title');
      const dzSub = document.getElementById('dz-sub');
      const dzc = document.getElementById('dzc');
      const thumbs = document.getElementById('thumbs');
      const upstats = document.getElementById('upstats');
      const form = document.getElementById('jobform');
      const MAX_FILES = {MAX_FILES};
      const MAX_BYTES = {MAX_FILE_BYTES};
      const RESIZE_MAX = 1600;   // longest-edge before upload
      const JPEG_Q = 0.9;

      // Stateful list of selected files across multiple pickers.
      let selected = [];  // Array<File>

      function fmt(n) {{ return n<1024?n+' B':n<1048576?(n/1024).toFixed(1)+' KB':(n/1048576).toFixed(1)+' MB'; }}

      function renderThumbs() {{
        thumbs.innerHTML = '';
        if (!selected.length) {{
          thumbs.style.display = 'none'; upstats.style.display = 'none'; dzc.textContent = '';
          return;
        }}
        thumbs.style.display = 'grid';
        let total = 0;
        selected.forEach((f, i) => {{
          total += f.size;
          const url = URL.createObjectURL(f);
          const cell = document.createElement('div');
          cell.style.cssText = 'position:relative;aspect-ratio:1/1;border-radius:8px;overflow:hidden;background:#f4f4f5';
          cell.innerHTML = `<img src="${{url}}" style="width:100%;height:100%;object-fit:cover" onload="URL.revokeObjectURL(this.src)">
            <button type=button aria-label="remove" style="position:absolute;top:2px;right:2px;width:22px;height:22px;border:0;border-radius:50%;background:rgba(0,0,0,.6);color:#fff;cursor:pointer;line-height:1;font-size:14px" onclick="removeAt(${{i}})">×</button>`;
          thumbs.appendChild(cell);
        }});
        dzc.textContent = selected.length + ' image' + (selected.length>1?'s':'') + ' ready';
        upstats.style.display = 'block';
        upstats.textContent = `Total ${{fmt(total)}} · will be resized to ${{RESIZE_MAX}}px & EXIF-stripped in your browser before upload`;
      }}

      window.removeAt = function(i) {{ selected.splice(i,1); renderThumbs(); }};

      function isImage(f) {{ return f && (f.type||'').startsWith('image/'); }}

      function addFiles(list) {{
        for (const f of list) {{
          if (!isImage(f)) continue;
          if (f.size > MAX_BYTES) continue;
          if (selected.length >= MAX_FILES) break;
          selected.push(f);
        }}
        renderThumbs();
      }}

      function onPick(ev) {{ addFiles(ev.target.files || []); ev.target.value = ''; }}

      function setMode(m) {{
        bFolder.classList.toggle('on', m==='folder');
        bFiles.classList.toggle('on', m==='files');
        bCamera.classList.toggle('on', m==='camera');
        inpFolder.style.display = m==='folder' ? '' : 'none';
        inpFiles.style.display = m==='files' ? '' : 'none';
        inpCamera.style.display = m==='camera' ? '' : 'none';
        dzTitle.textContent = m==='folder' ? 'Drop a folder here, or click to browse'
                            : m==='camera' ? 'Take a selfie (mobile camera)'
                            : 'Drop one or more photos here, or click to browse';
        dzSub.textContent = m==='folder' ? 'Clear, front-facing faces work best · JPG / PNG'
                          : m==='camera' ? 'Uses your device camera · one shot at a time'
                          : 'Single file or multi-select · JPG / PNG';
      }}

      ['dragenter','dragover'].forEach(e => dz.addEventListener(e, ev => {{
        ev.preventDefault(); dz.classList.add('hot');
      }}));
      ['dragleave','drop'].forEach(e => dz.addEventListener(e, ev => {{
        ev.preventDefault(); dz.classList.remove('hot');
      }}));
      dz.addEventListener('drop', ev => addFiles(ev.dataTransfer?.files || []));

      // Client-side downsize + EXIF strip. Canvas-based, no external deps:
      // decoding through <img> discards EXIF, and re-encoding to JPEG produces
      // a clean file. Keeps portrait-camera photos small (20 MB -> ~300 KB).
      async function shrinkOne(file) {{
        if (file.size < 300*1024) return file;  // small enough already, no gain
        const url = URL.createObjectURL(file);
        try {{
          const img = await new Promise((res, rej) => {{
            const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = url;
          }});
          const {{width:w, height:h}} = img;
          const scale = Math.min(1, RESIZE_MAX / Math.max(w, h));
          const nw = Math.round(w*scale), nh = Math.round(h*scale);
          const cvs = document.createElement('canvas');
          cvs.width = nw; cvs.height = nh;
          const ctx = cvs.getContext('2d');
          ctx.imageSmoothingQuality = 'high';
          ctx.drawImage(img, 0, 0, nw, nh);
          const blob = await new Promise(r => cvs.toBlob(r, 'image/jpeg', JPEG_Q));
          if (!blob || blob.size >= file.size) return file;  // never make it worse
          const base = (file.name || 'photo').replace(/\\.[^.]+$/, '') || 'photo';
          return new File([blob], base + '.jpg', {{type: 'image/jpeg'}});
        }} catch (e) {{ return file; }}
        finally {{ URL.revokeObjectURL(url); }}
      }}

      form.addEventListener('submit', async ev => {{
        if (!selected.length) {{ ev.preventDefault(); alert('Pick at least one photo.'); return; }}
        ev.preventDefault();
        const btn = form.querySelector('button[type=submit]');
        if (btn) {{ btn.disabled = true; btn.textContent = 'Preparing photos…'; }}
        upstats.textContent = 'Shrinking & stripping EXIF locally… (0 / ' + selected.length + ')';
        const shrunk = [];
        for (let i=0; i<selected.length; i++) {{
          shrunk.push(await shrinkOne(selected[i]));
          upstats.textContent = 'Shrinking & stripping EXIF locally… (' + (i+1) + ' / ' + selected.length + ')';
        }}
        const fd = new FormData(form);
        fd.delete('images');
        shrunk.forEach(f => fd.append('images', f, f.name));
        if (btn) btn.textContent = 'Uploading…';
        const r = await fetch(form.action, {{method: 'POST', body: fd, redirect: 'follow'}});
        if (r.redirected) window.location = r.url;
        else if (r.ok) window.location = '/';
        else {{ if (btn) {{ btn.disabled = false; btn.textContent = 'Generate'; }} alert('Upload failed: ' + r.status); }}
      }});
    </script>"""
    return HTMLResponse(_shell("FaceFoundry", "new", topbar, content))


# ============================================================================
# Create job
# ============================================================================
@app.post("/jobs")
async def create_job(
    images: list[UploadFile] = File(default=[]),
    style: str = Form("corporate"),
    resolution: str = Form("2048"),
    speed: str = Form("balanced"),
    face_enhance: str = Form(""),
    white_background: str = Form(""),
    multi_reference: str = Form("1"),
    background: str = Form(""),
    custom_prompt: str = Form(""),
    limit: str = Form(""),
    job_name: str = Form(""),
    identity: float = Form(0.8),
    adapter: float = Form(0.8),
    guidance: float = Form(5.0),
    seed: int = Form(42),
) -> RedirectResponse:
    job_id = slugify_job_id(job_name or f"job{int(time.time())}")
    _safe_job_id(job_id)

    incoming = images or []
    if len(incoming) > MAX_FILES:
        return RedirectResponse(f"/?error=too_many&max={MAX_FILES}", status_code=303)

    up_dir = JOBS_DIR / job_id / "uploads"
    up_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    total_bytes = 0
    for f in incoming:
        name = Path(f.filename or "").name
        if not name or Path(name).suffix.lower() not in IMAGE_EXTS:
            continue
        data = await f.read()
        if len(data) == 0 or len(data) > MAX_FILE_BYTES:
            continue
        total_bytes += len(data)
        if total_bytes > MAX_TOTAL_BYTES:
            return RedirectResponse("/?error=too_large", status_code=303)
        # Verify it's a real image, not just a lucky extension.
        if Image is not None:
            try:
                Image.open(io.BytesIO(data)).verify()
            except (UnidentifiedImageError, Exception):
                continue
        # Force the stem into our safe charset so downstream lookups can't glob-leak.
        stem = re.sub(r"[^A-Za-z0-9._\-]", "_", Path(name).stem)[:120] or f"img{saved}"
        ext = Path(name).suffix.lower()
        (up_dir / f"{stem}{ext}").write_bytes(data)
        saved += 1
    if saved == 0:
        return RedirectResponse("/?error=no_images", status_code=303)

    out_size = int(resolution) if resolution in {"1024", "2048", "4096"} else 2048
    steps = SPEED_STEPS.get(speed, 30)
    cfg = {
        "job_id": job_id, "input_mode": "images", "style_preset": style,
        "limit": int(limit) if limit.strip().isdigit() else None,
        "img_size": 1024, "output_size": out_size, "num_steps": steps, "guidance": guidance,
        "identity_scale": identity, "adapter_scale": adapter, "seed_base": seed,
        "face_enhance": bool(face_enhance), "white_background": bool(white_background),
        "multi_reference": bool(multi_reference),
        "background": background.strip()[:80], "custom_prompt": custom_prompt.strip()[:200],
    }
    db.create_job(job_id, style, cfg)
    runner.start_job(job_id, up_dir, cfg)
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


# ============================================================================
# Status JSON (polled by the progress page)
# ============================================================================
@app.get("/jobs/{job_id}/status")
def job_status(job_id: str) -> JSONResponse:
    j = db.get_job(job_id)
    if not j:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse({
        "status": j["status"], "step": j["step"] or 0, "stage": j["stage"],
        "message": j["message"], "total": j["total"], "ok": j["ok"], "failed": j["failed"],
        "elapsed": _elapsed(j["created_at"]), "elapsed_h": _fmt_elapsed(_elapsed(j["created_at"])),
    })


# ============================================================================
# Job page: live status OR review OR error
# ============================================================================
@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(job_id: str) -> HTMLResponse:
    j = db.get_job(job_id)
    if not j:
        return HTMLResponse(_shell("Not found", "", '<div class=tt>Not found</div>',
                                   '<div class="panel empty">We could not find that job. It may have been deleted. <a href="/">Start a new job</a>.</div>'), 404)
    if j["status"] in ("queued", "running"):
        return HTMLResponse(_progress_page(j))
    if j["status"] == "failed":
        return HTMLResponse(_failed_page(j))
    return HTMLResponse(_review_page(j))


def _stepper_html(current_step: int, status: str) -> str:
    rows = ""
    for i, (title, hint) in enumerate(STEPS, 1):
        if status == "failed" and i == current_step:
            cls, node = "error", ICON["alert"]
        elif i < current_step or status == "done":
            cls, node = "done", ICON["check"]
        elif i == current_step:
            cls, node = "active", f'<span class=spin>{ICON["spark"]}</span>'
        else:
            cls, node = "pending", f'<span style="font-size:12px;font-weight:600">{i}</span>'
        rows += (f'<li class="step {cls}" data-step="{i}"><span class=node>{node}</span>'
                 f'<div><div class=t>{escape(title)}</div><div class=m>{escape(hint)}</div></div></li>')
    return f'<ul class=stepper>{rows}</ul>'


def _progress_page(j: dict) -> str:
    jid = j["id"]
    step = j["step"] or 1
    total = len(STEPS)
    pct = round(step / total * 100)
    now_title = STEPS[min(step, total) - 1][0]
    topbar = (f'<div class=tt>Generating your headshots'
              f'<small>Job {escape(jid)}, started {escape(j["created_at"] or "")}. This page updates on its own.</small></div>'
              f'<div class=actions>{_delete_form(jid)}'
              f'<a class="btn ghost sm" href="/">{ICON["back"]} All jobs</a></div>')
    content = f"""
    <div class=dash>
      <div class=panel>
        <div class=ring id=ring style="--p:{pct}"><b><span id=pct>{pct}</span>%</b></div>
        <div class=nowbox>
          <div class=lab>Step <span id=stepnum>{step}</span> of {total}</div>
          <div class=big id=nowtitle>{escape(now_title)}</div>
          <div class=live id=live>{escape(j['message'] or '')}</div>
        </div>
        <div class=metrow>
          <div class=m><div class=n id=elapsed>{_fmt_elapsed(_elapsed(j['created_at']))}</div><div class=l>elapsed</div></div>
          <div class=m><div class=n>{_badge(j['status'])}</div><div class=l>status</div></div>
        </div>
      </div>
      <div class=panel>
        <h2>What is happening now</h2>
        <div id=stepper>{_stepper_html(step, j['status'])}</div>
        <p class=faint style="font-size:13px;margin:20px 0 0;border-top:1px solid var(--line);padding-top:16px">
          The AI runs on a free cloud GPU. The first run downloads the models, which takes about
          10 to 15 minutes, then each face takes only a few seconds. You can safely close this tab and
          come back later, your progress is saved.</p>
      </div>
    </div>
    <script>
    const STEP_TITLES = {json.dumps([s[0] for s in STEPS])}, TOTAL = {total};
    function render(step, status) {{
      const pct = Math.round(step / TOTAL * 100);
      ring.style.setProperty('--p', pct); pct_.textContent = pct;
      stepnum.textContent = Math.min(step, TOTAL); nowtitle.textContent = STEP_TITLES[Math.min(step,TOTAL)-1] || '';
      document.querySelectorAll('.step').forEach(el => {{
        const i = +el.dataset.step, node = el.querySelector('.node');
        el.classList.remove('done','active','pending','error');
        if (status==='failed' && i===step) {{ el.classList.add('error'); node.innerHTML='{ICON["alert"]}'; }}
        else if (i<step || status==='done') {{ el.classList.add('done'); node.innerHTML='{ICON["check"]}'; }}
        else if (i===step) {{ el.classList.add('active'); node.innerHTML='<span class=spin>{ICON["spark"]}</span>'; }}
        else {{ el.classList.add('pending'); node.innerHTML='<span style=\"font-size:12px;font-weight:600\">'+i+'</span>'; }}
      }});
    }}
    const ring=document.getElementById('ring'), pct_=document.getElementById('pct'),
          stepnum=document.getElementById('stepnum'), nowtitle=document.getElementById('nowtitle');
    if ('Notification' in window && Notification.permission === 'default') Notification.requestPermission();
    function notify(title, body) {{
      try {{ if ('Notification' in window && Notification.permission === 'granted') new Notification(title, {{body}}); }} catch(e) {{}}
    }}
    async function poll() {{
      try {{
        const s = await (await fetch("/jobs/{jid}/status")).json();
        document.getElementById('live').textContent = s.message || '';
        document.getElementById('elapsed').textContent = s.elapsed_h;
        render(s.step || 1, s.status);
        const pct = Math.round((s.step||1) / TOTAL * 100);
        document.title = pct + '% · {jid}';
        if (s.status==='done') {{ document.title='Done · {jid}'; notify('Headshots ready', '{jid}: '+(s.ok||0)+' of '+(s.total||0)+' generated'); setTimeout(()=>location.reload(), 800); return; }}
        if (s.status==='failed') {{ document.title='Failed · {jid}'; notify('Job failed', '{jid} stopped - open FaceFoundry to see why'); setTimeout(()=>location.reload(), 800); return; }}
      }} catch(e) {{}}
      setTimeout(poll, 3000);
    }}
    setTimeout(poll, 3000);
    </script>"""
    return _shell(f"{jid} · FaceFoundry", "", topbar, content)


def _failed_page(j: dict) -> str:
    jid = j["id"]
    log = JOBS_DIR / jid / "output" / "run.log"
    loglink = (f'<a class="btn ghost sm" href="/jobs/{jid}/file/run.log">View worker log</a>'
               if log.is_file() else "")
    topbar = (f'<div class=tt>This job could not finish'
              f'<small>Job {escape(jid)} stopped at step {j["step"] or "?"} of {len(STEPS)}.</small></div>'
              f'<div class=actions>{_delete_form(jid)}'
              f'<a class="btn ghost sm" href="/">{ICON["back"]} All jobs</a></div>')
    content = f"""
    <div class=panel>
      <h2 style="color:var(--bad);display:flex;gap:8px;align-items:center">{ICON['alert']} Something went wrong</h2>
      <p class=muted style="margin-top:0">The job stopped while it was <b>{escape(j['stage'] or 'starting up')}</b>.
        The technical details are below. If it is not clear, open the worker log or just start a new job and try again.</p>
      <pre class=err>{escape(j['error'] or j['message'] or 'Unknown error')}</pre>
      <div class=actions style="margin-top:16px">{loglink}
        <a class="btn sm" href="/">{ICON['plus']} Start a new job</a></div>
    </div>"""
    return _shell(f"{jid} · Failed", "", topbar, content)


def _review_page(j: dict) -> str:
    jid = j["id"]
    res = _results(jid) or {}
    reviews = db.get_reviews(jid)
    items = res.get("results", [])

    tiles = ""
    for r in items:
        stem = r.get("stem", "")
        ok = r.get("status") == "ok" and _output_path(jid, stem)
        decision = reviews.get(stem, "")
        cls = f" {decision}" if decision in ("approved", "rejected") else ""
        badge = _badge("done") if ok else _badge("failed")
        if ok:
            media = (f'<div class=pair>'
                     f'<figure><img loading=lazy src="/jobs/{jid}/image/input/{stem}" alt="original"><figcaption>before</figcaption></figure>'
                     f'<figure><img loading=lazy src="/jobs/{jid}/image/output/{stem}" alt="headshot"><figcaption>after</figcaption>'
                     f'<a class=dl href="/jobs/{jid}/download/{stem}" title="Download this headshot" onclick="event.stopPropagation()">{ICON["download"]}</a></figure></div>')
            ctl = (f'<div class=ctl>'
                   f'<button class="btn ok sm" onclick="review(\'{stem}\',\'approved\')">{ICON["check"]} Keep</button>'
                   f'<button class="btn ghost sm" onclick="review(\'{stem}\',\'rejected\')">{ICON["x"]} Reject</button>'
                   f'<a class="btn ghost sm" href="/jobs/{jid}/edit/{stem}" title="Image settings">{ICON["edit"]}</a></div>')
        else:
            media = f'<div class=failbox>{ICON["alert"]}&nbsp;{escape(r.get("error", "failed"))}</div>'
            ctl = ''
        tiles += (f'<div class="tile{cls}" id="t-{stem}" data-stem="{escape(stem)}"><div class=head>'
                  f'<span style="font-weight:600;font-size:13px">{escape(stem)}</span>{badge}</div>{media}{ctl}</div>')

    ok = res.get("ok", j["ok"] or 0); failed = res.get("failed", j["failed"] or 0); total = res.get("total", j["total"] or 0)
    approved_n = sum(1 for d in reviews.values() if d == "approved")
    # re-run same faces with a different style
    style_opts = "".join(f'<option value="{v}">{escape(n)}</option>' for v, n, _ in STYLES)
    topbar = (f'<div class=tt>Review your headshots'
              f'<small>Each card shows the original next to the new headshot. Keep the ones you like, then download. '
              f'Keyboard: A to keep, R to reject, J and K to move.</small></div>'
              f'<div class=actions>'
              f'<a class="btn ghost sm" href="/jobs/{jid}/edit">{ICON["edit"]} Edit all</a>'
              f'<a class="btn sm" href="/jobs/{jid}/download?scope=approved">{ICON["download"]} Download kept</a>'
              f'<a class="btn ghost sm" href="/jobs/{jid}/download?scope=all">{ICON["download"]} Download all</a>'
              f'{_delete_form(jid)}</div>')
    content = f"""
    <div class=stats>
      <div class=stat><div class=n>{total}</div><div class=l>total</div></div>
      <div class=stat><div class=n style="color:var(--ok)">{ok}</div><div class=l>generated</div></div>
      <div class=stat><div class=n style="color:var(--bad)">{failed}</div><div class=l>failed</div></div>
      <div class=stat><div class=n style="color:var(--accent)" id=appn>{approved_n}</div><div class=l>approved</div></div>
    </div>
    <div class=panel style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;justify-content:space-between">
      <div class=actions>
        <button class="btn ok sm" onclick="bulk('approved')">{ICON["check"]} Keep all</button>
        <button class="btn ghost sm" onclick="bulk('rejected')">{ICON["x"]} Reject all</button>
        <form action="/jobs/{jid}/reroll" method=post style=margin:0 title="Generate fresh versions of the failed and rejected photos">
          <button class="btn ghost sm" type=submit>{ICON["reroll"]} Retry the rejected ones</button></form>
      </div>
      <form action="/jobs/{jid}/rerun" method=post class=actions style="margin:0;align-items:center" title="Run the same faces again in a different style">
        <span class=faint style="font-size:13px">Try a different style:</span>
        <select name=style style="width:auto">{style_opts}</select>
        <button class="btn ghost sm" type=submit>{ICON["spark"]} Run again</button>
      </form>
    </div>
    <div class=gallery>{tiles or '<div class="panel empty">No headshots to show yet.</div>'}</div>
    <div id=toast></div>
    <script>
    function toast(m) {{ const t=document.getElementById('toast'); t.textContent=m; t.classList.add('show');
      clearTimeout(t._h); t._h=setTimeout(()=>t.classList.remove('show'), 1800); }}
    function setAppCount() {{
      document.getElementById('appn').textContent = document.querySelectorAll('.tile.approved').length;
    }}
    async function review(stem, decision) {{
      await fetch("/jobs/{jid}/review", {{method:'POST', headers:{{'Content-Type':'application/x-www-form-urlencoded'}},
        body:'stem='+encodeURIComponent(stem)+'&decision='+decision}});
      const t = document.getElementById('t-'+stem);
      t.classList.remove('approved','rejected'); t.classList.add(decision);
      setAppCount(); toast(decision==='approved'?'Kept '+stem:'Rejected '+stem);
    }}
    async function bulk(decision) {{
      await fetch("/jobs/{jid}/review_bulk", {{method:'POST', headers:{{'Content-Type':'application/x-www-form-urlencoded'}},
        body:'decision='+decision}});
      document.querySelectorAll('.tile[data-stem]').forEach(t => {{
        if (t.querySelector('.ctl')) {{ t.classList.remove('approved','rejected'); t.classList.add(decision); }}
      }});
      setAppCount(); toast(decision==='approved'?'Approved all':'Rejected all');
    }}
    // keyboard review: A keep, R reject, J/K move focus
    let cur = 0; const cards = [...document.querySelectorAll('.tile[data-stem] .ctl')].map(c=>c.closest('.tile'));
    function focusCard(i) {{ if(!cards.length) return; cur=(i+cards.length)%cards.length;
      cards.forEach((c,k)=>c.style.outline = k===cur?'2px solid var(--accent)':''); cards[cur].scrollIntoView({{block:'center',behavior:'smooth'}}); }}
    document.addEventListener('keydown', e => {{
      if (['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName) || !cards.length) return;
      const k = e.key.toLowerCase();
      if (k==='j') {{ focusCard(cur+1); e.preventDefault(); }}
      else if (k==='k') {{ focusCard(cur-1); e.preventDefault(); }}
      else if (k==='a') {{ review(cards[cur].dataset.stem,'approved'); e.preventDefault(); }}
      else if (k==='r') {{ review(cards[cur].dataset.stem,'rejected'); e.preventDefault(); }}
    }});
    if (cards.length) focusCard(0);
    </script>"""
    return _shell(f"{jid} · Review", "", topbar, content)


# ============================================================================
# Image-settings editor (client-side canvas: crop, adjust, bg-removal, logo, export)
# ============================================================================
# Plain template (NOT an f-string) so the JS braces need no escaping.
# __JID__ / __STEM__ are substituted per request.
_EDITOR_BODY = """
<div class=editor>
  <div class=canvas-wrap><canvas id=cv width=560 height=560></canvas></div>
  <div class=panel>
    <div class=ed-sec>
      <h3>Crop and frame</h3>
      <div class=chips id=aspects>
        <span class="chip on" data-r="1">1:1</span>
        <span class=chip data-r="0.8">4:5</span>
        <span class=chip data-r="0.75">3:4</span>
      </div>
      <div class=slider><label>Zoom</label><input id=zoom type=range min=100 max=300 value=100></div>
      <div class=ed-status>Drag the image to reposition.</div>
    </div>
    <div class=ed-sec>
      <h3>Colour and light</h3>
      <div class=slider><label>Brightness</label><input id=brightness type=range min=50 max=150 value=100></div>
      <div class=slider><label>Contrast</label><input id=contrast type=range min=50 max=150 value=100></div>
      <div class=slider><label>Saturation</label><input id=saturation type=range min=0 max=200 value=100></div>
      <div class=slider><label>Warmth</label><input id=warmth type=range min=-50 max=50 value=0></div>
      <button class="btn ghost sm" id=resetadj type=button>Reset colours</button>
    </div>
    <div class=ed-sec>
      <h3>Background</h3>
      <div class=chips>
        <button class="btn ghost sm" id=removebg type=button>Remove background</button>
        <button class="btn ghost sm" id=restorebg type=button>Restore</button>
      </div>
      <div class=slider><label>Fill color</label><input id=bgcolor type=color value="#ffffff"></div>
      <div class=ed-status id=bgstatus>Removes background and fills with the color (matches the corporate look).</div>
    </div>
    <div class=ed-sec>
      <h3>Add a logo</h3>
      <input id=logofile type=file accept="image/*">
      <div class=slider><label>Size</label><input id=logoscale type=range min=5 max=40 value=18></div>
      <div class=slider><label>Opacity</label><input id=logoop type=range min=20 max=100 value=100></div>
      <div class=ed-status>Upload a logo, then drag it on the image to place it.</div>
    </div>
    <div class=ed-sec>
      <h3>Export</h3>
      <div class=slider><label>Size</label>
        <select id=exsize><option value=1024>1K</option><option value=2048 selected>2K</option><option value=4096>4K</option></select></div>
      <div class=slider><label>Format</label>
        <select id=exfmt><option value=jpeg>JPG</option><option value=png>PNG</option></select></div>
      <button class=btn id=download type=button style="width:100%">Download this headshot</button>
    </div>
  </div>
</div>
<script>
(function(){
  var SRC = "/jobs/__JID__/image/output/__STEM__";
  var cv = document.getElementById('cv'), ctx = cv.getContext('2d');
  var img = new Image(); img.crossOrigin = 'anonymous';
  var fg = null, bgRemoved = false, logo = null;
  var logoPos = {x:0.70, y:0.05};
  var st = {r:1, zoom:1, panX:0, panY:0, brightness:100, contrast:100, saturation:100, warmth:0, bgcolor:'#ffffff', logoscale:0.18, logoop:1};

  function outDims(size){ return st.r>=1 ? {w:size, h:Math.round(size/st.r)} : {w:Math.round(size*st.r), h:size}; }
  function cropRect(sw, sh){
    var base = Math.min(sw, sh*st.r);
    var cw = base/st.zoom, ch = cw/st.r;
    var cx = sw/2 + st.panX, cy = sh/2 + st.panY;
    var sx = cx - cw/2, sy = cy - ch/2;
    sx = Math.max(0, Math.min(sw-cw, sx)); sy = Math.max(0, Math.min(sh-ch, sy));
    return {sx:sx, sy:sy, cw:cw, ch:ch};
  }
  function drawTo(c, size){
    var source = fg || img;
    var sw = source.naturalWidth||source.width, sh = source.naturalHeight||source.height;
    if(!sw||!sh){ return; }
    var cr = cropRect(sw, sh), dim = outDims(size);
    c.width = dim.w; c.height = dim.h;
    var g = c.getContext('2d');
    g.clearRect(0,0,dim.w,dim.h);
    if(bgRemoved){ g.fillStyle = st.bgcolor; g.fillRect(0,0,dim.w,dim.h); }
    g.filter = 'brightness('+st.brightness+'%) contrast('+st.contrast+'%) saturate('+st.saturation+'%)';
    g.drawImage(source, cr.sx, cr.sy, cr.cw, cr.ch, 0, 0, dim.w, dim.h);
    g.filter = 'none';
    if(st.warmth !== 0){
      g.globalCompositeOperation = 'soft-light';
      g.fillStyle = st.warmth>0 ? 'rgba(255,170,60,'+(st.warmth/100)+')' : 'rgba(60,140,255,'+(-st.warmth/100)+')';
      g.fillRect(0,0,dim.w,dim.h);
      g.globalCompositeOperation = 'source-over';
    }
    if(logo){
      var lw = dim.w*st.logoscale, lh = lw*(logo.height/logo.width);
      g.globalAlpha = st.logoop;
      g.drawImage(logo, logoPos.x*dim.w, logoPos.y*dim.h, lw, lh);
      g.globalAlpha = 1;
    }
  }
  function render(){ drawTo(cv, 560); }

  img.onload = render;
  img.onerror = function(){ document.getElementById('bgstatus').textContent = 'Could not load the source image.'; };
  img.src = SRC;

  function bind(id, key, div){ var el = document.getElementById(id);
    el.addEventListener('input', function(){ st[key] = div ? (+el.value/div) : (+el.value); render(); }); }
  bind('zoom','zoom',100); bind('brightness','brightness'); bind('contrast','contrast');
  bind('saturation','saturation'); bind('warmth','warmth'); bind('logoscale','logoscale',100); bind('logoop','logoop',100);
  document.getElementById('bgcolor').addEventListener('input', function(e){ st.bgcolor = e.target.value; render(); });
  document.getElementById('resetadj').onclick = function(){
    st.brightness=100; st.contrast=100; st.saturation=100; st.warmth=0;
    ['brightness','contrast','saturation'].forEach(function(i){ document.getElementById(i).value=100; });
    document.getElementById('warmth').value=0; render();
  };
  var aspects = document.getElementById('aspects');
  aspects.addEventListener('click', function(e){ if(!e.target.dataset.r) return;
    [].forEach.call(aspects.children, function(c){ c.classList.remove('on'); });
    e.target.classList.add('on'); st.r = +e.target.dataset.r; st.panX=0; st.panY=0; render(); });

  // drag to pan the image, or move the logo when the drag starts on it
  var drag = null;
  function ptr(e){ var r = cv.getBoundingClientRect(); var sc = cv.width/r.width;
    return {x:(e.clientX-r.left)*sc, y:(e.clientY-r.top)*sc}; }
  cv.addEventListener('pointerdown', function(e){
    cv.setPointerCapture(e.pointerId); var p = ptr(e);
    if(logo){
      var lw = cv.width*st.logoscale, lh = lw*(logo.height/logo.width);
      var lx = logoPos.x*cv.width, ly = logoPos.y*cv.height;
      if(p.x>=lx && p.x<=lx+lw && p.y>=ly && p.y<=ly+lh){ drag={type:'logo', p:p, lp:{x:logoPos.x,y:logoPos.y}}; return; }
    }
    drag = {type:'pan', p:p, pan:{x:st.panX, y:st.panY}};
  });
  cv.addEventListener('pointermove', function(e){ if(!drag) return; var p = ptr(e);
    if(drag.type==='logo'){ logoPos.x = drag.lp.x + (p.x-drag.p.x)/cv.width; logoPos.y = drag.lp.y + (p.y-drag.p.y)/cv.height; }
    else { var source = fg||img; var sw = source.naturalWidth||source.width, sh = source.naturalHeight||source.height;
      var cr = cropRect(sw, sh); var f = cr.cw/cv.width;
      st.panX = drag.pan.x - (p.x-drag.p.x)*f; st.panY = drag.pan.y - (p.y-drag.p.y)*f; }
    render();
  });
  cv.addEventListener('pointerup', function(){ drag = null; });

  document.getElementById('logofile').addEventListener('change', function(e){
    var f = e.target.files[0]; if(!f) return; var rd = new FileReader();
    rd.onload = function(){ var im = new Image(); im.onload = function(){ logo = im; render(); }; im.src = rd.result; }; rd.readAsDataURL(f);
  });

  document.getElementById('removebg').onclick = async function(){
    var s = document.getElementById('bgstatus'); s.textContent = 'Removing background - first run downloads a small model...';
    try {
      var mod = await import('https://cdn.jsdelivr.net/npm/@imgly/background-removal/+esm');
      var blob = await (await fetch(SRC)).blob();
      var out = await mod.removeBackground(blob);
      var im = new Image(); im.onload = function(){ fg = im; bgRemoved = true; s.textContent = 'Background removed. Adjust the fill color above.'; render(); };
      im.src = URL.createObjectURL(out);
    } catch(err){ s.textContent = 'Background removal needs internet access and could not load. ' + err; }
  };
  document.getElementById('restorebg').onclick = function(){ fg = null; bgRemoved = false;
    document.getElementById('bgstatus').textContent = 'Original background restored.'; render(); };

  document.getElementById('download').onclick = function(){
    var size = +document.getElementById('exsize').value, fmt = document.getElementById('exfmt').value;
    var off = document.createElement('canvas'); drawTo(off, size);
    off.toBlob(function(b){ var a = document.createElement('a'); a.href = URL.createObjectURL(b);
      a.download = '__STEM___edited.' + (fmt==='png'?'png':'jpg'); document.body.appendChild(a); a.click(); a.remove(); },
      fmt==='png'?'image/png':'image/jpeg', 0.95);
  };
})();
</script>
"""


@app.get("/jobs/{job_id}/edit/{stem}", response_class=HTMLResponse)
def editor_page(job_id: str, stem: str) -> HTMLResponse:
    if not _output_path(job_id, stem):
        return HTMLResponse(_shell("Not found", "", '<div class=tt>Not found</div>',
                                   '<div class="panel empty">We could not find that image. It may have been deleted.</div>'), 404)
    topbar = (f'<div class=tt>Fine-tune this headshot<small>Crop, adjust the colours, swap the background, or add a logo, then download. Job {escape(job_id)}, photo {escape(stem)}.</small></div>'
              f'<a class="btn ghost sm" href="/jobs/{escape(job_id)}">{ICON["back"]} Back to review</a>')
    body = _EDITOR_BODY.replace("__JID__", job_id).replace("__STEM__", stem)
    return HTMLResponse(_shell(f"Edit {stem}", "", topbar, body))


# Batch editor: configure once, apply to every successful headshot, download a ZIP.
_BATCH_BODY = """
<div class=editor>
  <div class=canvas-wrap><canvas id=cv width=560 height=560></canvas>
    <div class=ed-status style="position:absolute;bottom:10px" id=previewnote></div></div>
  <div class=panel>
    <div class=ed-sec>
      <h3>Preset</h3>
      <div class=slider><label>Saved</label>
        <select id=presetsel><option value="">- choose -</option></select></div>
      <div class=chips>
        <button class="btn ghost sm" id=savepreset type=button>Save current</button>
        <button class="btn ghost sm" id=delpreset type=button>Delete</button>
      </div>
    </div>
    <div class=ed-sec>
      <h3>Crop and frame</h3>
      <div class=chips id=aspects>
        <span class="chip on" data-r="1">1:1</span>
        <span class=chip data-r="0.8">4:5</span>
        <span class=chip data-r="0.75">3:4</span>
      </div>
      <div class=slider><label>Zoom</label><input id=zoom type=range min=100 max=300 value=100></div>
    </div>
    <div class=ed-sec>
      <h3>Colour and light</h3>
      <div class=slider><label>Brightness</label><input id=brightness type=range min=50 max=150 value=100></div>
      <div class=slider><label>Contrast</label><input id=contrast type=range min=50 max=150 value=100></div>
      <div class=slider><label>Saturation</label><input id=saturation type=range min=0 max=200 value=100></div>
      <div class=slider><label>Warmth</label><input id=warmth type=range min=-50 max=50 value=0></div>
    </div>
    <div class=ed-sec>
      <h3>Background</h3>
      <label class=switch><input type=checkbox id=batchbg><span class=track></span><span>Remove &amp; whiten all (slower)</span></label>
      <div class=slider style="margin-top:10px"><label>Fill color</label><input id=bgcolor type=color value="#ffffff"></div>
    </div>
    <div class=ed-sec>
      <h3>Add a logo</h3>
      <input id=logofile type=file accept="image/*">
      <div class=slider><label>Size</label><input id=logoscale type=range min=5 max=40 value=18></div>
      <div class=slider><label>Opacity</label><input id=logoop type=range min=20 max=100 value=100></div>
      <div class=ed-status>Drag the logo on the preview to place it.</div>
    </div>
    <div class=ed-sec>
      <h3>Export all (__COUNT__ images)</h3>
      <div class=slider><label>Size</label>
        <select id=exsize><option value=1024>1K</option><option value=2048 selected>2K</option><option value=4096>4K</option></select></div>
      <div class=slider><label>Format</label>
        <select id=exfmt><option value=jpeg>JPG</option><option value=png>PNG</option></select></div>
      <button class=btn id=applyall type=button style="width:100%">Apply to all &amp; download ZIP</button>
      <div class=ed-status id=batchstatus></div>
    </div>
  </div>
</div>
<script>
(function(){
  var JID = "__JID__";
  var STEMS = __STEMS__;
  var cv = document.getElementById('cv');
  var img = new Image(); img.crossOrigin='anonymous';
  var logo = null, logoPos = {x:0.70, y:0.05};
  var st = {r:1, zoom:1, panX:0, panY:0, brightness:100, contrast:100, saturation:100, warmth:0, bgcolor:'#ffffff', logoscale:0.18, logoop:1};

  function outDims(size){ return st.r>=1 ? {w:size, h:Math.round(size/st.r)} : {w:Math.round(size*st.r), h:size}; }
  function cropRect(sw, sh){
    var base = Math.min(sw, sh*st.r), cw = base/st.zoom, ch = cw/st.r;
    var cx = sw/2 + st.panX, cy = sh/2 + st.panY, sx = cx-cw/2, sy = cy-ch/2;
    sx = Math.max(0, Math.min(sw-cw, sx)); sy = Math.max(0, Math.min(sh-ch, sy));
    return {sx:sx, sy:sy, cw:cw, ch:ch};
  }
  function drawCore(c, size, source, whiteBg){
    var sw = source.naturalWidth||source.width, sh = source.naturalHeight||source.height; if(!sw) return;
    var cr = cropRect(sw, sh), dim = outDims(size); c.width = dim.w; c.height = dim.h;
    var g = c.getContext('2d'); g.clearRect(0,0,dim.w,dim.h);
    if(whiteBg){ g.fillStyle = st.bgcolor; g.fillRect(0,0,dim.w,dim.h); }
    g.filter = 'brightness('+st.brightness+'%) contrast('+st.contrast+'%) saturate('+st.saturation+'%)';
    g.drawImage(source, cr.sx, cr.sy, cr.cw, cr.ch, 0, 0, dim.w, dim.h); g.filter='none';
    if(st.warmth!==0){ g.globalCompositeOperation='soft-light';
      g.fillStyle = st.warmth>0?'rgba(255,170,60,'+(st.warmth/100)+')':'rgba(60,140,255,'+(-st.warmth/100)+')';
      g.fillRect(0,0,dim.w,dim.h); g.globalCompositeOperation='source-over'; }
    if(logo){ var lw=dim.w*st.logoscale, lh=lw*(logo.height/logo.width); g.globalAlpha=st.logoop;
      g.drawImage(logo, logoPos.x*dim.w, logoPos.y*dim.h, lw, lh); g.globalAlpha=1; }
  }
  function render(){ drawCore(cv, 560, img, document.getElementById('batchbg').checked); }
  function loadImg(src){ return new Promise(function(res,rej){ var im=new Image(); im.crossOrigin='anonymous';
    im.onload=function(){res(im);}; im.onerror=rej; im.src=src; }); }

  if(STEMS.length){ img.onload=render; img.src="/jobs/"+JID+"/image/output/"+STEMS[0];
    document.getElementById('previewnote').textContent='Preview: '+STEMS[0]; }

  function bind(id,key,div){ var el=document.getElementById(id);
    el.addEventListener('input', function(){ st[key]=div?(+el.value/div):(+el.value); render(); }); }
  bind('zoom','zoom',100); bind('brightness','brightness'); bind('contrast','contrast');
  bind('saturation','saturation'); bind('warmth','warmth'); bind('logoscale','logoscale',100); bind('logoop','logoop',100);
  document.getElementById('bgcolor').addEventListener('input', function(e){ st.bgcolor=e.target.value; render(); });
  document.getElementById('batchbg').addEventListener('change', render);
  var aspects=document.getElementById('aspects');
  aspects.addEventListener('click', function(e){ if(!e.target.dataset.r) return;
    [].forEach.call(aspects.children,function(c){c.classList.remove('on');}); e.target.classList.add('on');
    st.r=+e.target.dataset.r; st.panX=0; st.panY=0; render(); });
  var drag=null;
  function ptr(e){ var r=cv.getBoundingClientRect(), sc=cv.width/r.width; return {x:(e.clientX-r.left)*sc, y:(e.clientY-r.top)*sc}; }
  cv.addEventListener('pointerdown', function(e){ cv.setPointerCapture(e.pointerId); var p=ptr(e);
    if(logo){ var lw=cv.width*st.logoscale, lh=lw*(logo.height/logo.width), lx=logoPos.x*cv.width, ly=logoPos.y*cv.height;
      if(p.x>=lx&&p.x<=lx+lw&&p.y>=ly&&p.y<=ly+lh){ drag={type:'logo',p:p,lp:{x:logoPos.x,y:logoPos.y}}; return; } }
    drag={type:'pan',p:p,pan:{x:st.panX,y:st.panY}}; });
  cv.addEventListener('pointermove', function(e){ if(!drag) return; var p=ptr(e);
    if(drag.type==='logo'){ logoPos.x=drag.lp.x+(p.x-drag.p.x)/cv.width; logoPos.y=drag.lp.y+(p.y-drag.p.y)/cv.height; }
    else { var sw=img.naturalWidth, sh=img.naturalHeight, cr=cropRect(sw,sh), f=cr.cw/cv.width;
      st.panX=drag.pan.x-(p.x-drag.p.x)*f; st.panY=drag.pan.y-(p.y-drag.p.y)*f; } render(); });
  cv.addEventListener('pointerup', function(){ drag=null; });
  document.getElementById('logofile').addEventListener('change', function(e){ var f=e.target.files[0]; if(!f) return;
    var rd=new FileReader(); rd.onload=function(){ var im=new Image(); im.onload=function(){logo=im; render();}; im.src=rd.result; }; rd.readAsDataURL(f); });

  // presets (localStorage; logo not stored)
  function presets(){ try{ return JSON.parse(localStorage.getItem('ff_presets')||'{}'); }catch(e){ return {}; } }
  function refreshPresets(){ var sel=document.getElementById('presetsel'), p=presets();
    sel.innerHTML='<option value="">- choose -</option>'; Object.keys(p).forEach(function(k){ var o=document.createElement('option'); o.value=k; o.textContent=k; sel.appendChild(o); }); }
  document.getElementById('savepreset').onclick=function(){ var name=prompt('Preset name:'); if(!name) return;
    var p=presets(); p[name]=st; localStorage.setItem('ff_presets', JSON.stringify(p)); refreshPresets(); document.getElementById('presetsel').value=name; };
  document.getElementById('delpreset').onclick=function(){ var sel=document.getElementById('presetsel'); if(!sel.value) return;
    var p=presets(); delete p[sel.value]; localStorage.setItem('ff_presets', JSON.stringify(p)); refreshPresets(); };
  document.getElementById('presetsel').onchange=function(e){ var p=presets()[e.target.value]; if(!p) return;
    Object.assign(st, p);
    ['zoom','brightness','contrast','saturation','warmth'].forEach(function(k){ var el=document.getElementById(k); if(el) el.value = k==='zoom'?st.zoom*100:st[k]; });
    document.getElementById('logoscale').value=st.logoscale*100; document.getElementById('logoop').value=st.logoop*100;
    document.getElementById('bgcolor').value=st.bgcolor;
    [].forEach.call(aspects.children,function(c){ c.classList.toggle('on', +c.dataset.r===st.r); }); render(); };
  refreshPresets();

  function status(m){ document.getElementById('batchstatus').textContent=m; }
  document.getElementById('applyall').onclick=async function(){
    if(!STEMS.length){ status('Nothing to export.'); return; }
    var size=+document.getElementById('exsize').value, fmt=document.getElementById('exfmt').value, doBg=document.getElementById('batchbg').checked;
    status('Loading zip library...');
    var JSZip, bgMod=null;
    try { JSZip=(await import('https://cdn.jsdelivr.net/npm/jszip@3.10.1/+esm')).default; }
    catch(e){ status('Could not load zip library (needs internet).'); return; }
    if(doBg){ try{ bgMod=await import('https://cdn.jsdelivr.net/npm/@imgly/background-removal/+esm'); }
      catch(e){ status('Background removal unavailable; exporting without it.'); doBg=false; } }
    var zip=new JSZip(), off=document.createElement('canvas'), ext=(fmt==='png'?'png':'jpg'), mime=(fmt==='png'?'image/png':'image/jpeg');
    for(var i=0;i<STEMS.length;i++){ var stem=STEMS[i]; status('Processing '+(i+1)+'/'+STEMS.length+': '+stem);
      var src="/jobs/"+JID+"/image/output/"+stem, im, white=false;
      try { im=await loadImg(src); } catch(e){ continue; }
      if(doBg && bgMod){ try{ var b=await (await fetch(src)).blob(); var o=await bgMod.removeBackground(b); im=await loadImg(URL.createObjectURL(o)); white=true; }catch(e){} }
      else if(document.getElementById('batchbg').checked){ white=true; }
      drawCore(off, size, im, white);
      var blob=await new Promise(function(res){ off.toBlob(res, mime, 0.95); });
      zip.file(stem+'.'+ext, blob);
    }
    status('Zipping...'); var out=await zip.generateAsync({type:'blob'});
    var a=document.createElement('a'); a.href=URL.createObjectURL(out); a.download='facefoundry_'+JID+'_edited.zip';
    document.body.appendChild(a); a.click(); a.remove(); status('Done - '+STEMS.length+' images exported.');
  };
})();
</script>
"""


@app.get("/jobs/{job_id}/edit", response_class=HTMLResponse)
def batch_editor_page(job_id: str) -> HTMLResponse:
    j = db.get_job(job_id)
    res = _results(job_id) or {}
    stems = [r["stem"] for r in res.get("results", [])
             if r.get("status") == "ok" and _output_path(job_id, r["stem"])]
    if not j or not stems:
        return HTMLResponse(_shell("Batch edit", "", '<div class=tt>Batch edit</div>',
                                   '<div class="panel empty">There are no finished headshots to edit yet. Run a job first, then come back.</div>'), 404)
    topbar = (f'<div class=tt>Edit every headshot at once<small>Set it up once below, then apply the same crop, colours, background, and logo to all {len(stems)} headshots and download them as a zip.</small></div>'
              f'<a class="btn ghost sm" href="/jobs/{escape(job_id)}">{ICON["back"]} Back to review</a>')
    body = (_BATCH_BODY.replace("__JID__", job_id)
            .replace("__STEMS__", json.dumps(stems))
            .replace("__COUNT__", str(len(stems))))
    return HTMLResponse(_shell(f"Batch edit {job_id}", "", topbar, body))


# ============================================================================
# Serve images / files
# ============================================================================
@app.get("/jobs/{job_id}/image/{kind}/{stem}")
def job_image(job_id: str, kind: str, stem: str):
    p = _input_path(job_id, stem) if kind == "input" else _output_path(job_id, stem)
    if p and p.is_file():
        return FileResponse(p)
    return JSONResponse({"error": "not found"}, status_code=404)


@app.get("/jobs/{job_id}/file/{name}")
def job_file(job_id: str, name: str):
    p = JOBS_DIR / job_id / "output" / Path(name).name
    if p.is_file():
        return FileResponse(p, media_type="text/plain")
    return JSONResponse({"error": "not found"}, status_code=404)


# ============================================================================
# Review, download, re-roll
# ============================================================================
@app.post("/jobs/{job_id}/delete")
def delete_job_route(job_id: str) -> RedirectResponse:
    """Delete a job: its DB rows and its on-disk folder (uploads/input/output).
    Also asks Kaggle to cancel the kernel if the job is still running so we
    don't leave a phantom render burning the free GPU quota."""
    _safe_job_id(job_id)
    j = db.get_job(job_id)
    if j and j["status"] in ("running", "queued"):
        try:
            cancel_kernel(job_id)
        except Exception:
            pass  # best-effort; local delete must still proceed
    db.delete_job(job_id)
    job_dir = JOBS_DIR / job_id
    if job_dir.is_dir():
        import shutil
        shutil.rmtree(job_dir, ignore_errors=True)
    return RedirectResponse("/", status_code=303)


@app.post("/jobs/{job_id}/cancel")
def cancel_job_route(job_id: str) -> JSONResponse:
    """Ask Kaggle to stop the kernel and mark the job cancelled. Local files stay."""
    _safe_job_id(job_id)
    j = db.get_job(job_id)
    if not j:
        return JSONResponse({"error": "not found"}, status_code=404)
    if j["status"] not in ("running", "queued"):
        return JSONResponse({"ok": True, "already": j["status"]})
    ok = False
    try:
        ok = cancel_kernel(job_id)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    db.update_job(job_id, status="failed", stage="cancelled",
                  message="Cancelled by user", error="cancelled")
    return JSONResponse({"ok": True, "kaggle_ok": ok})


@app.post("/jobs/{job_id}/review")
def review(job_id: str, stem: str = Form(...), decision: str = Form(...)) -> JSONResponse:
    if decision not in ("approved", "rejected"):
        return JSONResponse({"error": "bad decision"}, status_code=400)
    db.set_review(job_id, stem, decision)
    return JSONResponse({"ok": True})


@app.get("/jobs/{job_id}/download")
def download(job_id: str, scope: str = "approved"):
    """scope=approved (default) zips approved shots (falls back to all successful
    when nothing is approved); scope=all zips every successful headshot."""
    res = _results(job_id) or {}
    all_ok = {r["stem"] for r in res.get("results", []) if r.get("status") == "ok"}
    if scope == "all":
        stems = all_ok
    else:
        approved = {s for s, d in db.get_reviews(job_id).items() if d == "approved"}
        stems = approved or all_ok

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for stem in sorted(stems):
            p = _output_path(job_id, stem)
            if p and p.is_file():
                z.write(p, arcname=f"{stem}.jpg")
    buf.seek(0)
    fname = f"facefoundry_{job_id}_{scope}.zip"
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="{fname}"'})


@app.get("/jobs/{job_id}/download/{stem}")
def download_one(job_id: str, stem: str):
    """Download a single generated headshot as a file attachment."""
    p = _output_path(job_id, stem)
    if p and p.is_file():
        return FileResponse(p, media_type="image/jpeg", filename=f"{stem}.jpg")
    return JSONResponse({"error": "not found"}, status_code=404)


@app.post("/jobs/{job_id}/review_bulk")
def review_bulk(job_id: str, decision: str = Form(...)) -> JSONResponse:
    """Approve or reject every successful image in one call."""
    if decision not in ("approved", "rejected"):
        return JSONResponse({"error": "bad decision"}, status_code=400)
    res = _results(job_id) or {}
    for r in res.get("results", []):
        if r.get("status") == "ok":
            db.set_review(job_id, r["stem"], decision)
    return JSONResponse({"ok": True})


@app.post("/jobs/{job_id}/reroll")
def reroll(job_id: str) -> RedirectResponse:
    """Re-run just the rejected/failed originals as a NEW job with a bumped seed."""
    j = db.get_job(job_id)
    res = _results(job_id) or {}
    reviews = db.get_reviews(job_id)
    if not j:
        return RedirectResponse("/", status_code=303)

    redo = [r["stem"] for r in res.get("results", [])
            if r.get("status") != "ok" or reviews.get(r["stem"]) == "rejected"]
    if not redo:
        return RedirectResponse(f"/jobs/{job_id}", status_code=303)

    new_id = slugify_job_id(f"{job_id}-r{int(time.time()) % 10000}")
    up_dir = JOBS_DIR / new_id / "uploads"
    up_dir.mkdir(parents=True, exist_ok=True)
    for stem in redo:
        p = _input_path(job_id, stem)
        if p:
            (up_dir / p.name).write_bytes(p.read_bytes())

    cfg = dict(json.loads(j["config"]))
    cfg["job_id"] = new_id
    cfg["seed_base"] = int(cfg.get("seed_base", 42)) + 1000
    cfg["limit"] = None
    db.create_job(new_id, j["style"], cfg)
    runner.start_job(new_id, up_dir, cfg)
    return RedirectResponse(f"/jobs/{new_id}", status_code=303)


@app.post("/jobs/{job_id}/rerun")
def rerun(job_id: str, style: str = Form(...)) -> RedirectResponse:
    """Re-run ALL the same source faces with a different style preset."""
    j = db.get_job(job_id)
    if not j:
        return RedirectResponse("/", status_code=303)
    if style not in {v for v, _, _ in STYLES}:
        style = j["style"]

    src_in = JOBS_DIR / job_id / "input"
    originals = [p for p in src_in.iterdir() if p.suffix.lower() in IMAGE_EXTS] if src_in.is_dir() else []
    if not originals:
        return RedirectResponse(f"/jobs/{job_id}", status_code=303)

    new_id = slugify_job_id(f"{job_id}-{style[:6]}{int(time.time()) % 1000}")
    up_dir = JOBS_DIR / new_id / "uploads"
    up_dir.mkdir(parents=True, exist_ok=True)
    for p in originals:
        (up_dir / p.name).write_bytes(p.read_bytes())

    cfg = dict(json.loads(j["config"]))
    cfg["job_id"] = new_id
    cfg["style_preset"] = style
    cfg["limit"] = None
    db.create_job(new_id, style, cfg)
    runner.start_job(new_id, up_dir, cfg)
    return RedirectResponse(f"/jobs/{new_id}", status_code=303)
