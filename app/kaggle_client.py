"""
Orchestrator - runs on YOUR PC. The core "one-click" engine.

Given a local folder of source images and a style, it:
  1. packages the images + a job.json into a dataset folder,
  2. pushes them as a PRIVATE Kaggle dataset,
  3. launches the GPU worker kernel (headshot_worker.py) with that dataset attached,
  4. polls until the kernel completes,
  5. downloads the finished headshots + results.json back to a local job folder.

All Kaggle auth quirks are handled here (see notes in auth_env / owner_slug).

Two ways to drive it:
  * CLI (headless test of the whole loop):
        python app/kaggle_client.py --images path/to/pics --style corporate --limit 3
  * Programmatic (used by the web UI):
        run_job(images_dir, cfg, job_id, on_event=callback)
    Emits (stage, message, extra) events for progress and raises JobError on
    failure instead of exiting the process.

Requires: `pip install kaggle`, and kaggle.json at ~/.kaggle/kaggle.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

# Windows consoles default to cp1252, which raises UnicodeEncodeError on any
# non-ASCII we print (crashed a real run: "'charmap' codec can't encode").
# Force UTF-8 with a safe fallback so logging can never kill a job.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO = Path(__file__).resolve().parent.parent
WORKER_SCRIPT = REPO / "worker" / "headshot_worker.py"
JOBS_DIR = REPO / "jobs"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

POLL_SECONDS = 20
TIMEOUT_MINUTES = 720  # 12h - a full batch can take ~8h

# Per-call subprocess timeouts. Kaggle CLI hangs on network flakes without one;
# these caps are generous but bounded so pollers can't deadlock forever.
CLI_TIMEOUT_SECONDS = 180        # single CLI call (status/list) - 3 min
CLI_PUSH_TIMEOUT_SECONDS = 900   # dataset/kernel push - 15 min (uploads)
CLI_RETRIES = 3                  # network flakes get 3 tries with backoff

# on_event(stage: str, message: str, extra: dict) -> None
EventFn = Optional[Callable[[str, str, dict], None]]


class JobError(RuntimeError):
    """Raised on any orchestration failure (instead of sys.exit) so callers
    like the web server can catch it and record the job as failed."""


def _emit(on_event: EventFn, stage: str, message: str, **extra) -> None:
    print(f"[{stage}] {message}")
    if on_event:
        try:
            on_event(stage, message, extra)
        except Exception:
            pass  # a broken progress callback must never kill the job


# ----------------------------------------------------------------------------
# Auth (see memory: KGAT tokens need KAGGLE_API_TOKEN; owner slug is normalized)
# ----------------------------------------------------------------------------
def _kaggle_json() -> Path:
    p = Path.home() / ".kaggle" / "kaggle.json"
    if not p.is_file():
        # On a hosted server there's no local kaggle.json - materialize it from
        # KAGGLE_USERNAME / KAGGLE_KEY env vars (set as secrets on the host).
        user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
        if user and key:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"username": user, "key": key}))
            try:
                p.chmod(0o600)
            except Exception:
                pass
        else:
            raise JobError(f"{p} not found and KAGGLE_USERNAME/KAGGLE_KEY unset. See SETUP.md.")
    return p


def credentials() -> tuple[str, str]:
    d = json.loads(_kaggle_json().read_text())
    return d["username"], d["key"]


def auth_env(key: str) -> dict:
    env = dict(os.environ)
    # A new-style KGAT_* token MUST be presented via KAGGLE_API_TOKEN. On a host
    # we also set KAGGLE_USERNAME / KAGGLE_KEY (only to materialize kaggle.json).
    # But the Kaggle CLI reads KAGGLE_KEY as a *legacy* API key and tries legacy
    # auth with it - which a KGAT token fails ("Authentication required to call
    # the Kaggle API"). So remove those env vars here; the CLI then authenticates
    # via kaggle.json + KAGGLE_API_TOKEN (the path proven to work locally).
    env.pop("KAGGLE_KEY", None)
    env.pop("KAGGLE_USERNAME", None)
    env["KAGGLE_API_TOKEN"] = key
    return env


def kaggle(args: list[str], env: dict, *, timeout: int | None = None,
           retries: int = CLI_RETRIES) -> subprocess.CompletedProcess:
    """Run the Kaggle CLI with a subprocess timeout and simple retry.

    Kaggle's CLI hangs occasionally on transient network flakes; without a
    timeout the poller would deadlock. Push operations get the longer cap.
    Retries only kick in for a timeout or a *non-zero* exit that looks like a
    transient network error - a successful call is always returned as-is.
    """
    is_push = bool(args) and args[0] in ("datasets", "kernels") and "push" in args
    if timeout is None:
        timeout = CLI_PUSH_TIMEOUT_SECONDS if is_push else CLI_TIMEOUT_SECONDS
    cmd = [sys.executable, "-m", "kaggle", *args]
    last: subprocess.CompletedProcess | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                               timeout=timeout)
        except subprocess.TimeoutExpired as e:
            print(f"[kaggle] timeout after {timeout}s on `{' '.join(args)}` (attempt {attempt})")
            last = subprocess.CompletedProcess(cmd, returncode=124, stdout=e.stdout or "",
                                               stderr=(e.stderr or "") + "\n[timeout]")
            if attempt < retries:
                time.sleep(2 * attempt)
                continue
            return last
        last = r
        if r.returncode == 0:
            return r
        blob = (r.stdout + r.stderr).lower()
        transient = any(k in blob for k in (
            "timed out", "timeout", "temporary failure", "connection reset",
            "connection aborted", "read operation timed out", "429",
            "503", "502", "504", "gateway", "networkerror",
        ))
        if not transient or attempt >= retries:
            return r
        print(f"[kaggle] transient error on attempt {attempt}, retrying: {blob[:120]}")
        time.sleep(2 * attempt)
    return last  # type: ignore[return-value]


def owner_slug(username: str, env: dict) -> str:
    """Resolve the owner slug for the AUTHENTICATED account only.

    Kaggle normalizes the slug (Ramana-7981 -> ramana7981). We derive it from the
    credentials in kaggle.json and NEVER adopt a foreign owner: if the account's
    kernel list happens to contain a kernel owned by someone else (e.g. a shared
    or collaborated kernel), we ignore it. This guarantees every dataset/kernel is
    created under YOUR account and blocks the tool from ever running as another
    user's slug.
    """
    expected = username.lower().replace("-", "").replace("_", "")
    listing = kaggle(["kernels", "list", "--mine", "--csv"], env)
    for line in listing.stdout.splitlines()[1:]:
        ref = line.split(",", 1)[0].strip()
        if "/" in ref:
            owner = ref.split("/", 1)[0]
            if owner.lower() == expected:      # only accept OUR own, confirmed
                return owner
    return expected                             # fall back to our normalized slug


# ----------------------------------------------------------------------------
# Step 1: package + push the image dataset
# ----------------------------------------------------------------------------
def build_job_folder(job_id: str, images_dir: Path, cfg: dict, on_event: EventFn = None) -> Path:
    job_input = JOBS_DIR / job_id / "input"
    if job_input.exists():
        shutil.rmtree(job_input)
    job_input.mkdir(parents=True)

    n = 0
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() in IMAGE_EXTS and p.is_file():
            shutil.copy2(p, job_input / p.name)
            n += 1
    if n == 0:
        raise JobError(f"no images found in {images_dir}")
    (job_input / "job.json").write_text(json.dumps(cfg, indent=2))
    _emit(on_event, "pack", f"{n} images + job.json packaged", count=n)
    return job_input


def push_dataset(job_id: str, job_input: Path, owner: str, env: dict, on_event: EventFn = None) -> str:
    slug = f"headshot-input-{job_id}"
    dataset_ref = f"{owner}/{slug}"
    meta = {
        "title": f"Headshot input {job_id}",
        "id": dataset_ref,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (job_input / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

    _emit(on_event, "dataset", f"creating private dataset {dataset_ref}")
    r = kaggle(["datasets", "create", "-p", str(job_input), "--dir-mode", "zip"], env)
    out = (r.stdout + r.stderr).strip()
    print("   " + out.replace("\n", "\n   "))
    if "already exists" in out.lower():
        _emit(on_event, "dataset", "exists - pushing a new version")
        r = kaggle(["datasets", "version", "-p", str(job_input),
                    "-m", f"job {job_id}", "--dir-mode", "zip"], env)
        print("   " + (r.stdout + r.stderr).strip())
    return dataset_ref


def wait_for_dataset(dataset_ref: str, env: dict, on_event: EventFn = None, tries: int = 30):
    """Datasets take a moment to finish processing before a kernel can attach."""
    _emit(on_event, "dataset", "waiting for dataset to finish processing")
    for _ in range(tries):
        r = kaggle(["datasets", "status", dataset_ref], env)
        # Strip the dataset ref before matching so a job named e.g. "error-fix"
        # (ref headshot-input-error-fix) can't trigger a false "error"/"ready".
        s = (r.stdout + r.stderr).strip().lower().replace(dataset_ref.lower(), "")
        if "ready" in s:
            _emit(on_event, "dataset", "dataset ready")
            return
        if "error" in s:
            raise JobError(f"dataset processing failed: {(r.stdout + r.stderr).strip()}")
        time.sleep(10)
    _emit(on_event, "dataset", "status not confirmed 'ready' - proceeding anyway")


# ----------------------------------------------------------------------------
# Step 2: launch the worker kernel
# ----------------------------------------------------------------------------
def cache_dataset_ref(owner: str, env: dict) -> str | None:
    """Return the model-cache dataset ref if it exists on the account, else None.
    Attaching it lets the worker skip the ~10-15 min model download."""
    ref = f"{owner}/facefoundry-models"
    r = kaggle(["datasets", "status", ref], env)
    out = (r.stdout + r.stderr).lower()
    return ref if "ready" in out else None


def launch_kernel(job_id: str, dataset_ref: str, owner: str, env: dict, on_event: EventFn = None) -> str:
    push_dir = JOBS_DIR / job_id / "kernel"
    if push_dir.exists():
        shutil.rmtree(push_dir)
    push_dir.mkdir(parents=True)
    shutil.copy2(WORKER_SCRIPT, push_dir / "headshot_worker.py")

    slug = f"headshot-run-{job_id}"
    kernel_ref = f"{owner}/{slug}"
    sources = [dataset_ref]
    cache_ref = cache_dataset_ref(owner, env)
    if cache_ref:
        sources.append(cache_ref)
        _emit(on_event, "kernel", "attaching model cache (faster setup)")
    meta = {
        "id": kernel_ref,
        "title": f"Headshot run {job_id}",
        "code_file": "headshot_worker.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": sources,
        "competition_sources": [],
        "kernel_sources": [],
    }
    (push_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

    _emit(on_event, "kernel", f"launching GPU worker {kernel_ref}", kernel_ref=kernel_ref)
    r = kaggle(["kernels", "push", "-p", str(push_dir)], env)
    print("   " + (r.stdout + r.stderr).strip())
    if r.returncode != 0:
        raise JobError(f"kernel push failed: {(r.stdout + r.stderr).strip()}\n\n{_auth_diag(env)}")
    return kernel_ref


def _auth_diag(env: dict) -> str:
    """Self-diagnosing footer for auth failures - tells us exactly what the live
    environment looks like (kaggle version, kaggle.json state, token presence)
    without ever revealing the secret key."""
    try:
        ver = kaggle(["--version"], env, retries=1).stdout.strip().splitlines()[0]
    except Exception as e:
        ver = f"(could not run kaggle --version: {e})"
    kj = Path.home() / ".kaggle" / "kaggle.json"
    info = f"kaggle.json exists: {kj.is_file()}"
    if kj.is_file():
        try:
            d = json.loads(kj.read_text())
            info += f" | user={d.get('username')!r} key_len={len(d.get('key',''))} key_prefix={d.get('key','')[:5]!r}"
        except Exception as e:
            info += f" | (unreadable: {e})"
    return ("[diagnostics] " + ver
            + f" | KAGGLE_API_TOKEN set: {'KAGGLE_API_TOKEN' in env}"
            + f" | legacy KAGGLE_KEY in CLI env: {'KAGGLE_KEY' in env} (should be False)"
            + f" | HOME={env.get('HOME')} | " + info
            + "\nIf 'kaggle' version is NOT 2.2.4, do a CLEAN rebuild on the host "
              "(Render: Manual Deploy -> Clear build cache & deploy).")


def _parse_kernel_state(text: str) -> str:
    """Extract the terminal state from a `kernels status` line, e.g.
    '... has status "KernelWorkerStatus.ERROR"' -> 'error'. Kaggle statuses:
    QUEUED, RUNNING, COMPLETE, ERROR, CANCEL_REQUESTED, CANCEL_ACKNOWLEDGED.

    Inspect only the enum value AFTER 'kernelworkerstatus.' so neither the word
    'worker' in the class name nor a status word inside the job id/ref can be
    mistaken for the state. (Previous bug: `"error" in low and "worker" not in
    low` was always False -> ERROR was never detected and jobs hung as running.)"""
    low = text.lower()
    tail = low.split("kernelworkerstatus.")[-1] if "kernelworkerstatus." in low else low
    if tail.startswith("complete"):
        return "complete"
    if tail.startswith("error"):
        return "error"
    if tail.startswith("cancel"):
        return "cancelled"
    if tail.startswith("running"):
        return "running"
    if tail.startswith("queued"):
        return "queued"
    return "unknown"


def poll_kernel(kernel_ref: str, env: dict, on_event: EventFn = None):
    _emit(on_event, "running", "Rendering on GPU")
    deadline = time.monotonic() + TIMEOUT_MINUTES * 60
    last = None
    while time.monotonic() < deadline:
        r = kaggle(["kernels", "status", kernel_ref], env)
        text = (r.stdout + r.stderr).strip()
        state = _parse_kernel_state(text)
        if state != last:
            friendly = {"queued": "Queued on Kaggle", "running": "Rendering on GPU",
                        "complete": "Finished - fetching results"}.get(state, text)
            _emit(on_event, "running", friendly)
            last = state
        if state == "complete":
            return True
        if state in ("error", "cancelled"):
            raise JobError(f"kernel {state}: {text}")
        time.sleep(POLL_SECONDS)
    raise JobError("timed out waiting for kernel")


def cancel_kernel(job_id: str) -> bool:
    """Ask Kaggle to cancel the kernel for this job. Returns True on success.
    Uses `kaggle kernels output --quiet` guard because there's no dedicated
    cancel in the CLI; instead we push a no-op version that supersedes it.
    Best-effort: callers should treat False / raise as "kernel may keep running".
    """
    try:
        username, key = credentials()
    except JobError:
        return False
    env = auth_env(key)
    owner = owner_slug(username, env)
    kernel_ref = f"{owner}/headshot-run-{job_id}"
    # Kaggle exposes no explicit cancel via CLI; overwrite the kernel with a
    # trivial script so a new schedule supersedes the running one. This is the
    # documented workaround used by community tooling.
    push = JOBS_DIR / job_id / "_cancel"
    if push.exists():
        shutil.rmtree(push, ignore_errors=True)
    push.mkdir(parents=True, exist_ok=True)
    (push / "cancel.py").write_text("print('cancelled by facefoundry control panel')\n")
    (push / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel_ref, "title": f"Headshot run {job_id} (cancelled)",
        "code_file": "cancel.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": False, "enable_internet": False,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
    }, indent=2))
    r = kaggle(["kernels", "push", "-p", str(push)], env, retries=1)
    ok = r.returncode == 0
    shutil.rmtree(push, ignore_errors=True)
    return ok


def fetch_output(job_id: str, kernel_ref: str, env: dict, on_event: EventFn = None) -> Path:
    out_dir = JOBS_DIR / job_id / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    _emit(on_event, "download", f"downloading results -> {out_dir}")
    r = kaggle(["kernels", "output", kernel_ref, "-p", str(out_dir)], env)
    print("   " + (r.stdout + r.stderr).strip())
    return out_dir


# ----------------------------------------------------------------------------
# Orchestrate
# ----------------------------------------------------------------------------
def run_job(images_dir: Path, cfg: dict, job_id: str, on_event: EventFn = None) -> dict:
    """Run the full loop. Returns the results summary dict. Raises JobError on
    failure. Emits (stage, message, extra) progress via on_event."""
    username, key = credentials()
    env = auth_env(key)
    owner = owner_slug(username, env)
    _emit(on_event, "auth", f"authenticated as {username} (owner_slug={owner})")

    job_input = build_job_folder(job_id, images_dir, cfg, on_event)
    dataset_ref = push_dataset(job_id, job_input, owner, env, on_event)
    wait_for_dataset(dataset_ref, env, on_event)
    kernel_ref = launch_kernel(job_id, dataset_ref, owner, env, on_event)
    poll_kernel(kernel_ref, env, on_event)
    out_dir = fetch_output(job_id, kernel_ref, env, on_event)
    return _finalize(out_dir, on_event)


def resume_job(job_id: str, on_event: EventFn = None) -> dict:
    """Reconnect to an already-launched kernel (e.g. after the control panel
    restarted) and keep tracking it to completion - no re-upload, no re-launch.
    Raises JobError if the kernel errored or nothing came back."""
    username, key = credentials()
    env = auth_env(key)
    owner = owner_slug(username, env)
    kernel_ref = f"{owner}/headshot-run-{job_id}"
    _emit(on_event, "running", f"reconnecting to {kernel_ref}")
    poll_kernel(kernel_ref, env, on_event)
    out_dir = fetch_output(job_id, kernel_ref, env, on_event)
    return _finalize(out_dir, on_event)


def _finalize(out_dir: Path, on_event: EventFn = None) -> dict:
    results = out_dir / "results.json"
    if results.is_file():
        summary = json.loads(results.read_text())
        if summary.get("fatal"):
            raise JobError(f"worker crashed: {summary.get('error')}\n{summary.get('traceback','')}")
        _emit(on_event, "done",
              f"job done - ok={summary.get('ok')} failed={summary.get('failed')}",
              ok=summary.get("ok"), failed=summary.get("failed"), total=summary.get("total"))
        return summary
    raise JobError(
        f"kernel finished but no results.json in {out_dir}. "
        f"Check run.log there for the worker error.")


# ----------------------------------------------------------------------------
# One-time model-cache builder (Phase 4 speed): downloads all models on Kaggle
# and publishes them as the private `facefoundry-models` dataset, which the
# worker then loads from to skip the ~10-15 min download on every job.
# ----------------------------------------------------------------------------
_BUILDER_SCRIPT = r'''
# FaceFoundry model-cache builder - runs once on Kaggle to stage all models.
import os, subprocess
from pathlib import Path
OUT = Path("/kaggle/working/facefoundry-models"); OUT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(OUT / "huggingface")
def sh(c): print("$", c, flush=True); subprocess.run(c, shell=True, check=False)
sh("pip install -q 'numpy<2' huggingface_hub==0.22.2")
from huggingface_hub import snapshot_download, hf_hub_download
# SDXL base (the big ~10GB one) into the HF cache layout
snapshot_download("stabilityai/stable-diffusion-xl-base-1.0",
                  allow_patterns=["*.json","*.txt","*.safetensors","*.model"])
# InstantID controlnet + ip-adapter into checkpoints/
ck = OUT / "checkpoints"; ck.mkdir(parents=True, exist_ok=True)
hf_hub_download("InstantX/InstantID", "ControlNetModel/config.json", local_dir=str(ck))
hf_hub_download("InstantX/InstantID", "ControlNetModel/diffusion_pytorch_model.safetensors", local_dir=str(ck))
hf_hub_download("InstantX/InstantID", "ip-adapter.bin", local_dir=str(ck))
# antelopev2 face models
ant = OUT / "antelopev2"; ant.mkdir(parents=True, exist_ok=True)
sh("wget -q -O /kaggle/tmp/antelopev2.zip https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip")
sh(f"unzip -qo /kaggle/tmp/antelopev2.zip -d /kaggle/tmp/ant && cp /kaggle/tmp/ant/antelopev2/*.onnx {ant}/")
print("[builder] done ->", OUT, flush=True)
'''


def build_cache() -> None:
    """Run the builder kernel and publish its output as the facefoundry-models
    dataset. One-time; afterwards every job attaches it automatically."""
    username, key = credentials()
    env = auth_env(key)
    owner = owner_slug(username, env)
    print(f"[cache] building model cache for {owner} (this runs a one-off Kaggle kernel)...")

    push = JOBS_DIR / "_cache_builder"
    if push.exists():
        shutil.rmtree(push)
    push.mkdir(parents=True)
    (push / "build_models.py").write_text(_BUILDER_SCRIPT)
    kref = f"{owner}/facefoundry-cache-builder"
    (push / "kernel-metadata.json").write_text(json.dumps({
        "id": kref, "title": "FaceFoundry cache builder", "code_file": "build_models.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": False, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
    }, indent=2))

    r = kaggle(["kernels", "push", "-p", str(push)], env)
    print("   " + (r.stdout + r.stderr).strip())
    if r.returncode != 0:
        raise JobError("cache builder push failed")
    poll_kernel(kref, env)

    out = JOBS_DIR / "_cache_builder" / "output"
    out.mkdir(parents=True, exist_ok=True)
    kaggle(["kernels", "output", kref, "-p", str(out)], env)
    models = out / "facefoundry-models"
    if not models.is_dir():
        raise JobError(f"builder produced no models folder in {out}")

    meta = {"title": "FaceFoundry models", "id": f"{owner}/facefoundry-models",
            "licenses": [{"name": "CC0-1.0"}]}
    (models / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    r = kaggle(["datasets", "create", "-p", str(models), "--dir-mode", "zip"], env)
    out_txt = (r.stdout + r.stderr).strip()
    print("   " + out_txt)
    if "already exists" in out_txt.lower():
        kaggle(["datasets", "version", "-p", str(models), "-m", "refresh", "--dir-mode", "zip"], env)
    print(f"[cache] done - future jobs will auto-attach {owner}/facefoundry-models")


def make_cfg(args) -> dict:
    return {
        "job_id": args.job_id,
        "input_mode": "images",
        "style_preset": args.style,
        "limit": args.limit,
        "img_size": args.img_size,
        "output_size": args.output_size,
        "custom_prompt": args.custom_prompt,
        "background": args.background,
        "face_enhance": args.face_enhance,
        "white_background": args.white_background,
        "num_steps": args.steps,
        "guidance": args.guidance,
        "identity_scale": args.identity,
        "adapter_scale": args.adapter,
        "seed_base": args.seed,
    }


def slugify_job_id(job_id: str | None) -> str:
    if not job_id:
        job_id = "j" + str(int(time.time()))
    return "".join(c if c.isalnum() else "-" for c in job_id).strip("-").lower()


def main():
    ap = argparse.ArgumentParser(description="FaceFoundry orchestrator")
    ap.add_argument("--images", help="folder of source images")
    ap.add_argument("--build-cache", dest="build_cache", action="store_true",
                    help="one-time: build the facefoundry-models dataset to speed up future jobs")
    ap.add_argument("--style", default="corporate",
                    choices=["corporate", "modern_tech", "warm_friendly", "formal_executive",
                             "linkedin_classic", "startup_casual", "healthcare", "academic"])
    ap.add_argument("--limit", type=int, default=None, help="cap number of images (test with a few)")
    ap.add_argument("--job-id", dest="job_id", default=None, help="job id (default: timestamp-ish)")
    ap.add_argument("--img-size", dest="img_size", type=int, default=1024)
    ap.add_argument("--output-size", dest="output_size", type=int, default=1024,
                    help="final saved resolution; upscaled from img_size (e.g. 2048, 4096)")
    ap.add_argument("--custom-prompt", dest="custom_prompt", default="")
    ap.add_argument("--background", default="")
    ap.add_argument("--face-enhance", dest="face_enhance", action="store_true")
    ap.add_argument("--white-background", dest="white_background", action="store_true")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--identity", type=float, default=0.8)
    ap.add_argument("--adapter", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.build_cache:
        try:
            build_cache()
        except JobError as e:
            sys.exit(f"ERROR: {e}")
        return

    if not args.images:
        sys.exit("ERROR: --images is required (or use --build-cache)")
    args.job_id = slugify_job_id(args.job_id)

    images_dir = Path(args.images).expanduser().resolve()
    if not images_dir.is_dir():
        sys.exit(f"ERROR: not a folder: {images_dir}")

    try:
        run_job(images_dir, make_cfg(args), args.job_id)
    except JobError as e:
        sys.exit(f"ERROR: {e}")


if __name__ == "__main__":
    main()
