"""
Background job runner. Spawns the orchestrator in a thread and streams its
progress events into the SQLite store so the web UI can poll status.
"""

from __future__ import annotations

import os
import threading
import traceback
from pathlib import Path

from . import api_engine, db, reporting
from .kaggle_client import JobError, resume_job, run_job

# job_id -> Thread, so we can tell if a job is still actively running.
# Guarded by _threads_lock because HTTP handlers and background threads both
# read/write this dict - an unlocked dict race can leak zombie references.
_threads: dict[str, threading.Thread] = {}
_threads_lock = threading.Lock()


def _max_concurrent() -> int:
    """Concurrent GPU jobs cap (spec §17). Configurable via MAX_CONCURRENT_JOBS
    so it can be tuned to the actual GPU/Kaggle capacity. Default 2."""
    try:
        return max(1, int(os.getenv("MAX_CONCURRENT_JOBS", "2")))
    except ValueError:
        return 2


# A fresh job acquires a slot before it does GPU work; extra jobs stay 'queued'
# in the DB until a slot frees. Resume (reconnect) is NOT gated - it's attaching
# to an already-running kernel, not starting new GPU work.
_JOB_SLOTS = threading.Semaphore(_max_concurrent())


# Map each orchestrator stage to a normalized step index for the UI stepper.
# Steps (1-based): 1 Prepare · 2 Upload · 3 Launch GPU · 4 Generate · 5 Download · 6 Complete
_STAGE_STEP = {
    "auth": 1, "pack": 1,
    "dataset": 2,
    "kernel": 3,
    "running": 4,
    "download": 5,
    "done": 6,
}
TOTAL_STEPS = 6


def _on_event(job_id: str):
    def on_event(stage: str, message: str, extra: dict) -> None:
        fields = {"stage": stage, "message": message}
        if stage in _STAGE_STEP:
            fields["step"] = _STAGE_STEP[stage]
        if "ok" in extra:
            fields["ok"] = extra.get("ok") or 0
        if "failed" in extra:
            fields["failed"] = extra.get("failed") or 0
        if "total" in extra:
            fields["total"] = extra.get("total") or 0
        db.update_job(job_id, **fields)
    return on_event


def _track(job_id: str, work) -> None:
    """Run a job callable (fresh run or resume) and record done/failed."""
    try:
        summary = work(_on_event(job_id))
        db.update_job(
            job_id, status="done", stage="done", step=6,
            message=f"Done - {summary.get('ok')} of {summary.get('total')} succeeded",
            total=summary.get("total") or 0,
            ok=summary.get("ok") or 0,
            failed=summary.get("failed") or 0,
            error=None,
        )
        # Materialize outputs/batch_NN/{approved,review,failed}/ + CSV (spec §24/§26).
        # Best-effort: a reporting hiccup must never mark a good job as failed.
        try:
            reporting.materialize_batch(job_id)
        except Exception:
            traceback.print_exc()
    except JobError as e:
        db.update_job(job_id, status="failed", stage="error", message=str(e).splitlines()[0],
                      error=str(e))
    except Exception as e:  # noqa: BLE001 - never let a thread die silently
        db.update_job(job_id, status="failed", stage="error",
                      message=f"{type(e).__name__}: {e}",
                      error=traceback.format_exc())


def _run(job_id: str, images_dir: Path, cfg: dict) -> None:
    # Wait for a concurrency slot (spec §17). While blocked the job shows as
    # 'queued'; the moment a slot frees it flips to 'running'.
    acquired = _JOB_SLOTS.acquire(blocking=False)
    if not acquired:
        db.update_job(job_id, status="queued", stage="queued", step=0,
                      message="Waiting for a free GPU slot")
        _JOB_SLOTS.acquire()  # block until a slot frees
    try:
        # Advanced engine: when an image-API key is configured, generate via the
        # image model (fast, photoreal). Otherwise fall back to the Kaggle GPU
        # pipeline. Same job-folder contract either way, so the UI is unchanged.
        if api_engine.is_configured():
            db.update_job(job_id, status="running", stage="pack", step=1,
                          message="Preparing images (image-API engine)")
            _track(job_id, lambda oe: api_engine.run_api_job(images_dir, cfg, job_id, on_event=oe))
            return
        db.update_job(job_id, status="running", stage="auth", step=1,
                      message="Authenticating with Kaggle")
        _track(job_id, lambda oe: run_job(images_dir, cfg, job_id, on_event=oe))
    finally:
        _JOB_SLOTS.release()


def _resume(job_id: str) -> None:
    db.update_job(job_id, status="running", stage="running", step=4,
                  message="Reconnecting to the running job")
    _track(job_id, lambda oe: resume_job(job_id, on_event=oe))


def start_job(job_id: str, images_dir: Path, cfg: dict) -> None:
    with _threads_lock:
        existing = _threads.get(job_id)
        if existing and existing.is_alive():
            return  # idempotent: same job double-clicked won't spawn twice
        t = threading.Thread(target=_run, args=(job_id, images_dir, cfg), daemon=True)
        _threads[job_id] = t
    t.start()


def resume_running(job_id: str) -> None:
    """Reconnect to a job whose thread was lost (e.g. panel restart)."""
    with _threads_lock:
        existing = _threads.get(job_id)
        if existing and existing.is_alive():
            return
        t = threading.Thread(target=_resume, args=(job_id,), daemon=True)
        _threads[job_id] = t
    t.start()


def is_running(job_id: str) -> bool:
    with _threads_lock:
        t = _threads.get(job_id)
    return bool(t and t.is_alive())
