"""
Background job runner. Spawns the orchestrator in a thread and streams its
progress events into the SQLite store so the web UI can poll status.
"""

from __future__ import annotations

import threading
import traceback
from pathlib import Path

from . import db
from .kaggle_client import JobError, resume_job, run_job

# job_id -> Thread, so we can tell if a job is still actively running.
# Guarded by _threads_lock because HTTP handlers and background threads both
# read/write this dict - an unlocked dict race can leak zombie references.
_threads: dict[str, threading.Thread] = {}
_threads_lock = threading.Lock()


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
    except JobError as e:
        db.update_job(job_id, status="failed", stage="error", message=str(e).splitlines()[0],
                      error=str(e))
    except Exception as e:  # noqa: BLE001 - never let a thread die silently
        db.update_job(job_id, status="failed", stage="error",
                      message=f"{type(e).__name__}: {e}",
                      error=traceback.format_exc())


def _run(job_id: str, images_dir: Path, cfg: dict) -> None:
    db.update_job(job_id, status="running", stage="auth", step=1,
                  message="Authenticating with Kaggle")
    _track(job_id, lambda oe: run_job(images_dir, cfg, job_id, on_event=oe))


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
