"""
Batch reporting + output materialization (spec §19, §24, §26, §27).

Engine-independent: reads each job's results.json plus the SQLite review/attempt
tables and produces:
  - per-image state derivation (spec §19),
  - a processing_report.csv (spec §26),
  - the outputs/batch_NN/{approved,review,failed}/ folder layout (spec §24),
  - aggregate dashboard numbers (spec §27).

No web framework here so it stays importable/testable on its own.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
from pathlib import Path

from . import db
from .kaggle_client import JOBS_DIR, REPO

OUTPUTS_ROOT = REPO / "outputs"

# Reprocess ceiling: after this many attempts a failing image is flagged for a
# human instead of being retried forever (spec §21).
MAX_ATTEMPTS = 3

# Per-image states (spec §19).
S_COMPLETED = "COMPLETED"
S_REVIEW = "REVIEW"
S_APPROVED = "APPROVED"
S_REJECTED = "REJECTED"
S_FAILED = "FAILED"
S_MANUAL = "MANUAL_REVIEW_REQUIRED"


def results_for(job_id: str) -> dict:
    """Load a job's output/results.json, or {} if absent."""
    p = JOBS_DIR / job_id / "output" / "results.json"
    if p.is_file():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def image_state(status: str, review: str | None, attempts: int) -> str:
    """Derive the spec §19 state for one finished image."""
    if status != "ok":
        return S_MANUAL if attempts >= MAX_ATTEMPTS else S_FAILED
    if review == "approved":
        return S_APPROVED
    if review == "rejected":
        return S_REJECTED
    return S_REVIEW


def per_image(job_id: str) -> list[dict]:
    """One row per generated image: stem, status, state, attempts, error, review."""
    res = results_for(job_id)
    reviews = db.get_reviews(job_id)
    attempts = db.get_attempts(job_id)
    rows = []
    for r in res.get("results", []):
        stem = r.get("stem", "")
        status = r.get("status", "failed")
        rev = reviews.get(stem)
        att = attempts.get(stem, 1)
        rows.append({
            "stem": stem,
            "status": status,
            "state": image_state(status, rev, att),
            "attempts": att,
            "output": r.get("output", "") if status == "ok" else "",
            "error": r.get("error", "") if status != "ok" else "",
            "review": rev or "",
        })
    return rows


def csv_report(job_id: str) -> str:
    """Build processing_report.csv content as a string (spec §26)."""
    job = db.get_job(job_id) or {}
    cfg = {}
    try:
        cfg = json.loads(job.get("config") or "{}")
    except Exception:
        cfg = {}
    batch_id = cfg.get("batch_id") or job_id
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["filename", "batch_id", "status", "attempts", "generation_time",
                "output_filename", "error", "review_status"])
    for row in per_image(job_id):
        w.writerow([
            row["stem"], batch_id, row["state"], row["attempts"],
            "",  # per-image generation_time not tracked by the worker
            row["output"], row["error"], row["review"],
        ])
    return buf.getvalue()


def _output_file(job_id: str, stem: str) -> Path | None:
    p = JOBS_DIR / job_id / "output" / "headshots_out" / f"{stem}.jpg"
    return p if p.is_file() else None


def materialize_batch(job_id: str) -> Path:
    """Copy generated images into outputs/batch_NN/{approved,review,failed}/
    according to review state. Never moves or overwrites originals (spec §24)."""
    job = db.get_job(job_id) or {}
    cfg = {}
    try:
        cfg = json.loads(job.get("config") or "{}")
    except Exception:
        cfg = {}
    batch_id = cfg.get("batch_id") or job_id
    base = OUTPUTS_ROOT / f"batch_{batch_id}"
    approved = base / "approved"
    review = base / "review"
    failed = base / "failed"
    for d in (approved, review, failed):
        d.mkdir(parents=True, exist_ok=True)
    for row in per_image(job_id):
        stem, state = row["stem"], row["state"]
        src = _output_file(job_id, stem)
        if state == S_APPROVED and src:
            shutil.copy2(src, approved / f"{stem}.jpg")
        elif state in (S_FAILED, S_MANUAL):
            # No image to copy; drop a marker so the folder reflects the failure.
            (failed / f"{stem}.txt").write_text(row["error"] or "failed")
        elif src:  # REVIEW or REJECTED both land in review/ for a human to sort
            shutil.copy2(src, review / f"{stem}.jpg")
    # Always (re)write the CSV alongside the batch outputs.
    (base / "processing_report.csv").write_text(csv_report(job_id))
    return base


def dashboard_totals(jobs: list[dict]) -> dict:
    """Aggregate spec §27 counters across a list of job dicts."""
    total = processed = approved = review = rejected = failed = 0
    batches = []
    for j in jobs:
        jid = j["id"]
        rows = per_image(jid)
        if not rows:
            # Job hasn't produced results yet.
            status = "PROCESSING" if j.get("status") in ("queued", "running") else "NOT STARTED"
            batches.append({"job": jid, "status": status, "total": j.get("total") or 0})
            continue
        n = len(rows)
        total += n
        processed += n
        approved += sum(r["state"] == S_APPROVED for r in rows)
        review += sum(r["state"] in (S_REVIEW,) for r in rows)
        rejected += sum(r["state"] == S_REJECTED for r in rows)
        failed += sum(r["state"] in (S_FAILED, S_MANUAL) for r in rows)
        batches.append({"job": jid, "status": "COMPLETED" if j.get("status") == "done"
                        else (j.get("status") or "").upper(), "total": n})
    return {
        "total": total, "processed": processed, "approved": approved,
        "review": review, "rejected": rejected, "failed": failed,
        "batches": batches,
    }
