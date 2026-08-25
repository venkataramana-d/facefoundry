"""
Tiny SQLite job store for FaceFoundry. Holds job history + live status so the
web UI survives restarts and can show past jobs. One file: jobs/facefoundry.db.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parent.parent
DB_PATH = REPO / "jobs" / "facefoundry.db"

_lock = threading.Lock()


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    return c


def init() -> None:
    with _lock, _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id          TEXT PRIMARY KEY,
                created_at  TEXT,
                updated_at  TEXT,
                status      TEXT,      -- queued | running | done | failed
                stage       TEXT,
                step        INTEGER DEFAULT 0,   -- normalized 0..6 for the UI stepper
                message     TEXT,
                style       TEXT,
                config      TEXT,      -- json
                total       INTEGER DEFAULT 0,
                ok          INTEGER DEFAULT 0,
                failed      INTEGER DEFAULT 0,
                error       TEXT
            )""")
        # migrate older DBs that predate the `step` column
        cols = {r["name"] for r in c.execute("PRAGMA table_info(jobs)").fetchall()}
        if "step" not in cols:
            c.execute("ALTER TABLE jobs ADD COLUMN step INTEGER DEFAULT 0")
        c.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                job_id    TEXT,
                stem      TEXT,
                decision  TEXT,        -- approved | rejected
                PRIMARY KEY (job_id, stem)
            )""")
        # Reprocess attempt counter per image (spec §21). Keyed by the ORIGINAL
        # job + stem so attempts accumulate across reprocess rounds.
        c.execute("""
            CREATE TABLE IF NOT EXISTS attempts (
                job_id    TEXT,
                stem      TEXT,
                attempt   INTEGER DEFAULT 1,
                PRIMARY KEY (job_id, stem)
            )""")


def create_job(job_id: str, style: str, config: dict) -> None:
    with _lock, _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO jobs "
            "(id, created_at, updated_at, status, stage, message, style, config) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (job_id, _now(), _now(), "queued", "queued", "waiting to start",
             style, json.dumps(config)),
        )


def update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = _now()
    cols = ", ".join(f"{k}=?" for k in fields)
    with _lock, _conn() as c:
        c.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))


def get_job(job_id: str) -> Optional[dict]:
    with _lock, _conn() as c:
        row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return dict(row) if row else None


def list_jobs(limit: int = 100) -> list[dict]:
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def delete_job(job_id: str) -> None:
    """Remove a job's DB rows (jobs + its review decisions)."""
    with _lock, _conn() as c:
        c.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        c.execute("DELETE FROM reviews WHERE job_id=?", (job_id,))
        c.execute("DELETE FROM attempts WHERE job_id=?", (job_id,))


# ---- Phase 4: per-image review decisions -----------------------------------
def set_review(job_id: str, stem: str, decision: str) -> None:
    with _lock, _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO reviews (job_id, stem, decision) VALUES (?,?,?)",
            (job_id, stem, decision),
        )


def get_reviews(job_id: str) -> dict[str, str]:
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT stem, decision FROM reviews WHERE job_id=?", (job_id,)
        ).fetchall()
    return {r["stem"]: r["decision"] for r in rows}


# ---- Reprocess attempt tracking (spec §21) ---------------------------------
def get_attempts(job_id: str) -> dict[str, int]:
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT stem, attempt FROM attempts WHERE job_id=?", (job_id,)
        ).fetchall()
    return {r["stem"]: r["attempt"] for r in rows}


def bump_attempts(job_id: str, stems: list[str]) -> None:
    """Increment (or initialize to 2 - i.e. a first reprocess) the attempt count
    for each stem. A brand-new job's images are implicitly attempt 1."""
    if not stems:
        return
    with _lock, _conn() as c:
        for stem in stems:
            c.execute(
                "INSERT INTO attempts (job_id, stem, attempt) VALUES (?,?,2) "
                "ON CONFLICT(job_id, stem) DO UPDATE SET attempt = attempt + 1",
                (job_id, stem),
            )


def set_attempts(job_id: str, stem: str, attempt: int) -> None:
    with _lock, _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO attempts (job_id, stem, attempt) VALUES (?,?,?)",
            (job_id, stem, attempt),
        )
