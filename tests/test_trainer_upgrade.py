"""Tests for the Corporate Trainer Profile upgrade (spec §7-§27).

Run with:  pytest -q tests/
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from app import reporting, styles
from app.server import STYLES, ASPECTS, VALID_ASPECTS, MAX_FILES

REPO = Path(__file__).resolve().parent.parent


def _load_worker():
    """Import worker/headshot_worker.py without its ML deps (all heavy imports
    are inside functions, so the module imports fine for reading STYLE_PRESETS)."""
    path = REPO / "worker" / "headshot_worker.py"
    spec = importlib.util.spec_from_file_location("headshot_worker", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["headshot_worker"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestStyleRegistration:
    def test_default_is_trainer(self):
        assert styles.DEFAULT_STYLE == "corporate_trainer_profile"

    def test_trainer_in_ui_list(self):
        keys = {v for v, _, _ in STYLES}
        assert "corporate_trainer_profile" in keys

    def test_worker_has_trainer_preset(self):
        w = _load_worker()
        assert "corporate_trainer_profile" in w.STYLE_PRESETS
        preset = w.STYLE_PRESETS["corporate_trainer_profile"]
        assert "charcoal" in preset["prompt"].lower()
        assert "polka-dot" in preset["prompt"].lower()
        # §9 identity negatives must be present.
        for term in ("different person", "missing glasses", "office background"):
            assert term in preset["negative"].lower()

    def test_prompt_fits_clip_budget(self):
        """CLIP reads only the first 77 tokens. Use a conservative word-count
        proxy (~1.3 tokens/word) so a future edit can't silently truncate the
        wardrobe/background out of the prompt."""
        w = _load_worker()
        prompt = w.STYLE_PRESETS["corporate_trainer_profile"]["prompt"]
        words = prompt.replace(",", " ").split()
        assert len(words) <= 58, f"prompt too long ({len(words)} words) - risks 77-token truncation"


class TestAspect:
    def test_3545_registered(self):
        assert "portrait_3545" in VALID_ASPECTS
        assert "portrait_3545" in {v for v, _, _ in ASPECTS}

    def test_3545_dims_exact_ratio(self):
        w, h = styles.aspect_dims("portrait_3545")
        assert (w, h) == (896, 1152)
        assert w % 64 == 0 and h % 64 == 0        # SDXL-friendly
        assert round(w / h, 4) == round(35 / 45, 4)  # == 0.7778

    def test_unknown_aspect_falls_back_square(self):
        assert styles.aspect_dims("nonsense") == (1024, 1024)


class TestBatchLimit:
    def test_max_files_is_100(self):
        assert MAX_FILES == 100


class TestImageState:
    def test_ok_no_review_is_review(self):
        assert reporting.image_state("ok", None, 1) == reporting.S_REVIEW

    def test_ok_approved(self):
        assert reporting.image_state("ok", "approved", 1) == reporting.S_APPROVED

    def test_ok_rejected(self):
        assert reporting.image_state("ok", "rejected", 2) == reporting.S_REJECTED

    def test_failed_under_limit(self):
        assert reporting.image_state("failed", None, 1) == reporting.S_FAILED

    def test_failed_at_limit_is_manual(self):
        assert reporting.image_state("failed", None, reporting.MAX_ATTEMPTS) == reporting.S_MANUAL


class TestCsvSchema:
    def test_header_columns(self):
        # csv_report tolerates a missing job (returns just the header row).
        text = reporting.csv_report("no-such-job-xyz")
        header = text.splitlines()[0]
        assert header == ("filename,batch_id,status,attempts,generation_time,"
                          "output_filename,error,review_status")
