"""Tests for the safety helpers added in Bundle A.

Run with:  pytest -q tests/
"""
from __future__ import annotations

import pytest

from app.server import _safe_job_id, _safe_stem, MAX_FILES, MAX_FILE_BYTES
from app.kaggle_client import _parse_kernel_state, slugify_job_id
from fastapi import HTTPException


class TestSafeJobId:
    def test_accepts_slugified(self):
        assert _safe_job_id("job1234") == "job1234"
        assert _safe_job_id("linkedin-batch-1") == "linkedin-batch-1"

    @pytest.mark.parametrize("bad", [
        "", "../etc", "job/1", "JOB", "a b", "j" * 65, ".job", "-job", "job?", "job*",
    ])
    def test_rejects_bad(self, bad):
        with pytest.raises(HTTPException) as e:
            _safe_job_id(bad)
        assert e.value.status_code == 400


class TestSafeStem:
    def test_accepts_normal(self):
        assert _safe_stem("photo") == "photo"
        assert _safe_stem("IMG_0123") == "IMG_0123"
        assert _safe_stem("selfie-2026.01") == "selfie-2026.01"

    @pytest.mark.parametrize("bad", [
        "", "../etc", "a/b", "a b", "a*b", "a?b", "a[b", "*", "?", "[",
    ])
    def test_rejects_traversal_or_glob(self, bad):
        with pytest.raises(HTTPException):
            _safe_stem(bad)


class TestSlugifyJobId:
    def test_lowercases_and_dashes(self):
        assert slugify_job_id("My Job 42") == "my-job-42"

    def test_strips_edges(self):
        assert slugify_job_id("--foo--") == "foo"

    def test_falls_back_when_empty(self):
        out = slugify_job_id("")
        assert out.startswith("j") and out[1:].isdigit()


class TestParseKernelState:
    def test_complete(self):
        assert _parse_kernel_state('... "KernelWorkerStatus.COMPLETE"') == "complete"

    def test_error(self):
        assert _parse_kernel_state('KernelWorkerStatus.ERROR happened') == "error"

    def test_running(self):
        assert _parse_kernel_state('kernelworkerstatus.running now') == "running"

    def test_error_word_in_ref_is_not_state(self):
        # A kernel ref that mentions 'error' or 'worker' must not be misparsed
        # as an actual ERROR state - this was a real bug historically.
        assert _parse_kernel_state('owner/error-worker-job has status "KernelWorkerStatus.RUNNING"') == "running"

    def test_unknown_falls_back(self):
        assert _parse_kernel_state("garbled output") == "unknown"


def test_upload_limits_are_sane():
    assert 1 <= MAX_FILES <= 500
    assert 1024 * 1024 <= MAX_FILE_BYTES <= 200 * 1024 * 1024
