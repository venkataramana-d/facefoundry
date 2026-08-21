"""Tests for the access-control hardening: fail-closed auth, CSRF (same-origin),
and per-IP rate limiting.

Run with:  pytest -q tests/
"""
from __future__ import annotations

import importlib
import os

from fastapi.testclient import TestClient


def _fresh_app(**env):
    """Reload app.server with a specific env so module-level auth/host flags
    (evaluated at import) take effect."""
    for k in ("RENDER", "FACEFOUNDRY_PASSWORD", "FACEFOUNDRY_REQUIRE_AUTH"):
        os.environ.pop(k, None)
    os.environ.update({k: v for k, v in env.items()})
    from app import server
    importlib.reload(server)
    return server


def teardown_function(_):
    for k in ("RENDER", "FACEFOUNDRY_PASSWORD", "FACEFOUNDRY_REQUIRE_AUTH"):
        os.environ.pop(k, None)


class TestCsrf:
    def test_cross_origin_post_blocked(self):
        s = _fresh_app()
        with TestClient(s.app) as c:
            r = c.post("/jobs/nope/review", data={"stem": "a", "decision": "approved"},
                       headers={"Origin": "https://evil.example.com", "Host": "testserver"})
            assert r.status_code == 403

    def test_same_origin_post_allowed(self):
        s = _fresh_app()
        with TestClient(s.app) as c:
            r = c.post("/jobs/nope/review", data={"stem": "a", "decision": "approved"},
                       headers={"Origin": "http://testserver", "Host": "testserver"})
            assert r.status_code != 403

    def test_get_never_blocked(self):
        s = _fresh_app()
        with TestClient(s.app) as c:
            assert c.get("/", headers={"Origin": "https://evil.example.com"}).status_code == 200


class TestRateLimit:
    def test_unsafe_requests_are_limited(self):
        s = _fresh_app()
        with TestClient(s.app) as c:
            codes = [c.post("/jobs/nope/review",
                            data={"stem": "a", "decision": "approved"}).status_code
                     for _ in range(s._RATE_MAX + 15)]
            assert 429 in codes

    def test_get_not_rate_limited(self):
        s = _fresh_app()
        with TestClient(s.app) as c:
            assert all(c.get("/healthz").status_code == 200 for _ in range(200))


class TestFailClosed:
    def test_hosted_without_password_refuses(self):
        s = _fresh_app(FACEFOUNDRY_REQUIRE_AUTH="1")
        with TestClient(s.app) as c:
            assert c.get("/").status_code == 503
            assert c.get("/healthz").status_code == 200   # health check still open

    def test_require_auth_with_password_gates_normally(self):
        s = _fresh_app(FACEFOUNDRY_REQUIRE_AUTH="1", FACEFOUNDRY_PASSWORD="pw")
        with TestClient(s.app) as c:
            assert c.get("/").status_code == 401           # gated, not 503

    def test_no_password_is_open(self):
        s = _fresh_app()                                    # no password, not strict
        with TestClient(s.app) as c:
            assert c.get("/").status_code == 200           # open, no login
