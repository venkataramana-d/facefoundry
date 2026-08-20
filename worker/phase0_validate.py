"""
Phase 0 validation — the make-or-break test for the whole tool.

Goal: prove we can, entirely through the Kaggle API and with NO manual clicking:
  1. push a GPU-enabled kernel,
  2. run it,
  3. poll until it finishes,
  4. pull its output back locally,
  5. confirm a real GPU (T4) was actually used.

If this passes, the automation design in TOOL_BUILD_PLAN.md is sound and we build
the orchestrator + UI on top of it. If it fails, we fall back to semi-automation.

Usage (after completing SETUP.md):
    pip install kaggle
    python phase0_validate.py

You do NOT need to edit anything. The script reads your username from kaggle.json.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

POLL_SECONDS = 15
TIMEOUT_MINUTES = 15


def find_kaggle_json() -> Path:
    """Locate kaggle.json in the standard spot."""
    candidates = [
        Path.home() / ".kaggle" / "kaggle.json",
        Path(os.environ.get("KAGGLE_CONFIG_DIR", "")) / "kaggle.json",
    ]
    for c in candidates:
        if c and c.is_file():
            return c
    sys.exit(
        "ERROR: kaggle.json not found.\n"
        f"Expected at: {Path.home() / '.kaggle' / 'kaggle.json'}\n"
        "Follow SETUP.md Step 4 to place it there, then re-run."
    )


def get_credentials(kaggle_json: Path) -> tuple[str, str]:
    """Return (username, key) from kaggle.json."""
    data = json.loads(kaggle_json.read_text())
    if "username" not in data or "key" not in data:
        sys.exit("ERROR: kaggle.json missing 'username'/'key'. Re-create it (SETUP.md).")
    return data["username"], data["key"]


def auth_env(key: str) -> dict:
    """Build an environment that satisfies both legacy and new Kaggle auth.

    The KGAT_* tokens are Kaggle's newer format and must be supplied via
    KAGGLE_API_TOKEN; the older kaggle.json username/key path doesn't cover
    write operations like `kernels push` on recent CLI versions.
    """
    env = dict(os.environ)
    env["KAGGLE_API_TOKEN"] = key
    return env


def run_cli(args: list[str], env: dict | None = None, **kw) -> subprocess.CompletedProcess:
    """Run a kaggle CLI command and return the completed process (never raises on nonzero)."""
    return subprocess.run(
        [sys.executable, "-m", "kaggle", *args],
        capture_output=True,
        text=True,
        env=env,
        **kw,
    )


def check_kaggle_installed(env=None):
    proc = run_cli(["--version"], env=env)
    if proc.returncode != 0:
        sys.exit(
            "ERROR: the 'kaggle' package isn't installed or importable.\n"
            "Run:  pip install kaggle\n"
            f"Detail: {proc.stderr.strip()}"
        )


# The tiny program that runs ON Kaggle's GPU. It just proves the GPU is real.
GPU_CHECK_SCRIPT = '''\
import json, platform, subprocess, sys

result = {"python": platform.python_version(), "cuda_available": False, "gpu_name": None}
try:
    import torch
    result["torch"] = torch.__version__
    result["cuda_available"] = bool(torch.cuda.is_available())
    if result["cuda_available"]:
        result["gpu_name"] = torch.cuda.get_device_name(0)
except Exception as e:
    result["torch_error"] = repr(e)

# also capture nvidia-smi as a second signal
try:
    smi = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True)
    result["nvidia_smi_gpu"] = smi.strip()
except Exception as e:
    result["nvidia_smi_error"] = repr(e)

with open("/kaggle/working/gpu_report.json", "w") as f:
    json.dump(result, f, indent=2)

print("GPU_REPORT", json.dumps(result))
'''


def resolve_ref(slug: str, fallback: str, env: dict) -> str:
    """Find the canonical 'owner/slug' ref by listing the user's kernels.

    Kaggle lowercases/normalizes the owner part of a kernel ref, which won't
    match the display username in kaggle.json. We match on the kernel slug.
    """
    listing = run_cli(["kernels", "list", "--mine", "--csv"], env=env)
    for line in listing.stdout.splitlines():
        ref = line.split(",", 1)[0].strip()
        if ref.endswith("/" + slug):
            return ref
    return fallback


def main():
    print("=== Phase 0: Kaggle GPU automation validation ===\n")

    kaggle_json = find_kaggle_json()
    username, key = get_credentials(kaggle_json)
    env = auth_env(key)
    check_kaggle_installed(env)
    print(f"[ok] Found credentials for user: {username}")

    slug = "headshot-studio-phase0-gpu-check"
    kernel_id = f"{username}/{slug}"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "gpu_check.py").write_text(GPU_CHECK_SCRIPT)

        metadata = {
            "id": kernel_id,
            "title": "FaceFoundry Phase0 GPU Check",
            "code_file": "gpu_check.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": True,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
        }
        (tmp / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))

        print(f"[..] Pushing GPU kernel: {kernel_id}")
        push = run_cli(["kernels", "push", "-p", str(tmp)], env=env)
        print(push.stdout.strip() or push.stderr.strip())
        if push.returncode != 0:
            sys.exit("ERROR: push failed. See message above. (Common cause: GPU not unlocked — verify phone, SETUP.md Step 2.)")

    # Resolve the CANONICAL kernel ref. Kaggle normalizes the owner slug
    # (e.g. display name "Ramana-7981" -> ref owner "ramana7981"), so we must
    # discover the real ref from the API instead of building it from the name.
    kernel_id = resolve_ref(slug, kernel_id, env)
    print(f"[ok] Canonical kernel ref: {kernel_id}")

    # Poll until complete
    print("\n[..] Waiting for the kernel to run on Kaggle's GPU (this can take a few minutes)...")
    deadline = time.monotonic() + TIMEOUT_MINUTES * 60
    last = None
    while time.monotonic() < deadline:
        status = run_cli(["kernels", "status", kernel_id], env=env)
        text = (status.stdout + status.stderr).strip()
        if text != last:
            print(f"    status: {text}")
            last = text
        low = text.lower()
        if "complete" in low:
            break
        if "error" in low:
            sys.exit("ERROR: kernel reported an error. Open it on kaggle.com to see the log.")
        time.sleep(POLL_SECONDS)
    else:
        sys.exit("ERROR: timed out waiting for the kernel. Check it on kaggle.com.")

    # Fetch output
    out_dir = Path(__file__).parent / "phase0_output"
    out_dir.mkdir(exist_ok=True)
    print(f"\n[..] Downloading output to: {out_dir}")
    fetch = run_cli(["kernels", "output", kernel_id, "-p", str(out_dir)], env=env)
    print(fetch.stdout.strip() or fetch.stderr.strip())

    report_path = out_dir / "gpu_report.json"
    if not report_path.is_file():
        sys.exit("ERROR: no gpu_report.json came back. Check the kernel output on kaggle.com.")

    report = json.loads(report_path.read_text())
    print("\n=== RESULT ===")
    print(json.dumps(report, indent=2))

    if report.get("cuda_available") and report.get("gpu_name"):
        print(f"\n✅ PASS — GPU automation works. Kaggle ran our code on: {report['gpu_name']}")
        print("   The Phase 0 foundation is solid. We can build the orchestrator + UI next.")
    else:
        print("\n❌ FAIL — kernel ran but no GPU was detected.")
        print("   Likely the account's GPU isn't unlocked (phone verification) or accelerator wasn't granted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
