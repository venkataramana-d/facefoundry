---
title: FaceFoundry
emoji: 📸
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# FaceFoundry

Turn a folder of ordinary profile photos into professional headshots, in bulk,
for **$0** - using free Kaggle GPUs driven automatically from your PC.

The heavy AI (SDXL + InstantID) runs on Kaggle's free Tesla P100. Your PC only
runs a small local control panel that uploads images, launches the GPU worker,
polls it, and pulls the finished headshots back. No always-on server, no GPU
bill.

```
Your PC (control panel)                 Kaggle (free GPU)
┌───────────────────────┐   upload    ┌────────────────────────┐
│ web UI / orchestrator │ ──────────► │ private dataset        │
│  app/server.py        │   launch    │ headshot_worker.py     │
│  app/kaggle_client.py │ ──────────► │  SDXL + InstantID       │
│                       │ ◄────────── │  → headshots + results  │
└───────────────────────┘  download   └────────────────────────┘
```

## Quick start

1. One-time Kaggle setup - see [SETUP.md](SETUP.md) (account, phone-verify for
   GPU, API token at `~/.kaggle/kaggle.json`).
2. Launch the control panel:

   ```bat
   run.bat
   ```

   (or manually: `pip install -r requirements.txt` then
   `python -m uvicorn app.server:app --port 8000`)
3. Open <http://localhost:8000>, click **New job**, pick a folder of photos and
   a style, and hit **Generate**.
4. Watch live progress → **review** the results (approve/reject, re-roll
   failures) → **download** the approved set.

> First run of a job spends ~10-15 min downloading the AI models on Kaggle, then
> a few seconds per image. Later jobs in the same Kaggle session are faster.

Want to host it instead of running locally? See [DEPLOY.md](DEPLOY.md).

## Command line (no UI)

```bash
python app/kaggle_client.py --images test_images --style corporate --limit 3 --job-id smoketest
```

## Upload modes

The **New job** page has three ways to add photos:

- **Folder** - pick a whole folder (uses `webkitdirectory`).
- **Single / files** - pick one or more individual images.
- **Camera** - on mobile, opens the front camera for a selfie.

Photos are **downsized to 1600px and re-encoded to JPEG in your browser** before
upload - this strips EXIF, keeps uploads small (20 MB → ~300 KB), and never
touches the source files on disk. The server also enforces limits: **50 files
per job, 25 MB per file, 400 MB total**.

## Styles & options

**8 style presets:** `corporate` · `modern_tech` · `warm_friendly` ·
`formal_executive` · `linkedin_classic` · `startup_casual` · `healthcare` ·
`academic` (edit `STYLE_PRESETS` in `worker/headshot_worker.py` to add your own).

**New-job options:**
- **Resolution** - Standard 1K · High 2K · **Ultra 4K** (generated at 1024 then
  upscaled with Real-ESRGAN 4x).
- **Render speed** - Fast (~20 steps) · Balanced (~30) · Best (~45).
- **Face enhancement (GFPGAN)** - optional, sharper eyes/skin. Fail-safe: if the
  optional deps don't install it's simply skipped, never breaking the job.
- **Pure white background** - clean corporate look (post-processed on Kaggle).
- **Multi-reference identity** - averages L2-normalized ArcFace embeddings
  across every source photo. Recommended for 2+ photos.
- **Background** and **extra prompt details** - free-text prompt tweaks
  (sanitized server-side against nsfw/gore terms and prompt-splitting chars).
- **Advanced** - identity strength, adapter scale, guidance, seed, limit.

**Review screen:** before/after grid, per-image download, download approved / all,
**Approve all / Reject all**, keyboard review (**A** keep · **R** reject · **J/K**
move), **re-roll** failures, and **re-run** the same faces in a different style.
Live runs show a progress ring + stepper and fire a browser notification when done.

## Faster runs (model cache)

Each Kaggle run re-downloads ~10 GB of models (~10-15 min). Build the cache **once**:

```bash
python app/kaggle_client.py --build-cache
```

This runs a one-off Kaggle kernel that downloads all models and publishes them as a
private **`facefoundry-models`** dataset. After that, every job auto-attaches it and the
worker loads models from it instead of downloading - cutting most of the setup wait.

## Image Settings editor

After a job finishes, each headshot has an **Edit** (✏️) button opening a browser studio:
crop / zoom / reframe, brightness / contrast / saturation / warmth, **background removal
→ fill color** (in-browser, for the clean corporate white look), **logo overlay** (drag to
place), and export at 1K/2K/4K JPG/PNG. **Batch edit** (from the review top bar) applies
one setup to every headshot and downloads a ZIP, with saveable presets.

## Layout

```
facefoundry/
├── run.bat                    # one-click launcher (Windows)
├── requirements.txt
├── Dockerfile                 # runs as non-root ff user
├── .dockerignore              # excludes jobs/, .kaggle, plans
├── render.yaml                # Render blueprint (free tier)
├── SETUP.md                   # one-time Kaggle account setup
├── DEPLOY.md                  # hosting on Render / Railway / VM
├── worker/
│   └── headshot_worker.py     # SDXL+InstantID pipeline (runs ON Kaggle)
├── app/
│   ├── kaggle_client.py       # orchestrator: upload → run → poll → download
│   ├── server.py              # FastAPI web UI (new job / progress / review)
│   ├── runner.py              # background job runner
│   └── db.py                  # SQLite job history + review decisions
├── tests/
│   └── test_safety.py         # 31 tests: safe_job_id/stem, kernel-state parser
├── test_images/               # sample faces for smoke tests
└── jobs/                       # per-job input/output + facefoundry.db (runtime)
```

## How a job flows

1. **Package** - your photos + a `job.json` (style, sliders) are copied into
   `jobs/<id>/input/`.
2. **Upload** - pushed as a **private** Kaggle dataset (employee photos stay
   private by default).
3. **Launch** - the worker kernel starts on a GPU with the dataset attached.
4. **Poll** - the control panel waits for completion.
5. **Download** - headshots land in `jobs/<id>/output/headshots_out/`, with a
   structured `results.json` (per-image status/seed) and a `run.log`.

## Safety hardening

- Uploads capped at 50 files / 25 MB per file / 400 MB total, PIL-verified
  after upload.
- Every path route regex-validates `job_id` and image stem - no glob leaks.
- Kaggle CLI calls have per-call timeouts (180 s / 900 s push) and retry 3x on
  transient errors.
- `POST /jobs/{id}/cancel` cancels a running kernel; deleting a job also cancels.
- Thread-locked in-memory job table; `start_job` is idempotent.
- Dockerfile runs as a non-root `ff` user.
- `sanitize_user_text()` strips nsfw/gore terms and prompt-splitting characters
  from user-supplied prompt text.

## Hosting extras

- **`/healthz`** - open endpoint (bypasses the password gate) so hosting
  platforms can health-check without credentials.
- **Password gate (optional)** - set `FACEFOUNDRY_PASSWORD` (and optionally
  `FACEFOUNDRY_USER`, default `team`) as env vars on the host to gate every
  route except `/healthz` behind HTTP basic auth.

## Troubleshooting

- **Job fails instantly at auth** - check `~/.kaggle/kaggle.json`. New `KGAT_`
  tokens are handled automatically (passed via `KAGGLE_API_TOKEN`).
- **Kernel errors** - open the job page → **run.log** link, or read
  `jobs/<id>/output/run.log`. The worker tees everything there even when Kaggle
  drops its own log.
- **"no face detected"** - the source photo needs a reasonably clear, front-ish
  face. Re-roll or swap the photo.

## Cost

| Item | Cost |
|---|---|
| Kaggle account + GPU (30 hr/wk) | $0 |
| SDXL / InstantID / antelopev2 weights | $0 (open weights) |
| Render free tier (optional hosting) | $0 |
| **Total** | **$0** |

Upgrade path if the free tier is ever too slow: swap the GPU backend (Replicate,
RunPod, Colab Pro) - the UI and worker stay the same.
