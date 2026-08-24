# FaceFoundry

Turn a folder of ordinary profile photos into professional corporate headshots,
in bulk, for **$0** — using free Kaggle GPUs driven automatically from your PC.

The heavy AI (a photoreal SDXL checkpoint + InstantID for face preservation)
runs on Kaggle's free Tesla P100. Your PC only runs a small local control panel
that uploads images, launches the GPU worker, polls it, and pulls the finished
headshots back. No always-on server, no GPU bill.

```
Your PC (control panel)                 Kaggle (free GPU)
┌───────────────────────┐   upload    ┌────────────────────────┐
│ web UI / orchestrator │ ──────────► │ private dataset         │
│  app/server.py        │   launch    │ headshot_worker.py      │
│  app/kaggle_client.py │ ──────────► │  RealVisXL + InstantID  │
│                       │ ◄────────── │  → headshots + results  │
└───────────────────────┘  download   └────────────────────────┘
```

## Quick start

1. One-time Kaggle setup — see [SETUP.md](SETUP.md) (account, phone-verify for
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

> The first job in a Kaggle session spends a few minutes downloading the AI
> models, then only a few seconds per image. Later jobs in the same session are
> faster. Build a model cache (below) to cut most of the wait.

Want to host it instead of running locally? See [DEPLOY.md](DEPLOY.md).

## The interface

A clean "Executive Navy" studio interface:

- **Sidebar** — brand, New job, recent-jobs list, and a profile account menu
  (name, email, live job/headshot stats, plan, cost).
- **New job** — a two-column workspace: photos + style on the left, output
  settings and a live summary card with the Generate button on the right.
- **Style filter chips** — filter the presets by category (Corporate, Executive,
  Creative, Medical, Academic).
- **Progress** — a live stepper + ring, with a browser notification when done.
- **Review** — before/after grid, approve/reject, re-roll, batch edit.

## Command line (no UI)

```bash
python app/kaggle_client.py --images test_images --style edstellar_executive --limit 3 --job-id smoketest
```

## Upload modes

The **New job** page has three ways to add photos:

- **Folder** — pick a whole folder (uses `webkitdirectory`).
- **Single / files** — pick one or more individual images.
- **Camera** — on mobile, opens the front camera for a selfie.

Photos are **downsized to 1600px and re-encoded to JPEG in your browser** before
upload — this strips EXIF, keeps uploads small (20 MB → ~300 KB), and never
touches the source files on disk. The server also enforces limits: **50 files
per job, 25 MB per file, 400 MB total**.

## Styles & options

**9 style presets**, grouped by category:

| Category | Presets |
|---|---|
| Corporate | `corporate`, `linkedin_classic` |
| Executive | `formal_executive`, `edstellar_executive` |
| Creative | `modern_tech`, `warm_friendly`, `startup_casual` |
| Medical | `healthcare` |
| Academic | `academic` |

`edstellar_executive` is tuned to a corporate reference: navy suit, light-blue
shirt, navy tie, pure white studio background, glasses kept. Edit `STYLE_PRESETS`
in `worker/headshot_worker.py` to add your own.

> **Prompt discipline:** SDXL's CLIP text encoder only reads the first **77
> tokens**, so every preset prompt is kept short (~50–65 tokens) and front-loads
> the colour, wardrobe, and background words. Longer prompts get silently
> truncated and produce washed-out results — keep new presets concise.

**New-job options:**
- **Resolution** — Standard 1K · High 2K · **Ultra 4K** (generated at 1024 then
  upscaled with Real-ESRGAN 4x).
- **Render speed** — Fast (~20 steps) · Balanced (~30) · Best (~45).
- **Face enhancement (GFPGAN)** — optional, sharper eyes/skin, blended at 70%
  to keep natural skin texture. Fail-safe: if the optional deps don't install
  it's simply skipped, never breaking the job.
- **Pure white background** — clean corporate look (post-processed on Kaggle).
- **Multi-reference identity** — averages L2-normalized ArcFace embeddings across
  every source photo. Recommended for 2+ photos.
- **Background** and **extra prompt details** — free-text prompt tweaks
  (sanitized server-side against nsfw/gore terms and prompt-splitting chars).
- **Advanced** — identity strength, adapter scale, guidance, seed, limit.

## Generation engine

- **Default — free Kaggle GPU.** The worker runs **RealVisXL V4.0** (a photoreal
  SDXL checkpoint) with **InstantID** for face preservation, plus optional GFPGAN
  face restore, rembg white background, and Real-ESRGAN upscale. Falls back to
  base SDXL if the photoreal checkpoint can't be fetched. **$0.**
- **Optional — image API.** `app/api_engine.py` can instead generate via an image
  model (Google Gemini 2.5 Flash Image). It activates automatically when an API
  key is present (env `FACEFOUNDRY_API_KEY` / `GEMINI_API_KEY`, or a git-ignored
  `.api_key` file); otherwise the tool uses the free Kaggle engine. Note: Gemini
  image generation requires **billing enabled** on your Google account, so the
  free Kaggle engine is the default.

## Faster runs (model cache)

Each fresh Kaggle session re-downloads several GB of models. Build the cache **once**:

```bash
python app/kaggle_client.py --build-cache
```

This runs a one-off Kaggle kernel that downloads all models and publishes them as a
private **`facefoundry-models`** dataset. After that, every job auto-attaches it and the
worker loads models from it instead of downloading — cutting most of the setup wait.

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
├── render.yaml                # Render blueprint (free tier)
├── SETUP.md                   # one-time Kaggle account setup
├── DEPLOY.md                  # hosting on Render / Railway / VM
├── worker/
│   └── headshot_worker.py     # RealVisXL + InstantID pipeline (runs ON Kaggle)
├── app/
│   ├── kaggle_client.py       # orchestrator: upload → run → poll → download
│   ├── server.py              # FastAPI web UI (new job / progress / review / editor)
│   ├── api_engine.py          # optional image-API engine (Gemini)
│   ├── runner.py              # background job runner (selects engine)
│   └── db.py                  # SQLite job history + review decisions
├── tests/
│   ├── test_safety.py         # safe_job_id/stem, kernel-state parser
│   └── test_security_gate.py  # fail-closed auth, CSRF, rate limiting
├── test_images/               # sample faces for smoke tests
└── jobs/                       # per-job input/output + facefoundry.db (runtime, git-ignored)
```

## How a job flows

1. **Package** — your photos + a `job.json` (style, sliders) are copied into
   `jobs/<id>/input/`.
2. **Upload** — pushed as a **private** Kaggle dataset (employee photos stay
   private by default).
3. **Launch** — the worker kernel starts on a GPU with the dataset attached.
4. **Poll** — the control panel waits for completion.
5. **Download** — headshots land in `jobs/<id>/output/headshots_out/`, with a
   structured `results.json` (per-image status/seed) and a `run.log`.

## Safety hardening

- Uploads capped at 50 files / 25 MB per file / 400 MB total, PIL-verified after upload.
- Every path route regex-validates `job_id` and image stem — no glob leaks.
- Kaggle CLI calls have per-call timeouts and retry on transient errors; the
  orchestrator only ever runs under **your own** authenticated Kaggle account.
- Optional password gate + fail-closed auth, same-origin CSRF check, and per-IP
  rate limiting (see `tests/test_security_gate.py`).
- Dockerfile runs as a non-root `ff` user.
- `sanitize_user_text()` strips nsfw/gore terms and prompt-splitting characters
  from user-supplied prompt text.
- Secrets are never committed: `kaggle.json`, `.api_key*`, `.env`, `*.db`, and
  the `jobs/` folder (employee photos) are all git-ignored.

## Hosting extras

- **`/healthz`** — open endpoint (bypasses the password gate) so hosting
  platforms can health-check without credentials.
- **Password gate (optional)** — set `FACEFOUNDRY_PASSWORD` (and optionally
  `FACEFOUNDRY_USER`, default `team`) as env vars on the host to gate every
  route except `/healthz` behind HTTP basic auth.

> **Note on the free Render tier:** it has no persistent disk, so jobs and results
> are wiped on every redeploy/restart. Download results promptly, or attach a
> persistent disk if you need the live site to keep history.

## Troubleshooting

- **Job fails instantly at auth** — check `~/.kaggle/kaggle.json`. New `KGAT_`
  tokens are handled automatically (passed via `KAGGLE_API_TOKEN`).
- **Grey / washed-out output** — a style was truncated past 77 tokens, or the
  wrong style was picked. Check `run.log` for `Token indices ... (N > 77)`.
- **Kernel errors** — open the job page → **run.log** link, or read
  `jobs/<id>/output/run.log`. The worker tees everything there even when Kaggle
  drops its own log.
- **"no face detected"** — the source photo needs a reasonably clear, front-ish
  face. Re-roll or swap the photo.

## Cost

| Item | Cost |
|---|---|
| Kaggle account + GPU (30 hr/wk) | $0 |
| RealVisXL / InstantID / antelopev2 weights | $0 (open weights) |
| Render free tier (optional hosting) | $0 |
| **Total** | **$0** |

Upgrade paths if the free tier is ever too slow: build the model cache, attach a
Render disk for persistence, or plug in an image API (`app/api_engine.py`) — the
UI and worker stay the same.
