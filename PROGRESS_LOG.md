# FaceFoundry — Build Progress Log

**Date:** 2026-08-18
**Owner:** Edstellar internal team (Ramana)
**Working dir:** `D:\bulk-headshot-tool`
**Tool name:** **FaceFoundry**

This log captures everything decided and built in this session, so anyone (or a future session) can pick up exactly where we left off.

---

## 1. What we started with

An existing folder with a **one-shot Kaggle notebook** that converts low-quality profile pics into professional headshots using **SDXL + InstantID** (AI image model + face-preservation), running free on Kaggle's GPU.

Original files:
| File | Purpose |
|---|---|
| `PLAN.md` | Strategy for the one-time 1,500-image batch job |
| `notebook.py` | The pipeline code (paste into Kaggle) |
| `urls_template.csv` | Example input format |
| `README.md` | Setup + tuning reference |

**Key realization:** this was a *one-shot batch job*, not a reusable tool. The goal of this session became: **turn it into a reusable tool (FaceFoundry).**

---

## 2. Decisions made

| Question | Decision | Why |
|---|---|---|
| Usage model | **Internal admin / batch tool** | Best fit for $0 + occasional big batches. A self-serve web app / API would need always-on GPU = real cost. |
| Budget | **$0 / free-tier** | No budget approved yet. Clean upgrade path kept for later. |
| Scale | **Occasional big batches** | Mostly idle, then hundreds/thousands at once. |
| GPU backend | **Free Kaggle, driven by API** | Only realistic $0 GPU for an ~8h batch; automated so no manual notebook running. |
| Control panel | **Runs locally on the PC** | The PC (Intel Iris Xe, no VRAM) can't do GPU but is fine as an orchestrator + UI. |
| Tool name | **FaceFoundry** (final) | Used everywhere. |

Full plan: [`TOOL_BUILD_PLAN.md`](TOOL_BUILD_PLAN.md)

---

## 3. What the tool does (v1 scope)

1. Drag in a **folder of images** (or CSV of URLs).
2. Pick a **style preset** + tune sliders (identity strength, steps, etc.).
3. Click **Run** → the tool automatically: uploads images as a private Kaggle dataset → launches the GPU worker → polls → downloads finished headshots.
4. **Review** side-by-side, approve/reject, **re-roll** failures.
5. **Download** the approved set. Job history remembered.

---

## 4. Setup completed (Ramana's part)

- ✅ Kaggle account created (`Ramana-7981`)
- ✅ Phone verified → **GPU unlocked** (note: verification is now inside a notebook's GPU toggle, not in Settings)
- ✅ API token created (`KGAT_...` new-style token)
- ✅ `kaggle.json` placed at `C:\Users\Ramana\.kaggle\kaggle.json`
- ⚠️ **TODO (security):** rotate the API key — it was visible in screenshots. Settings → API → Expire Token → Create New Token → rebuild `kaggle.json`.

---

## 5. Build progress

### ✅ Phase 0 — GPU automation validated (THE make-or-break test)
Proved we can drive Kaggle's free GPU **fully hands-off** via the API: pushed a kernel → it ran on a real **Tesla P100 (16GB)** → output pulled back automatically. **Zero clicks in Kaggle's website.** This retired the biggest risk in the whole plan.

Result:
```json
{ "cuda_available": true, "gpu_name": "Tesla P100-PCIE-16GB", "torch": "2.10.0+cu128" }
```

**Two gotchas discovered & fixed:**
1. New `KGAT_` tokens need to be passed via the `KAGGLE_API_TOKEN` env var for write operations (`kernels push`) — the legacy `kaggle.json` alone isn't enough.
2. Kaggle normalizes the owner slug (`Ramana-7981` → `ramana7981`); the kernel ref must be discovered from the API, not built from the display name.

### ✅ Phase 1 — Parameterized worker
`worker/headshot_worker.py` — the full SDXL+InstantID pipeline from `notebook.py`, now:
- reads a `job.json` config (style, sliders, limit) instead of hard-coded values,
- accepts **image files directly** (no more "must be public URL"),
- emits a structured `results.json` (per-image status/seed) for the future review UI,
- resumable (skips already-done images).

### ✅ Phase 2 — Orchestrator (the "one-click" engine)
`app/kaggle_client.py` — from a local folder, does the full loop automatically:
package images + job.json → push **private** Kaggle dataset → launch GPU worker → poll → download results.
Confirmed: datasets upload **private by default** (employee photos are safe).

### 🔁 End-to-end smoke test — debugging the worker (Kaggle 2026 image)
The first smoke test (`smoketest1`) errored with an **empty Kaggle log**. Made the
worker self-diagnosing (tees all output to `/kaggle/working/run.log`; always writes
`results.json` with the traceback even on a setup crash), then fixed **three**
separate dependency/setup bugs surfaced one run at a time:
1. **antelopev2 zip nesting** — unzip into `…/models/` not `…/models/antelopev2/`.
2. **NumPy 2 incompatibility** — pin `numpy<2` (insightface/onnxruntime/opencv are 1.x-era).
3. **peft ↔ accelerate mismatch** — pin `peft==0.11.1` (newer peft imports
   `accelerate.clear_device_cache`, absent in accelerate 0.29.2).

`smoketest3` got all the way through: deps, face init, **full 10.3GB SDXL download**,
pipeline component load — the peft fix was the last blocker. Re-running to confirm
3 headshots come back. (Full details saved to memory: `facefoundry-worker-deps.md`.)

### ✅ Phase 3 — Web UI (BUILT)
FastAPI control panel (`app/server.py` + `app/runner.py` + `app/db.py`):
New Job (folder upload + style + sliders) → live progress (polls status) →
review grid → download. Job history + review decisions in SQLite (`jobs/facefoundry.db`).
Orchestrator refactored to emit progress events and raise `JobError` (no more `sys.exit`)
so the server can run jobs in a background thread. All routes tested via TestClient.

### ✅ Phase 4 — Review & re-roll (BUILT)
Side-by-side original/generated grid, approve/reject per image, **one-click re-roll**
of rejected/failed faces as a new job with a bumped seed, download-approved as a zip.

### ✅ Phase 5 — Polish (BUILT)
`run.bat` one-click launcher (installs deps, warns if `kaggle.json` missing, opens
browser), full `README.md`, error surfacing on the job page (links to `run.log`).

---

## 6. Files created this session (in `headshot-studio/`, to be renamed `facefoundry/`)

```
headshot-studio/
├── SETUP.md                    # one-time Kaggle account setup steps
├── README.md                   # full tool docs (NEW)
├── run.bat                     # one-click launcher (NEW)
├── requirements.txt            # kaggle + fastapi + uvicorn + multipart
├── worker/
│   ├── phase0_validate.py      # GPU automation proof (PASSED)
│   └── headshot_worker.py      # SDXL+InstantID pipeline (self-logging, 3 dep bugs fixed)
├── app/
│   ├── __init__.py             # (NEW) package marker
│   ├── kaggle_client.py        # orchestrator: upload→run→poll→download (event/JobError refactor)
│   ├── server.py               # (NEW) FastAPI web UI — new job / progress / review
│   ├── runner.py               # (NEW) background job runner → SQLite
│   └── db.py                   # (NEW) SQLite job history + review decisions
├── test_images/                # 3 sample faces for testing
└── jobs/                       # per-job inputs/outputs + facefoundry.db (runtime)
```

> Note: the folder is still named `headshot-studio/` and will be renamed to `facefoundry/` once no job is running (renaming mid-run would break the active process).

---

## 7. Immediate next steps

1. **Confirm `smoketest3`** returns 3 real headshots (peft fix was the last blocker).
2. **Launch the UI** (`run.bat`) and run a job end-to-end from the browser.
3. **Rename** `headshot-studio/` → `facefoundry/` (once no job is running).
4. **Rotate the Kaggle API key** (security — was visible in screenshots).
5. Provide **real sample images** to validate quality before a full batch.

> Phases 3–5 (Web UI, review/re-roll, polish) are **built and tested**. The only
> open verification is a clean end-to-end GPU run producing headshots.

---

## 8. Cost recap (unchanged)

| Item | Cost |
|---|---|
| Kaggle account + GPU (30 hr/wk) | $0 |
| SDXL / InstantID / antelopev2 weights | $0 (open weights) |
| **Total** | **$0** |

Upgrade path if free tier ever falls short: Replicate API (~$45/1,500), RunPod (~$3/1,500), or Colab Pro ($10/mo) — the UI + worker stay the same, only the GPU backend swaps.
