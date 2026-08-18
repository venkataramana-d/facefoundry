# Build Plan — FaceFoundry (Reusable Internal Tool)

**Owner:** Edstellar internal team
**Author of plan:** dev team
**Last updated:** 2026-08-18
**Status:** Proposed — awaiting go-ahead

This plan turns the one-shot Kaggle notebook (`notebook.py` / `PLAN.md`) into a **repeatable internal tool** with a real UI, one-click runs, and automated GPU offload. It is the "future project" that the original plan deferred ([PLAN.md §10](PLAN.md)).

---

## 1. Decisions locked in

| Decision | Choice | Why |
|---|---|---|
| **Usage model** | Internal admin / batch tool | Best fit for $0 + occasional big batches. A self-serve web app or API would need always-on GPU = real monthly cost, wasteful when idle most of the time. |
| **Budget** | $0 / free-tier for v1 | No budget approved yet. Design keeps a clean paid upgrade path (see §9). |
| **Scale** | Occasional big batches | Mostly idle, then hundreds–thousands at once. Perfect for free GPU that spins up on demand and spins down. |
| **GPU backend** | Free Kaggle, driven by API | Only realistic $0 GPU that fits an ~8h batch. Automated so no manual notebook running. |
| **Control panel** | Runs locally on your PC | Your PC (Intel Iris Xe, no VRAM) can't do GPU, but it's fine as a lightweight orchestrator + UI. |

---

## 2. What the tool does (v1 scope)

**In scope:**
1. Drag-and-drop a **folder of images** OR a **CSV of URLs** into a local web UI.
2. Pick a **style preset** (corporate / modern_tech / warm_friendly / formal_executive) and tune sliders (identity scale, steps, etc.).
3. Click **Run** → tool automatically:
   - uploads images to a private Kaggle dataset (no more "must be public URL"),
   - launches the GPU kernel on Kaggle,
   - polls progress,
   - downloads the finished ZIP when done.
4. **Review screen:** side-by-side source → headshot grid; mark approve / reject.
5. **Re-roll** rejected or failed images (different seed / higher identity scale) in one click.
6. **Download** approved set as a ZIP, or export to a chosen folder.
7. **Job history:** past batches, their settings, and outputs are remembered (so results are reproducible).

**Out of scope for v1 (future):**
- Multi-user accounts / permissions.
- Always-on public web app or API.
- Automatic face-quality scoring / auto-approval.
- Cloud storage integrations (SharePoint/Drive/S3) — v1 just reads/writes local folders.

---

## 3. Architecture

```
YOUR PC (local control panel)                         KAGGLE (free GPU, on demand)
┌────────────────────────────────────┐                ┌─────────────────────────────────┐
│  Browser UI (localhost:8000)       │                │                                 │
│   - upload folder / CSV            │                │   Private Dataset               │
│   - pick style + sliders           │                │   (your images, auto-uploaded)  │
│   - watch progress                 │                │            │                    │
│   - review + approve/reroll        │                │            ▼                    │
│            ▲    │                   │   Kaggle API   │   GPU Kernel (headshot_worker)  │
│            │    ▼                   │◀──────────────▶│   SDXL + InstantID (from        │
│  Local backend (FastAPI)           │  push / run /  │   notebook.py, parameterized)   │
│   - job manager (SQLite)           │  poll / output │            │                    │
│   - Kaggle orchestration           │                │            ▼                    │
│   - stores inputs/outputs locally  │                │   headshots_<job>.zip (output)  │
└────────────────────────────────────┘                └─────────────────────────────────┘
        │                                                          │
        └──────────────  downloads ZIP back automatically  ◀───────┘
```

**End-to-end data flow:**
`local images` → packaged → `Kaggle private dataset` → `GPU kernel runs SDXL+InstantID` → `output ZIP` → auto-downloaded to `local job folder` → shown in review UI → approved set exported.

---

## 4. Tech stack (all free / open-source)

| Layer | Choice | Notes |
|---|---|---|
| Backend | **Python + FastAPI** | Reuses all existing model/pipeline code; same language as `notebook.py`. |
| Job state | **SQLite** | Zero-setup local DB for job history + status. |
| Frontend | **Plain HTML + htmx** (or a small React app) | Keep it simple; it's an internal tool, not a product. htmx avoids a build step. |
| GPU orchestration | **Kaggle API** (`kaggle` pip package) | `datasets create/version`, `kernels push`, `kernels status`, `kernels output`. |
| GPU worker | **Existing `notebook.py`, refactored** | Turned into a parameterized kernel that reads a job config file. |
| Packaging | **Local run script** (`run.bat` / `run.sh`) | Double-click to start the tool on your PC. |

---

## 5. Component breakdown

### 5.1 GPU worker (refactor of `notebook.py`)
- Read job parameters (style, sliders, sample mode) from a `job.json` in the input dataset instead of hard-coded globals.
- Read images from the attached dataset (not just URLs).
- Keep the **resumable batch loop** and `gen_failures` logging (already in `notebook.py`).
- Emit a structured `results.json` (per-image: status, seed, output filename) alongside the ZIP so the review UI can render it.

### 5.2 Kaggle orchestrator (backend module)
- `push_dataset(job_id, images)` → create/version a private dataset.
- `launch_kernel(job_id, config)` → write `kernel-metadata.json` (GPU on, internet on, dataset attached), `kaggle kernels push`.
- `poll(job_id)` → `kaggle kernels status` until complete/error.
- `fetch_output(job_id)` → `kaggle kernels output` → unzip into the local job folder.

### 5.3 Job manager (backend)
- Create job → track state machine: `queued → uploading → running → downloading → review → done`.
- Persist to SQLite; survive restarts (resume polling a running Kaggle job).

### 5.4 Web UI
- **New Job** page: upload, style, sliders, run.
- **Job** page: live status, progress bar, log tail.
- **Review** page: source→output grid, approve/reject, re-roll, export.
- **History** page: list past jobs with settings + outputs.

---

## 6. Phased build roadmap

Each phase is independently testable and leaves you with something that works.

### Phase 0 — Prep & credentials (~half a day)
- Kaggle account with phone-verified GPU (already needed for notebook).
- Generate a Kaggle API token (`kaggle.json`).
- Confirm you can push+run a trivial GPU kernel via API and pull its output. **This de-risks the whole plan — do it first.**

### Phase 1 — Parameterize the worker (~1 day)
- Refactor `notebook.py` to read `job.json` and image dataset.
- Test it manually on Kaggle with 10 images. Output = ZIP + `results.json`.

### Phase 2 — Orchestrator (headless) (~2 days)
- Backend that, from a local folder, does the full loop automatically: upload dataset → launch kernel → poll → download ZIP.
- Runs from command line, no UI yet. **This is the core value — automation.**

### Phase 3 — Web UI (~2–3 days)
- FastAPI + pages from §5.4. Wire New Job → orchestrator → Review.
- Job history in SQLite.

### Phase 4 — Review & re-roll (~1–2 days)
- Side-by-side grid, approve/reject, one-click re-roll of failures, export approved set.

### Phase 5 — Polish (~1 day)
- `run.bat` launcher, error messages, quota/failure surfacing, README.

**Rough total: ~1.5–2 weeks of focused build** for a solid internal v1.

---

## 7. Repo / file structure (proposed)

```
facefoundry/
├── worker/
│   └── headshot_worker.py      # refactored notebook.py (runs on Kaggle)
│   └── kernel-metadata.json    # generated per job
├── app/
│   ├── main.py                 # FastAPI entry
│   ├── kaggle_client.py        # push/run/poll/fetch
│   ├── jobs.py                 # job manager + SQLite
│   ├── templates/              # UI pages
│   └── static/
├── jobs/                       # local job folders (inputs + outputs) — gitignored
├── run.bat / run.sh
├── requirements.txt
└── README.md
```

---

## 8. Key risks & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **Kaggle API can't launch GPU kernels the way we need** | High — kills the design | **Validate in Phase 0 before any other work.** If blocked, fall back to semi-automation (tool preps everything, you click Run in Kaggle). |
| Kaggle ToS / automation limits | Medium | Occasional batches only; stay well under 30 GPU-hrs/wk. Don't hammer the API. |
| Weekly GPU quota exhausted | Medium | Tool shows remaining quota; queue batches across weeks if needed. |
| Big image uploads slow to push as dataset | Low | Compress/resize sources before upload; show progress. |
| Session dies mid-batch | Low | Worker loop already resumable; orchestrator re-launches and skips done images. |
| Kaggle changes free policy | Low/High | Paid upgrade path in §9 keeps the same UI + worker, swaps the backend. |

---

## 9. $0 now, clean upgrade path later

The UI, job manager, and worker code **don't change** if you later get budget — only the "where GPU runs" swaps out:

| Tier | GPU backend | Effort to switch | Cost |
|---|---|---|---|
| **v1 (now)** | Free Kaggle via API | — | $0 |
| Upgrade A | **Replicate / fal.ai API** (`instant-id` model) | Swap `kaggle_client.py` for an API client. No UI changes. | ~$0.03/image (~$45 / 1,500) |
| Upgrade B | **RunPod / rented T4** | Point orchestrator at a rented GPU running the worker as a service. | ~$0.30/hr (~$3 / 1,500) |
| Upgrade C | **Always-on** (enables self-serve web app / API model) | Add auth + hosting; reuse worker + UI. | Monthly infra |

This is why the internal-tool model is the right v1: it's the cheapest thing that works **and** the foundation the paid models build on top of.

---

## 10. Success criteria for v1

- [ ] Non-technical teammate can run a batch **without touching Kaggle or code**.
- [ ] Full loop (upload → GPU → download) is **one click**, no manual notebook steps.
- [ ] 90%+ of a mixed batch produces usable headshots (matches notebook quality).
- [ ] Failed/rejected images can be **re-rolled** without re-running the whole batch.
- [ ] Every job's settings + outputs are **saved and reproducible**.
- [ ] Runs at **$0**.

---

## 11. Open questions (need your input before/during build)

1. **UI simplicity:** plain HTML+htmx (no build step, fastest) vs a small React app (nicer, more work)? Recommend htmx for v1.
2. **Where do source images come from most often** — a local folder, or URLs? (Affects which upload path we polish first.)
3. **Approval workflow:** is a simple approve/reject enough, or do you need notes/tagging per image?
4. **Output destination:** local folder ZIP only for v1, or do you already know you'll need SharePoint/Drive export soon?
5. **Who runs it:** just you, or several teammates on their own PCs? (v1 assumes single local user; multi-user is a later phase.)

---

## 12. Immediate next step

**Do Phase 0 first** — prove that a GPU Kaggle kernel can be pushed, run, and its output pulled entirely through the Kaggle API. Everything else depends on it. If you want, I can write that Phase 0 validation script next so we confirm the foundation before building the rest.
