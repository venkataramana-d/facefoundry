# FaceFoundry — Corporate Trainer Profile Upgrade: Implementation Plan

**Status:** IMPLEMENTED (all increments). Local app verified (53/53 tests pass,
server + routes smoke-tested). The worker's image *quality* still needs a real
Kaggle GPU run to validate (spec Phase 2/3) — code is in place but unrun on GPU.
**Engine decision (locked):** Kaggle / SDXL + InstantID only. The paid Gemini
image-edit path is left in place as-is but is **not** the target of this upgrade.
**Spec:** `FaceFoundry – Corporate Trainer Profile Image Upgrade Specification.md`

### Implemented changes (this session)
- `app/styles.py` (new): `CORPORATE_TRAINER_PROFILE`, `DEFAULT_STYLE`, `aspect_dims`.
- `worker/headshot_worker.py`: new `corporate_trainer_profile` preset (77-token
  prompt + §9 negatives); native 35:45 render at 896×1152 with aspect-correct
  keypoint canvas; ratio-aware upscale.
- `app/server.py`: trainer style is default+recommended; `portrait_3545` aspect;
  `MAX_FILES` 100; batch validation report; `/dashboard`; `report.csv`,
  `failed_report`, `materialize`, `/compare` (+`/jobs/{id}/compare`) routes;
  active-config + validation + per-image state on the review page.
- `app/runner.py`: `MAX_CONCURRENT_JOBS` queue; multi-reference off for trainer;
  auto-materialize batch outputs on completion.
- `app/db.py`: `attempts` table + accessors.
- `app/reporting.py` (new): per-image state, CSV, `outputs/batch_NN/` layout,
  dashboard aggregation.
- `tests/test_trainer_upgrade.py` (new): 15 tests (style/aspect/state/CSV/limit).

---

## 0. TL;DR

We keep the entire existing app (upload → job → review → keep/reject → retry →
download) and change **only** the generation configuration plus add a batch-ops
layer. The three increments:

- **A. New style + composition** — add a centralized `Corporate Trainer Profile`
  style, wire the Master Prompt (compressed to fit CLIP's 77-token limit), add a
  true **35:45** aspect, make it the default. *(spec §7, §8, §9, §11, §13, §14)*
- **B. Quality loop + A/B test mode** — compare old vs new config on 1 → 10
  trainers before scaling. *(spec §2, §32, §33, Phases 2–3)*
- **C. Batch-100 operations** — 100-image batches, validation report, job queue,
  progress, per-image states, reprocess w/ attempt tracking, `outputs/batch_NN/`
  layout, ZIP + CSV, dashboard, usage. *(spec §15–§27, §30)*

Nothing in Increment A/C removes an existing feature (spec §3).

---

## 1. Diagnosis recap (spec §35 — "Immediate Next Action")

| Aspect | Current Kaggle/InstantID implementation | File |
|---|---|---|
| Model | `RealVisXL_V4.0` (SDXL photoreal finetune); fallback SDXL-base-1.0 | `worker/headshot_worker.py:371` |
| Identity | InstantID = ControlNet + IP-Adapter over an **ArcFace embedding** (InsightFace `antelopev2`); optional multi-photo embedding averaging | `worker:333`, `worker:696` |
| Pipeline | **Text-to-image** conditioned on embedding + face-keypoint control map. Original pixels are **not** fed through | `worker:620` |
| Prompt | `STYLE_PRESETS` dict, front-loaded, hard-capped at **77 CLIP tokens** | `worker:170` |
| Negative | Shared `_NEG` string | `worker:159` |
| Gen params | steps 30 · guidance 5.0 · identity(controlnet) 0.8 · adapter 0.8 · seed `seed_base+i` · 1024² · SDPA attn | `worker:620` |
| Preproc | face detect → keypoints → headroom reframe (shrink kps 72%, shift down 22%) | `worker:600` |
| Postproc | GFPGAN blend 0.45 → rembg white-bg → vibrance (edstellar only) → Real-ESRGAN upscale → 4:5 pad | `worker:628` |
| Aspect | `square` (1:1) or `portrait` (4:5 by **padding**, never crop) | `worker:684` |
| Batch | 1 job = 1 Kaggle dataset + 1 kernel, images processed in a **sequential loop**, resumable (skips existing outputs) | `worker:747`, `kaggle_client.py:404` |
| Filename | Output name = source stem `.jpg` — **already preserved** ✅ | `worker:748` |

### Root cause of inconsistent results (for the Kaggle path we're committing to)

```text
CURRENT SYSTEM → text2img from a 512-d face embedding (InstantID)
PROBLEM        → identity drift + style inconsistency
ROOT CAUSE     → (1) fine detail (exact glasses, hairline, asymmetry, age) is
                     synthesized, not copied — a known ceiling of embedding-only
                     identity (spec §6's warned-against pattern);
                 (2) 77-token CLIP cap truncates long instructions;
                 (3) no trainer style, no true 35:45, MAX_FILES=50, no batch ops.
PROPOSED CHANGE → tune within InstantID's safe envelope + a compressed, correct
                     prompt + native 35:45 rendering + full batch-ops layer.
```

### Honest constraints of the Kaggle engine (must be accepted)

1. **77-token prompt cap is real.** The full Master Prompt (§8) *cannot* be sent
   verbatim; it must be compressed and front-loaded. The rest of the intent lives
   in the negative prompt, generation params, and post-processing. Watch
   `run.log` for `Token indices ... (N > 77)`.
2. **Identity has a ceiling.** InstantID reconstructs the face from an embedding.
   We can raise recognizability (multi-reference averaging, tuned identity scale)
   but cannot guarantee pixel-faithful glasses/skin the way an image *edit* would.
   Success criteria §33 ("clearly recognizable") is achievable; "pixel-identical"
   is not, by design of this engine.
3. **No live per-image progress mid-kernel.** Kaggle returns kernel output only
   when the run *finishes* (`kernels output`). So §18/§19 live states are
   **stage-level while running** (Prepare/Upload/Launch/Generate/Download), and
   **per-image states resolve on completion** from `results.json`. We will not
   fake a streaming per-image bar the engine can't back.
4. **Batch size vs. runtime.** 100 images in one kernel is fine, but Kaggle
   session/GPU time limits apply; first run of the day spends ~10–15 min warming
   models. Plan for 100/kernel, resumable if it times out.

---

## 2. Increment A — Corporate Trainer Profile style (spec §7, §8, §9, §11, §13, §14)

### A1. Centralized style config (spec §14 — "do not duplicate the prompt")

Add one source of truth. Proposed location: a new `app/styles.py` imported by
both `server.py` (UI list + defaults) and referenced conceptually by the worker.
(The worker runs on Kaggle from its own file, so its `STYLE_PRESETS` entry is the
executable copy; `styles.py` holds the canonical spec + is the single place the
local app reads style metadata from.)

```python
# app/styles.py  (new)
CORPORATE_TRAINER_PROFILE = {
    "key": "corporate_trainer_profile",
    "name": "Corporate Trainer Profile",
    "category": "Corporate",
    "aspect_ratio": "35:45",            # new aspect, default for this style
    "background": "white to very light-gray studio",
    "clothing": "charcoal/dark-gray suit, crisp white shirt, light-gray polka-dot tie",
    "lighting": "soft professional studio, even facial illumination",
    "expression": "calm, confident, subtle natural smile",
    "identity_preservation": "maximum",
    "logo": False,
    "text": False,
    "default_resolution": "2048",
    "default_speed": "balanced",
    "face_enhance": True,               # gentle (blended 0.45), keeps skin natural
    "white_background": False,          # backdrop comes from the prompt, not rembg
}
```

### A2. Worker prompt entry (spec §8, §9) — compressed to ≤77 tokens

Add to `STYLE_PRESETS` in `worker/headshot_worker.py`. The Master Prompt (§8) is
distilled to its color/wardrobe/background/identity essentials, front-loaded:

```python
"corporate_trainer_profile": {
    "prompt": ("RAW color photo, professional corporate LinkedIn headshot of a "
               "person, chest-up, tailored charcoal dark-grey suit, crisp white "
               "dress shirt, light-grey polka-dot tie, clean white to light-grey "
               "studio backdrop, soft even studio lighting, keeps eyeglasses, "
               "keeps real age, calm confident subtle smile, natural detailed "
               "skin texture, sharp focus on the eyes, 85mm, realistic photograph"),
    "negative": _NEG + (", different person, identity change, face replacement, "
               "beautified, de-aged, missing glasses, extra glasses, office "
               "background, outdoor background, busy background, strong shadows, "
               "dramatic lighting, illustration, 3d render, uncanny"),
},
```

*Rationale:* every §9 negative term is honored either in `_NEG` or the appended
tail. Every §8 positive intent that fits the token budget is front-loaded; the
overflow ("no logo/text/watermark", "no beauty retouch") is enforced via the
negative prompt + the existing postproc discipline (GFPGAN blended at only 0.45
so skin stays natural, `worker:642`).

### A3. True 35:45 aspect (spec §11) — render native, don't pad

Current `portrait` pads a square to 4:5. For the trainer style we render
**natively** at a 35:45 canvas for correct framing and full resolution.

- 35:45 = 0.7778. SDXL-friendly dims: **896 × 1152** (both /64, ratio exact).
- Changes in `worker/headshot_worker.py`:
  - `make_processor`: when `cfg["aspect"] == "portrait_3545"`, set
    `gen_w, gen_h = 896, 1152` (instead of square `gen_size`), and pass
    `width=gen_w, height=gen_h` to `pipe(...)` (`worker:626`).
  - **Build the keypoint control map at the target aspect** so InstantID doesn't
    squash it: construct the `kctrl` canvas at `896×1152` (not source-square) in
    the headroom block (`worker:608`). This is the one subtle correctness fix.
  - Upscale ratio-aware: replace the square `resize((out_size, out_size))` calls
    (`worker:671`, `worker:674`) with ratio-preserving resize to
    `(out_w, out_h)` where the long edge = `out_size`.
  - Keep the existing `portrait` (4:5 pad) branch untouched for other styles.
- Changes in `app/server.py`:
  - `ASPECTS` (`server.py:181`): add `("portrait_3545", "Portrait", "35:45")`.
  - `create_job` aspect validation (`server.py:1090`): widen the allowed set to
    `{"square", "portrait", "portrait_3545"}`.

### A4. UI wiring (spec §7 default, §13 preset visibility)

- `STYLES` (`server.py:144`): prepend
  `("corporate_trainer_profile", "Corporate Trainer Profile", "Charcoal suit, white shirt, 35:45")`.
- `STYLE_CAT` (`server.py:156`): `"corporate_trainer_profile": "Corporate"`.
- Make it the **default selection**: the "checked"/"Recommended" logic currently
  hard-codes `edstellar_executive` (`server.py:720`, `:722`); point both at
  `corporate_trainer_profile`.
- `create_job` "opinionated preset" block (`server.py:1079`): generalize the
  `is_edstellar` special-casing so the trainer style forces its config
  (aspect=35:45, face_enhance=on, white_background handled by prompt) and
  **the active configuration is shown** in the job page (spec §13 "clearly show
  which configuration is active") — render the resolved `cfg` summary on
  `/jobs/{id}`.
- Add a `BRAND_PACKS` entry (`server.py:190`) "Corporate Trainer" → this style,
  2048, balanced, `portrait_3545`, face_enhance=True.

### A5. Branding removal (spec §10)

- Generation already adds no logo; `text/logo/watermark` are in `_NEG`. The new
  style's config sets `logo=False, text=False`.
- The **manual** logo tool in the client-side editor (`server.py:1389`) is a
  user-initiated post-export overlay — it does not auto-brand output. Per §10 we
  leave the tool available but ensure no style/pack pre-loads a logo. (Optional:
  hide the logo panel when the trainer style is active — confirm if wanted.)

**Increment A acceptance:** one trainer photo → recognizable person, charcoal
suit / white shirt / light-grey tie, clean light studio bg, 35:45 framing, no
logo/text. (spec Phase 2, §33.)

---

## 3. Increment B — Quality loop + A/B test mode (spec §2, §32, §33)

Per spec §32, don't blind-swap the engine/config. Add a **test mode** that runs
the *same* source images two ways and shows them side by side:

- **A/B option:** "Old config" (previous defaults) vs "New Corporate Trainer
  Profile config". Both use the committed Kaggle engine — the comparison is of
  **prompt/params/aspect**, not of two different engines.
- Implementation: a lightweight `?compare=1` job variant that launches two jobs
  (old-cfg, new-cfg) over the same uploads and renders a paired gallery. Reuses
  the existing `rerun` plumbing (`server.py:1862`) — minimal new code.
- **Gate:** validate identity / glasses / clothing / bg / framing / quality
  (spec §28 checklist) on **1**, then **10** representative trainers. Only
  proceed to batch-100 when the 10-set is consistently good (spec Phase 3, §33).

---

## 4. Increment C — Batch-100 operations (spec §15–§27, §30)

These are **local app** features (server + DB + a new reporting module); they are
engine-independent.

### C1. Batch size (spec §15)
- `MAX_FILES` 50 → **100** (`server.py:50`). Keep `MAX_FILE_BYTES` (25 MB) and
  raise `MAX_TOTAL_BYTES` if 100 × large files exceed 400 MB (bump to ~600 MB).
- Batches modeled as normal jobs tagged `batch_NN`; the 1,500-image project = 15
  jobs. No requirement to upload all 1,500 at once (spec §15).

### C2. Batch validation report (spec §16 — "do not silently discard")
- Today invalid files are silently `continue`d (`server.py:1055`, `:1068`).
- Add a pre-launch validation pass returning
  `{selected, valid, invalid[], duplicate[]}` (unsupported format, corrupt image
  via `PIL.verify()`, zero-byte, duplicate filename). Surface it on the new-job
  page before the job starts; nothing is dropped without being reported.

### C3. Processing queue (spec §17)
- Add `MAX_CONCURRENT_JOBS` (default **2**, configurable via env) enforced in
  `app/runner.py` with a `threading.Semaphore` around `start_job`. Extra jobs
  sit in a `queued` state (already a valid DB status) and start as slots free.
- Note: within a single kernel, images are already serial — this gates
  concurrent *kernels*, which is the real GPU-contention risk (spec §17 intent).

### C4. Progress + per-image states (spec §18, §19)
- Live: keep the stage stepper (Prepare/Upload/Launch/Generate/Download) which
  already updates via polling `/jobs/{id}/status` (`server.py:1101`).
- Per-image states `QUEUED/PROCESSING/COMPLETED/REVIEW/APPROVED/REJECTED/FAILED/
  REPROCESSING`: resolve from `results.json` + the `reviews` table on completion.
  Add a small state-derivation helper; render a per-image status grid on the job
  page. (Mid-kernel per-image streaming is not possible on Kaggle — see §1.3.)

### C5. Reprocess with attempt tracking (spec §21, §22)
- `reroll` already re-runs rejected/failed **from the original** as a new job
  (`server.py:1831`) — matches §21's "use the original image again". Extend:
  - New DB table `attempts(job_id, stem, attempt, ts, status)` (migration in
    `db.init()`), incremented on each reprocess.
  - After `MAX_ATTEMPTS` (default 3) mark the image **MANUAL REVIEW REQUIRED**.
  - Per-image `[RETRY]` on failures; one failure never stops the batch (already
    true — `worker:761` catches per-image, §22 satisfied).

### C6. Output folder structure (spec §24)
- On completion, materialize:
  ```
  outputs/batch_NN/approved/   review/   failed/
  ```
  by copying (never moving/overwriting) from `jobs/<id>/output/headshots_out/`
  according to review decisions. Originals stay in `jobs/<id>/input/` untouched
  (spec §24 "never overwrite originals").

### C7. Downloads + CSV (spec §25, §26)
- Existing: Download Approved / Download All (`server.py:1781`). Add **Download
  Failed Report** and a per-batch `processing_report.csv` with columns:
  `filename, batch_id, status, attempts, generation_time, output_filename, error,
  review_status`. New module `app/reporting.py`.
- Primary artifact name: `Batch_NN_Approved.zip`.

### C8. Dashboard (spec §27)
- New `/dashboard` route: totals (Total/Processed/Approved/Review/Rejected/
  Failed/Remaining) across all `batch_*` jobs + a per-batch status table
  (COMPLETED / PROCESSING / NOT STARTED) for batches 01–15.

### C9. Usage/cost tracking (spec §30)
- Kaggle GPU is free, so "cost" = usage counters per batch: images, successful,
  retries, failed, approx GPU minutes (from `run.log` timing). Show at job/batch
  level. No paid-API accounting needed on this engine.

---

## 5. Spec coverage matrix

| Spec § | Requirement | Increment | Notes |
|---|---|---|---|
| §3 | Preserve existing features | all | No removals; only additive/config |
| §4, §35 | Diagnose current pipeline | done | §1 above |
| §5, §6 | Identity preservation | A | Best-effort within InstantID ceiling (§1.2) |
| §7 | New Corporate Trainer Profile style + default | A | A2, A4 |
| §8 | Master Prompt | A | Compressed to 77 tokens (A2) |
| §9 | Negative prompt | A | `_NEG` + appended tail |
| §10 | Remove branding | A | Negatives + no auto-logo |
| §11 | 35:45 + preserve 1:1/4:5 | A | Native 896×1152 render (A3) |
| §12 | Highest practical resolution + safe upscale | A | Real-ESRGAN, ratio-aware |
| §13, §14 | Preset visibility + centralized config | A | `app/styles.py`, cfg summary |
| §15 | Batch max 100 | C1 | MAX_FILES→100 |
| §16 | Batch validation | C2 | Report, no silent discard |
| §17 | Queue + MAX_CONCURRENT_JOBS | C3 | Semaphore, default 2 |
| §18 | Progress | C4 | Stage-live; per-image on completion |
| §19 | Per-image states | C4 | Derived from results.json + reviews |
| §20 | Review system | exists | Approve/Reject/Reprocess present |
| §21 | Reprocess from original + attempts | C5 | Extend `reroll` + attempts table |
| §22 | Failed handling | exists+C5 | Per-image try/except already |
| §23 | Filename preservation | exists | Output stem = source stem ✅ |
| §24 | Output folder structure | C6 | `outputs/batch_NN/...` |
| §25 | Batch download | C7 | +Failed report, batch ZIP name |
| §26 | CSV report | C7 | `app/reporting.py` |
| §27 | Dashboard | C8 | `/dashboard` |
| §28 | Quality control checklist | B | Human review gates (no auto-QA claim) |
| §29 | Review before publish | exists | No auto-publish path exists |
| §30 | Usage tracking | C9 | Free GPU → usage counters |
| §31 | Dev phases | B | 1→10→100→1500 |
| §32 | Old-vs-new test mode | B | A/B compare |
| §33 | Success criteria | B | Validation gates |

---

## 6. Files touched (summary)

| File | Change |
|---|---|
| `app/styles.py` *(new)* | Canonical `CORPORATE_TRAINER_PROFILE` + style metadata |
| `worker/headshot_worker.py` | New `STYLE_PRESETS` entry; native 35:45 render + keypoint-canvas fix; ratio-aware upscale |
| `app/server.py` | `STYLES`/`STYLE_CAT`/`BRAND_PACKS`/`ASPECTS` additions; default→trainer style; aspect validation; MAX_FILES→100; validation report; `/dashboard`; CSV/failed-report downloads; per-image status grid; active-config display |
| `app/runner.py` | `MAX_CONCURRENT_JOBS` semaphore/queue |
| `app/db.py` | `attempts` table + migration; batch tagging |
| `app/reporting.py` *(new)* | `processing_report.csv` + `outputs/batch_NN/` materialization |
| `tests/` | Unit tests: prompt token-count guard, aspect math, validation report, CSV schema, attempt/queue logic |

No changes to `app/api_engine.py` (Gemini path left intact but unused for this work).

---

## 7. Testing & rollout (spec §31 Phases 2–5)

1. **Phase 2 (1 image):** trainer photo → new style; eyeball vs reference.
2. **Phase 3 (10 images):** A/B old-vs-new; score §28 checklist; iterate prompt/
   params (identity scale, guidance) only if needed.
3. **Phase 4 (100 = Batch 01):** exercise validation, queue, progress, failures,
   retry, review, ZIP, CSV.
4. **Phase 5 (1,500):** 15 batches; validate cost/quality on Batch 01 before
   scaling (spec §30 "do not process all 1,500 before validating").

Token-budget regression test: assert the trainer prompt encodes to ≤77 CLIP
tokens so a future edit can't silently truncate the wardrobe/background.

---

## 8. Open questions before coding

1. **Live deployment engine:** the Render instance uses whichever engine its env
   vars select. To use the Kaggle engine there, `KAGGLE_USERNAME`/`KAGGLE_KEY`
   must be set and no image-API key present. Confirm the Render service is on the
   Kaggle path (or whether this upgrade targets local runs first).
2. **Identity ceiling acceptance:** OK to accept "clearly recognizable" (not
   pixel-identical glasses/skin) as the §33 bar for the Kaggle engine? If
   pixel-faithful glasses are mandatory, that's the one thing this engine can't
   guarantee and we'd revisit the engine choice.
3. **Logo tool:** hide the manual logo panel when the trainer style is active
   (§10), or leave it available for other styles?
4. **Batch identity averaging:** trainers are distinct people, so per-batch
   `multi_reference` embedding averaging must be **off** for batches (it's meant
   for multiple photos of *one* person). Confirm — I'll default it off for
   batch jobs.
