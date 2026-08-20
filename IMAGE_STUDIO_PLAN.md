# FaceFoundry — "Image Settings" Studio + Corporate-Quality Plan

Goal: match the corporate reference look (navy suit, **pure white background**, crisp
studio lighting) and give a **post-render image-settings editor** so each headshot can
be fine-tuned and branded (e.g. Edstellar logo) before download.

## What the reference images tell us
- Pure **white** background (not studio grey) — biggest single differentiator.
- Even, bright key light; slight sharpening; neutral-to-cool white balance.
- Consistent framing (head-and-shoulders, centered).
- Optional **brand logo** top-right.

So two workstreams: (A) make generation land closer to that, (B) a browser editor to
finish each image.

---

## Researched building blocks (to integrate)

| Need | Pick | License | Why |
|---|---|---|---|
| Background → white | [`@imgly/background-removal`](https://github.com/imgly/background-removal-js) (in-browser ONNX/WASM) | AGPL | Runs client-side, no server/GPU, private |
| Background → white (server option) | [`rembg`](https://github.com/danielgatis/rembg) (Python u2net) | MIT | Batch, offline |
| Crop / zoom | [Cropper.js](https://github.com/fengyuanchen/cropperjs) or custom canvas | MIT | Small, dependency-light |
| Face restore | GFPGAN (already wired, fail-safe) | Apache-2 | Sharper eyes/skin |
| Upscale | Real-ESRGAN (bundled w/ GFPGAN) or Lanczos (current) | BSD | 4K delivery |

Reference pipelines reviewed: `lucataco/proHeadshot`, `astriaai/headshots-starter`,
`gnaaruag/foss-headshot-generator` — mostly SaaS wrappers; our InstantID core is stronger.
The reusable idea from them is **post-processing** (bg cleanup + framing), captured below.

---

## Phase 1 — Post-render Image-Settings editor  (client-side, no GPU)  ← building now
A dedicated `/jobs/{id}/edit/{stem}` page: canvas preview + settings panel.

- **Adjust**: brightness, contrast, saturation, warmth (live, canvas filters).
- **Crop & reframe**: aspect presets (1:1, 4:5, 3:4), zoom + drag-to-pan.
- **Background**: one-click **Remove background** (`@imgly`) → fill with a chosen color
  (white default) for the reference look. Graceful fallback if the model can't load.
- **Logo overlay**: upload a logo, drag to place, scale/opacity (for the Edstellar mark).
- **Export**: choose size (1K/2K/4K) + format (JPG/PNG), download. All in-browser.

Entry point: an **Edit** button on each review tile.

## Phase 2 — Better generation defaults (worker)
- Add a **"pure white background"** style variant + push the corporate prompt toward
  the reference (bright softbox, white seamless, cool WB).
- Optional **server-side rembg** pass so batch output is white without manual editing.
- Turn **GFPGAN on by default** for the corporate styles once validated on GPU.

## Phase 3 — Batch image-settings
- Apply one editor preset (crop + adjust + bg + logo) to a **whole job** at once.
- Save presets ("Edstellar corporate") and reuse across jobs.

## Phase 4 — Speed
- Model-cache dataset (plumbing already in) to cut the ~10–15 min setup.

---

## Status — ALL PHASES COMPLETE
- **Phase 1** ✅ single-image editor (`/jobs/{id}/edit/{stem}`): crop/zoom/pan, brightness/
  contrast/saturation/warmth, in-browser background removal + fill color, logo overlay,
  export 1K/2K/4K JPG/PNG. Verified live on a real headshot.
- **Phase 2** ✅ worker `white_background` (fail-safe rembg → pure white) + `face_enhance`
  default on + UI toggles.
- **Phase 3** ✅ batch editor (`/jobs/{id}/edit`): configure once, **apply to all → ZIP**
  (JSZip), save/load **presets** (localStorage), optional per-image background removal.
- **Phase 4** ✅ model-cache: worker loads from an attached `facefoundry-models` dataset;
  orchestrator auto-attaches it; **`python app/kaggle_client.py --build-cache`** creates it
  once to cut the ~10-15 min download.

Worker-side additions (GFPGAN, rembg, model-cache) are **fail-safe** — if an optional
dep can't install, that feature is skipped and the job still succeeds.

Validated: the core pipeline generates real headshots (torch/P100 fix confirmed,
`ok=1 failed=0`). Editors verified live. All routes/options covered by the test suite.
