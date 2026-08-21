# FaceFoundry - Match the Edstellar Reference Look

Goal: make the generated headshot match the reference (navy suit, light-blue shirt,
navy tie, **pure white** background, full **color**, crisp studio lighting, glasses
kept, optional Edstellar logo top-right).

## What the reference tells us
- Full-color, well-lit studio portrait (NOT desaturated/grey).
- Navy business suit, light-blue dress shirt, navy dotted tie.
- Pure white seamless background.
- Sharp detail, even soft-box lighting, neutral expression.
- Glasses and grey hair preserved (identity).
- Edstellar logo, top-right corner.
- Head-and-shoulders, centred.

## Why the current output looks grey
The `corporate` preset prompt says "charcoal suit, neutral light gray background,
soft key light" - that produces a muted, low-colour result. The reference is
brighter and more saturated.

---

## Plan (in priority order)

### 1. New "Edstellar Executive" style preset  (worker, biggest impact)
Add a preset tuned to the reference:
- Prompt: navy business suit, light-blue dress shirt, navy tie, **bright white
  seamless studio background**, even soft-box lighting, **natural vivid colour,
  colour photograph**, keep the person's eyeglasses, sharp focus, editorial
  corporate portrait, 85mm.
- Negative: **grayscale, monochrome, black and white, desaturated, muted, dark
  background**, casual clothing, blurry.
This fixes colour + wardrobe + background wording in one shot.

### 2. Pure white background ON by default for this preset
Run the fail-safe `rembg` pass so the background is truly pure white (not just
prompt-dependent), matching the reference exactly.

### 3. Colour / vibrance safeguard  (post-process)
After generation, apply a small saturation/contrast lift (PIL) so results never
come out flat/grey, plus the existing GFPGAN face-enhance ON by default.

### 4. Sharpness + resolution
Default to High (2K) or Ultra (4K) with face-enhance on, so detail matches the
reference crispness.

### 5. Identity / glasses preservation
Nudge identity_scale/adapter_scale up slightly so glasses, hair, and face carry
through reliably. Add "wearing eyeglasses" to the prompt.

### 6. Edstellar logo, top-right  (two options)
- (a) Now: use the built-in image editor's **logo overlay** (already works) -
  upload the Edstellar logo, drag to top-right, apply to all via Batch edit.
- (b) Later: bake the logo in the worker automatically when a logo file is
  attached (a cleaner, hands-off result). Proposed as a follow-up.

---

## How we test (localhost first, no live push)
The worker code is pushed to Kaggle fresh with every job, so testing from the
LOCAL control panel already exercises the new quality - no Render deploy needed.

1. Implement changes 1-5 in `worker/headshot_worker.py` (+ expose the new style
   in the UI). Do NOT push to GitHub/Render.
2. Run the local panel: `run.bat` (or uvicorn) -> New job -> pick "Edstellar
   Executive", 2K/4K, white background + enhance on -> upload the reference
   person's real photo -> Generate.
3. Review the result against the reference. Iterate on the prompt / settings.
4. Only once it looks right, push to GitHub -> Render (live).

## Status
Plan drafted. Awaiting go-ahead to implement changes 1-5 locally (no push).
