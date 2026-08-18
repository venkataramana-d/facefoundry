# Bulk Profile Pic → Professional Headshot

Free, GPU-powered pipeline that converts low-quality profile pics into professional headshots using SDXL + InstantID for face preservation. Runs entirely on Kaggle's free T4 GPU. Zero API cost.

**Built for:** internal batch processing of ~1,500 images.
**Time:** ~6–8 hours for 1,500 images on a single Kaggle session.
**Cost:** $0.

---

## What's in this folder

| File | Purpose |
|---|---|
| `PLAN.md` | **Detailed strategic plan** — architecture, phased execution, risks, success criteria. Read this first. |
| `notebook.py` | The full processing pipeline. Paste into a Kaggle notebook. |
| `urls_template.csv` | Example format for your input CSV. |
| `README.md` | This file — quick reference for setup + tuning. |

---

## One-time setup (~15 minutes)

### 1. Create a Kaggle account
- Go to [kaggle.com](https://kaggle.com) → Sign up (free).
- **Verify your phone number** in Account settings. This is required to unlock GPU access — Kaggle blocks GPU for unverified accounts.

### 2. Prepare your URLs CSV
- Make a CSV with one column named `url`, one image URL per row. See `urls_template.csv` for the format.
- URLs must be publicly reachable (Kaggle's servers download them). Google Drive share links, S3 public URLs, CDN URLs all work.

### 3. Upload as a Kaggle Dataset
- Kaggle sidebar → **Create → New Dataset** → upload your CSV → set visibility to **Private** → publish.
- Note the dataset path (e.g. `/kaggle/input/my-headshot-urls/urls.csv`).

### 4. Create a new Notebook
- **Create → New Notebook**.
- Right sidebar → **Settings**:
  - Accelerator: **GPU T4 x2**
  - Internet: **ON**
  - Persistence: **Files only**
- Right sidebar → **Add Data** → attach the dataset you uploaded.

### 5. Paste in the code
- Open `notebook.py` from this folder.
- Split it into cells at the `# %% CELL N:` markers (or paste it all as one cell — works either way).
- Update `INPUT_CSV_PATH` in Cell 2 to match your dataset path.

---

## Running the pipeline

Always run in three stages. Do NOT jump straight to the full 1,500.

### Stage 1: Sample of 10 (~5 minutes)
```python
SAMPLE_MODE = "sample_10"
```
Runs cells 1–10 top to bottom. Cell 10 shows a side-by-side preview grid. **Check every one.** If faces look wrong, tune before scaling up:

- Faces don't match source → raise `IDENTITY_SCALE` (try 0.9)
- Faces look plasticky or over-processed → lower `IDENTITY_SCALE` (try 0.7)
- Clothing/background wrong → tweak the prompt in `STYLE_PRESETS`
- Wrong style → switch `STYLE_PRESET` to another preset

### Stage 2: Sample of 50 (~20 minutes)
```python
SAMPLE_MODE = "sample_50"
```
Catches edge cases the 10-sample missed: glasses, dark backgrounds, side profiles, low-res sources, group photos. Fix any issues before the full pass.

### Stage 3: Full batch (~6–8 hours)
```python
SAMPLE_MODE = "full"
```
Set this, then run **Run All** from the top menu. Kaggle sessions cap at 12 hours; 1,500 images fits comfortably. The batch loop is resumable — if the session dies at image 900, restart the notebook and it skips already-done images.

When done, Cell 9 creates a ZIP in `/kaggle/working/`. Right sidebar → **Output** → right-click the ZIP → **Download**.

---

## Style presets

Set `STYLE_PRESET` in Cell 2 to one of these:

| Preset | Look |
|---|---|
| `corporate` | Neutral gray studio background, charcoal suit, crisp white shirt, soft studio light. **Safest default for a mixed batch.** |
| `modern_tech` | Soft office bokeh, smart casual (sweater or open-collar), natural window light, friendly. |
| `warm_friendly` | Cream/beige background, cozy knit, golden hour lighting, subtle smile. |
| `formal_executive` | Deep navy background, tailored dark suit + tie, rembrandt lighting, authoritative. |

You can also edit the `prompt` and `negative` strings inside `STYLE_PRESETS` directly if you want a custom look.

---

## Tuning knobs (Cell 2)

| Setting | Range | Effect |
|---|---|---|
| `IDENTITY_SCALE` | 0.6–1.0 | How strongly to preserve the source face. Higher = more face fidelity, less style freedom. |
| `ADAPTER_SCALE` | 0.6–1.0 | How much of the source face's appearance (skin, features) carries through. Usually match `IDENTITY_SCALE`. |
| `NUM_STEPS` | 25–35 | Inference steps. 30 is the sweet spot. Under 25 = artifacts, over 35 = diminishing returns. |
| `GUIDANCE` | 4.0–7.0 | How closely to follow the prompt. 5.0 is balanced. Higher = more prompt-adherent but less natural. |
| `IMG_SIZE` | 1024 | Output size in pixels. SDXL is native at 1024. Larger = more VRAM. |

---

## Troubleshooting

**"CUDA out of memory"**
Reduce `IMG_SIZE` to 896 or enable more memory savings by adding `pipe.enable_model_cpu_offload()` after loading the pipeline in Cell 5.

**"No face detected" errors**
Some source images may be too small, blurry, or not have a clear face. These are logged in `gen_failures`. Manually review and either replace those source images or accept them as skipped.

**Kaggle session disconnects at ~1 hour**
This happens if the browser tab goes idle. Keep the tab open in the foreground, or use Kaggle's mobile app to keep the session alive.

**Downloaded ZIP is missing images**
Check `gen_failures` list — those images weren't generated. The ZIP only contains successful outputs.

**Some URLs fail to download**
Some CDNs block datacenter IPs (Kaggle's servers). Check the `failures` list from Cell 6. Re-host those images somewhere accessible (Google Drive public link, S3 public URL, imgbb.com) and re-run.

**Kaggle says "Weekly GPU quota exceeded"**
You get 30 GPU-hours/week free. If you've used them, wait for the reset (visible in Kaggle account settings).

---

## Quality expectations

- **Best results**: source photo shows a clear frontal face, no heavy shadows, decent lighting. Output is near-professional-studio quality.
- **Decent results**: source is slightly blurry, off-angle, or poorly lit. Output usable, may need cherry-picking.
- **Poor results**: source has multiple faces, heavy occlusion (masks, hands over face), extreme angles, or resolution below ~200px. Output may look different from the person.

For the ~5–10% of images that come out wrong, re-run them individually with a different `seed` value (Cell 7: change `seed=42 + idx` to a different offset) — often a re-roll fixes it.

---

## Cost recap

| Item | Cost |
|---|---|
| Kaggle account | Free |
| Kaggle GPU time (30 hrs/wk) | Free |
| SDXL, InstantID, antelopev2 models | Free (open weights) |
| Storage for outputs | Free (Kaggle working dir) |
| **Total for 1,500 headshots** | **$0** |

---

## If Kaggle isn't working out

Fallbacks if you hit repeated quota / session issues:

- **Colab Pro** ($10/mo): more reliable sessions, higher quotas. Same notebook works with minor path changes.
- **Replicate API** (~$0.03/image = ~$45 for 1,500): swap Cells 4–8 for a simple `replicate.run()` call using the `zsxkib/instant-id` model. Instant scale, no GPU management.
- **RunPod** ($0.30/hr for T4): rent a GPU, run the notebook there. ~$3–4 total for 1,500 images.
