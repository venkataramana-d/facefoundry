# Detailed Plan — Bulk Profile Pic → Professional Headshot

**Owner:** Edstellar internal team
**Volume:** 1,500 low-quality profile pics
**Deadline:** Not set — one-time batch job
**Budget:** $0 (free-tier only)
**Last updated:** 2026-08-18

---

## 1. Goal

Convert 1,500 low-quality profile pictures (varied source, one face per photo) into consistent, professional-looking headshots suitable for internal directory / marketing / LinkedIn-style use. Preserve each person's identity while regenerating clothing, background, lighting, and grooming.

---

## 2. Why this approach

| Route considered | Decision | Reason |
|---|---|---|
| Paid API (Replicate/fal.ai) | ❌ Rejected | User requires $0 cost. Paid route would be ~$45 for the batch. |
| Self-hosted on user's PC | ❌ Rejected | User has Intel Iris Xe (integrated graphics, no dedicated VRAM). Can't run SDXL locally. |
| Colab free tier | ❌ Rejected | Aggressive idle disconnects, unreliable for 8-hour batch. |
| Hugging Face Spaces free | ❌ Rejected | Rate-limited, queued, not viable for bulk. |
| **Kaggle notebooks (free)** | ✅ **Chosen** | 30 hrs/wk free GPU (T4 16GB), 12h sessions (fits our ~8h batch in one run), no idle disconnect, reliable, reproducible. |

**Model choice:** SDXL 1.0 + InstantID (ControlNet + IP-Adapter for face identity). SDXL fits on a T4 with fp16. InstantID is currently the strongest open-source face-preservation adapter — better identity fidelity than IP-Adapter FaceID or PhotoMaker for professional headshot use.

---

## 3. Architecture

```
┌──────────────────┐
│  urls.csv        │  (uploaded to Kaggle Dataset)
│  (1500 rows)     │
└────────┬─────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│  Kaggle Notebook (GPU T4 x2, Internet ON)                  │
│                                                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Cell 1-4     │───▶│ Cell 5       │───▶│ Cell 6       │  │
│  │ Install +    │    │ Load pipeline│    │ Download     │  │
│  │ download     │    │ (SDXL +      │    │ source imgs  │  │
│  │ models       │    │  InstantID)  │    │ (16 threads) │  │
│  └──────────────┘    └──────────────┘    └──────┬───────┘  │
│                                                 │          │
│                                                 ▼          │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Cell 8 — Batch loop (resumable)                    │    │
│  │  for each source image:                            │    │
│  │    1. Face detect (antelopev2)                     │    │
│  │    2. Extract face embedding + keypoints           │    │
│  │    3. SDXL + InstantID inference (~20s on T4)      │    │
│  │    4. Save 1024x1024 JPG                           │    │
│  │  Progress logged every 10 images                   │    │
│  └────────────────────┬───────────────────────────────┘    │
│                       │                                    │
│                       ▼                                    │
│  ┌──────────────────────────────────────────────────┐      │
│  │ Cell 9 — Zip outputs                              │      │
│  │ Cell 10 — Optional side-by-side preview grid      │      │
│  └──────────────────────────────────────────────────┘      │
└────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  headshots_<style>_<mode>.zip│  (downloaded from Kaggle Output)
└──────────────────────────────┘
```

**Data flow:** URL list → downloaded JPEGs in `/kaggle/working/downloads/` → per-image face extraction → generated headshots in `/kaggle/working/headshots_out/` → zipped for download.

---

## 4. Basic setup checklist

Complete these once before the first run. Estimated total time: **~15 minutes**.

### 4.1 Kaggle account
- [ ] Sign up at [kaggle.com](https://kaggle.com) with email (free)
- [ ] **Verify phone number** in Account Settings → Phone Verification
      _(Required to unlock GPU access. Non-negotiable — no GPU without this.)_
- [ ] Confirm quota shows "30 hours/week" of accelerator time

### 4.2 Prepare input data
- [ ] Collect all 1,500 image URLs into a single CSV
- [ ] CSV must have exactly one column named `url` (see `urls_template.csv`)
- [ ] Verify URLs are publicly reachable (open one in a private browser window — if it loads without login, Kaggle can fetch it)
- [ ] **Common gotchas:**
      - Google Drive: use `https://drive.google.com/uc?id=FILE_ID` format, not the share URL
      - S3: bucket must have public read on the objects, or use pre-signed URLs
      - Dropbox: append `?dl=1` to force direct download
      - Any URL requiring cookies/auth won't work

### 4.3 Upload dataset to Kaggle
- [ ] Kaggle → sidebar → **Create → New Dataset**
- [ ] Drag in `urls.csv`
- [ ] Set **Visibility: Private**
- [ ] Give it a slug like `headshot-urls` — note the full path (e.g. `/kaggle/input/headshot-urls/urls.csv`)
- [ ] Click **Create**

### 4.4 Create the notebook
- [ ] Kaggle → sidebar → **Create → New Notebook**
- [ ] Right sidebar → **Settings**:
      - Accelerator: **GPU T4 x2**
      - Internet: **ON**
      - Persistence: **Files only** (checkpoints survive across sessions)
      - Environment: **Pin to original environment** (avoids surprise breakage)
- [ ] Right sidebar → **Add Data** → search your dataset name → attach
- [ ] Confirm the dataset shows under `/kaggle/input/` in the file browser

### 4.5 Load the code
- [ ] Open `notebook.py` from this folder
- [ ] Copy contents into the Kaggle notebook
- [ ] Split into cells at `# %% CELL N:` markers (Kaggle: click cell → **Insert → Add cell below**, paste next section)
      _Alternative: paste it all into one cell — works but harder to iterate on._
- [ ] In Cell 2, update `INPUT_CSV_PATH` to match your dataset path

---

## 5. Phased execution plan

Do **not** run the full 1,500 as the first pass. Follow this three-stage approach — it's saved every batch job like this from expensive full re-runs.

### Phase 1 — Sanity check on 10 images (~10 min)

**Config:**
```python
SAMPLE_MODE = "sample_10"
STYLE_PRESET = "corporate"   # start with the safest preset
```

**What to do:**
1. Run cells 1–10 top to bottom (first run: model downloads add ~10 min)
2. Cell 10 renders a side-by-side preview grid
3. Inspect every output:
   - Is the face recognizably the same person?
   - Is the style consistent across the 10?
   - Any artifacts (extra fingers, weird backgrounds, distorted features)?

**Decision gate:**
- ✅ 8+/10 look good → proceed to Phase 2
- ⚠️ 5–7/10 look good → tune `IDENTITY_SCALE` / prompt, re-run Phase 1
- ❌ <5/10 look good → step back, consider switching preset or model

**Common Phase 1 tuning:**
| Symptom | Fix |
|---|---|
| Face looks like a different person | Raise `IDENTITY_SCALE` to 0.9 |
| Face looks plasticky / over-smoothed | Lower `IDENTITY_SCALE` to 0.7 |
| Clothing/background wrong | Edit prompt in `STYLE_PRESETS` |
| Every output has same weird artifact | Change `seed=42 + idx` offset |

### Phase 2 — Edge case sweep on 50 images (~25 min)

**Config:**
```python
SAMPLE_MODE = "sample_50"
```

**What to do:**
1. Change `SAMPLE_MODE`, run cells 6–10 (skip re-loading models)
2. Sample of 50 catches edge cases the 10 missed:
   - Glasses (InstantID sometimes drops them)
   - Very dark or very light skin tones
   - Beards / facial hair
   - Side profiles / non-frontal angles
   - Low-resolution sources (<400px)
   - Group photos where wrong face was picked
3. Flag any consistent failure pattern

**Decision gate:**
- ✅ 45+/50 look good → proceed to Phase 3
- ⚠️ Systematic issue on a subgroup (e.g. all glasses-wearers fail) → adjust prompt (add "wearing glasses" when detected) or accept the tradeoff
- ❌ Widespread quality regression vs Phase 1 → the source pool has issues Phase 1 didn't sample; investigate

### Phase 3 — Full batch of 1,500 (~6–8 hours)

**Config:**
```python
SAMPLE_MODE = "full"
```

**What to do:**
1. Change `SAMPLE_MODE`, click **Run All** from top menu
2. Monitor first 15 minutes to confirm rate is ~2–3 images/min
3. Leave running — keep the browser tab open (avoid idle disconnect)
4. Batch loop is resumable: if session dies partway, restart the notebook and it skips completed images

**Expected timings:**
- Model load (Cells 1–5): ~15 min first time, ~3 min on cached restart
- Image downloads (Cell 6): ~5–10 min for 1,500 URLs
- Generation (Cell 8): ~20s/image × 1,500 = **~500 min = ~8.3 hours**
- Zip + download (Cell 9): ~5 min

**End state:**
- ZIP file in `/kaggle/working/` named `headshots_corporate_full.zip`
- Right sidebar → Output → download the ZIP
- `gen_failures` list shows which images didn't generate (typically 3–8% of a mixed batch)

---

## 6. Success criteria

- [ ] **Coverage:** 90%+ of 1,500 source images produce a usable headshot
- [ ] **Identity:** Faces are recognizably the same person as the source
- [ ] **Consistency:** Chosen style (background, attire, lighting) is uniform across the batch
- [ ] **Cost:** $0 spent
- [ ] **Time:** Complete within 2–3 calendar days (one Kaggle session for the batch + one for tuning)

---

## 7. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Kaggle session times out mid-batch | Low | Medium | Batch loop is resumable — just restart the notebook |
| Source URLs blocked from Kaggle IPs | Medium | Medium | Cell 6 logs failures; re-host blocked URLs on Google Drive |
| Face detection fails on some source images | Medium | Low | Logged in `gen_failures`; can manually retry with different seed |
| Quality varies across the batch | High | Medium | Phase 1 + Phase 2 tuning catches most cases before Phase 3 |
| Weekly GPU quota exhausted during tuning | Low | High | Quota resets weekly; spread tuning + batch across two weeks if hit |
| Kaggle changes free-tier policy | Very low | High | Fallback: Colab ($10/mo) or Replicate API ($45 one-time) |

---

## 8. Cost recap

| Item | Cost |
|---|---|
| Kaggle account + GPU (30hr/wk) | $0 |
| SDXL, InstantID, antelopev2 model weights | $0 (open weights) |
| Storage (Kaggle working dir) | $0 |
| **Total for 1,500 headshots** | **$0** |

If we ever need to scale beyond Kaggle's free tier:
- **Replicate API:** ~$0.03/image → $45 per 1,500
- **RunPod T4 rental:** ~$0.30/hour → ~$3 per 1,500 (needs same notebook, minor path changes)
- **Colab Pro:** $10/month, similar to Kaggle experience with fewer limits

---

## 9. Post-batch checklist

After the ZIP is downloaded:

- [ ] Spot-check 20 random outputs from the ZIP
- [ ] Review `gen_failures` list — decide whether to re-run those with a different seed or accept as skipped
- [ ] Archive `notebook.py` version used for this batch (in case results need reproducing)
- [ ] Note the total generation time + cost for future planning
- [ ] If quality is inconsistent, consider a second pass with a stricter identity scale on the underperformers only

---

## 10. Open questions / decisions deferred

- **Style preset for the batch:** Defaulted to `corporate`. Confirm this matches Edstellar's brand guidelines before Phase 3.
- **Handling of failures:** Currently skipped and logged. If we want 100% coverage, need a manual review + re-run pass (~30 min for a typical 5% failure rate).
- **Storage of outputs:** Currently just downloaded as ZIP. If we need shared team access, decide on final destination (SharePoint, Google Drive, S3, internal directory tool).
- **Reusability:** This notebook is a one-shot for the 1,500. If Edstellar wants it as a repeatable tool (new hires monthly), a small web UI wrapper is a future project — separate scope.
