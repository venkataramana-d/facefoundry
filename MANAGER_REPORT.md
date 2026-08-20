# FaceFoundry — AI Bulk Headshot Tool
### Internal project status report

**Prepared by:** Ramana
**Date:** 18 Aug 2026
**Status:** ✅ Working — validated end-to-end

---

## Summary
I built **FaceFoundry**, an internal tool that turns ordinary employee photos into
professional, corporate-style headshots **in bulk** — at **$0 cost**, using a free
cloud GPU. It's working: I've generated a real headshot from a sample photo end-to-end.

## The problem it solves
Getting consistent, professional headshots for the whole team normally means a paid
photographer or a paid AI service (~$20–40 per person). For a large team that's
significant cost and coordination. FaceFoundry does it in-house, for free, in batches.

## What it does
1. Upload a **folder of photos** (or a whole team at once).
2. Pick a **style** (Corporate, LinkedIn, Executive, Healthcare, + 4 more).
3. The tool runs the AI on a **free GPU**, then shows a **before/after review** grid.
4. **Approve** the good ones, fine-tune in a built-in **image editor** (crop, lighting,
   **pure-white background**, **company logo overlay**), and **download** — up to **4K**.

## Key points
| | |
|---|---|
| **Cost** | **$0 per batch** (free Kaggle GPU; open-source AI models) |
| **Scale** | Built for bulk — dozens/hundreds of people per run |
| **Quality** | SDXL + InstantID (keeps the person's real face), optional face enhancement + 4K |
| **Branding** | Add the Edstellar logo + clean white background per image |
| **Status** | Core pipeline validated (real headshot generated successfully) |

## See it live
A working demo is currently accessible here (runs from my machine):
**<DEMO_LINK>**
*(temporary link for the demo — a permanent hosted version can be set up for team use)*

## What's next (with your go-ahead)
- **Host it permanently** so the team can use it anytime (a small always-on server).
- Add a **login** so access is controlled.
- Run a **pilot batch** of real team photos to confirm quality at scale.

---
*Built as an internal tool. Employee photos are processed privately and are not shared publicly.*
