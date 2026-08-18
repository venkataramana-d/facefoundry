# =============================================================================
# BULK PROFILE PIC -> PROFESSIONAL HEADSHOT (Kaggle notebook, free tier)
# -----------------------------------------------------------------------------
# Runtime: Kaggle Notebook, Accelerator = GPU T4 x2 (or P100). Internet: ON.
# Cost: $0. Time for 1500 images: ~6-8 hours on T4 (well within 12h session).
#
# HOW TO USE:
#   1) Create a Kaggle account (free) and start a new Notebook.
#   2) Right sidebar -> Settings: Accelerator = "GPU T4 x2", Internet = ON.
#   3) Upload your URLs.csv as a Kaggle Dataset (one column named "url",
#      one URL per row). Attach the dataset to the notebook.
#   4) Update INPUT_CSV_PATH below to point to your dataset (e.g.
#      "/kaggle/input/headshot-urls/urls.csv").
#   5) Split this file into cells by the "# %% CELL:" markers, or paste it
#      as one big cell. Run top to bottom.
#   6) First run: leave SAMPLE_MODE = "sample_10" to verify quality on 10
#      images. Then flip to "sample_50", then "full" for the real bulk pass.
# =============================================================================


# %% CELL 1: Install dependencies (~2-3 min)
# -----------------------------------------------------------------------------
!pip install -q diffusers==0.27.2 transformers==4.39.3 accelerate==0.29.2
!pip install -q insightface==0.7.3 onnxruntime-gpu==1.17.1
!pip install -q opencv-python-headless controlnet-aux==0.0.7
!pip install -q huggingface_hub==0.22.2

# Clone InstantID for its custom SDXL pipeline module
!git clone -q https://github.com/InstantID/InstantID.git /kaggle/working/InstantID
import sys
sys.path.append("/kaggle/working/InstantID")


# %% CELL 2: Config — edit these
# -----------------------------------------------------------------------------
INPUT_CSV_PATH = "/kaggle/input/headshot-urls/urls.csv"  # <-- change to your dataset path
URL_COLUMN     = "url"           # column name in the CSV

STYLE_PRESET   = "corporate"     # one of: corporate, modern_tech, warm_friendly, formal_executive
SAMPLE_MODE    = "sample_10"     # one of: sample_10, sample_50, full

OUTPUT_DIR     = "/kaggle/working/headshots_out"
DOWNLOAD_DIR   = "/kaggle/working/downloads"
IMG_SIZE       = 1024            # SDXL native
NUM_STEPS      = 30              # 25-35 is the sweet spot
GUIDANCE       = 5.0
IDENTITY_SCALE = 0.8             # InstantID face-preservation strength (0.6-1.0)
ADAPTER_SCALE  = 0.8             # InstantID appearance strength

STYLE_PRESETS = {
    "corporate": {
        "prompt": "professional corporate headshot, {person}, neutral light gray studio background, tailored charcoal business suit, crisp white shirt, soft key light with subtle rim light, sharp focus on eyes, shallow depth of field, high detail skin texture, photorealistic, shot on 85mm lens",
        "negative": "cartoon, anime, illustration, painting, 3d render, blurry, low quality, distorted, deformed face, extra limbs, watermark, text, logo, oversaturated, harsh shadows",
    },
    "modern_tech": {
        "prompt": "professional headshot, {person}, softly blurred modern office background with warm bokeh, smart casual attire crew neck sweater or open collar shirt, natural window light, friendly confident expression, photorealistic, shot on 50mm lens",
        "negative": "cartoon, anime, illustration, painting, 3d render, blurry face, low quality, distorted, deformed, extra limbs, watermark, text, oversaturated",
    },
    "warm_friendly": {
        "prompt": "warm approachable professional headshot, {person}, soft cream beige background, cozy knit sweater in muted tone, golden hour lighting, genuine subtle smile, photorealistic, shot on 85mm lens",
        "negative": "cartoon, anime, illustration, painting, 3d render, blurry, low quality, distorted face, extra limbs, watermark, text, harsh light, cold tone",
    },
    "formal_executive": {
        "prompt": "formal executive portrait, {person}, deep navy blue background with subtle vignette, tailored dark suit and tie, dramatic rembrandt lighting, confident authoritative expression, photorealistic, shot on 85mm lens, editorial quality",
        "negative": "cartoon, anime, illustration, painting, 3d render, blurry, low quality, distorted face, extra limbs, watermark, text, casual, oversaturated",
    },
}


# %% CELL 3: Imports + setup
# -----------------------------------------------------------------------------
import os, csv, io, time, zipfile, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import cv2
import numpy as np
from PIL import Image
import torch
from huggingface_hub import hf_hub_download
from insightface.app import FaceAnalysis
from diffusers.models import ControlNetModel
from pipeline_stable_diffusion_xl_instantid import (
    StableDiffusionXLInstantIDPipeline,
    draw_kps,
)

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR = "/kaggle/working/checkpoints"
Path(CHECKPOINTS_DIR).mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "GPU not detected. In Kaggle: Settings -> Accelerator -> GPU T4 x2."
print(f"Device: {device} | {torch.cuda.get_device_name(0)}")


# %% CELL 4: Download models (~5-10 min first run, cached after)
# -----------------------------------------------------------------------------
print("Downloading InstantID ControlNet + IP-Adapter...")
hf_hub_download(repo_id="InstantX/InstantID",
                filename="ControlNetModel/config.json",
                local_dir=CHECKPOINTS_DIR)
hf_hub_download(repo_id="InstantX/InstantID",
                filename="ControlNetModel/diffusion_pytorch_model.safetensors",
                local_dir=CHECKPOINTS_DIR)
hf_hub_download(repo_id="InstantX/InstantID",
                filename="ip-adapter.bin",
                local_dir=CHECKPOINTS_DIR)

print("Downloading InsightFace antelopev2 model...")
# insightface auto-downloads on first FaceAnalysis() call, but its default
# mirror can be flaky; we prefetch to be safe.
os.makedirs("/root/.insightface/models", exist_ok=True)
!wget -q -O /kaggle/working/antelopev2.zip https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip
!mkdir -p /root/.insightface/models/antelopev2
!unzip -qo /kaggle/working/antelopev2.zip -d /root/.insightface/models/antelopev2

print("Models ready.")


# %% CELL 5: Load pipeline (~2 min, one-time)
# -----------------------------------------------------------------------------
face_app = FaceAnalysis(name="antelopev2",
                        root="/root/.insightface",
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))

controlnet = ControlNetModel.from_pretrained(
    f"{CHECKPOINTS_DIR}/ControlNetModel",
    torch_dtype=torch.float16,
)

pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16,
).to(device)
pipe.load_ip_adapter_instantid(f"{CHECKPOINTS_DIR}/ip-adapter.bin")
pipe.set_ip_adapter_scale(ADAPTER_SCALE)

# Small memory wins on T4
pipe.enable_vae_tiling()
pipe.enable_xformers_memory_efficient_attention() if hasattr(pipe, "enable_xformers_memory_efficient_attention") else None
print("Pipeline loaded.")


# %% CELL 6: Load URLs + download source images in parallel
# -----------------------------------------------------------------------------
def load_urls(csv_path, column):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row[column].strip() for row in reader if row.get(column, "").strip()]

def download_one(idx_url):
    idx, url = idx_url
    out = Path(DOWNLOAD_DIR) / f"src_{idx:05d}.jpg"
    if out.exists():
        return idx, str(out), None
    try:
        r = requests.get(url, timeout=30, stream=True)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        img.save(out, "JPEG", quality=92)
        return idx, str(out), None
    except Exception as e:
        return idx, None, f"{type(e).__name__}: {e}"

all_urls = load_urls(INPUT_CSV_PATH, URL_COLUMN)
print(f"Loaded {len(all_urls)} URLs.")

if   SAMPLE_MODE == "sample_10": urls = all_urls[:10]
elif SAMPLE_MODE == "sample_50": urls = all_urls[:50]
else:                            urls = all_urls
print(f"Processing {len(urls)} images in mode='{SAMPLE_MODE}'.")

t0 = time.time()
downloads = []
failures = []
with ThreadPoolExecutor(max_workers=16) as ex:
    futures = {ex.submit(download_one, (i, u)): i for i, u in enumerate(urls)}
    for fut in as_completed(futures):
        idx, path, err = fut.result()
        if err: failures.append((idx, err))
        else:   downloads.append((idx, path))
downloads.sort()
print(f"Downloaded {len(downloads)}/{len(urls)} in {time.time()-t0:.1f}s. "
      f"Failures: {len(failures)}")
if failures[:5]: print("First few failures:", failures[:5])


# %% CELL 7: Face-detect + generate headshot
# -----------------------------------------------------------------------------
preset = STYLE_PRESETS[STYLE_PRESET]
PROMPT   = preset["prompt"].replace("{person}", "a person")
NEGATIVE = preset["negative"]
print(f"Style: {STYLE_PRESET}\nPrompt: {PROMPT[:120]}...")

def process_image(src_path: str, dst_path: str, seed: int = 42):
    face_img = cv2.imread(src_path)
    if face_img is None:
        return False, "unreadable image"
    face_info = face_app.get(face_img)
    if not face_info:
        return False, "no face detected"
    # pick the largest face (area of bbox)
    face_info = sorted(face_info,
                       key=lambda x: (x["bbox"][2]-x["bbox"][0])*(x["bbox"][3]-x["bbox"][1]),
                       reverse=True)[0]
    face_emb = face_info["embedding"]
    face_kps = draw_kps(Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)),
                        face_info["kps"])

    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        image_embeds=face_emb,
        image=face_kps,
        controlnet_conditioning_scale=IDENTITY_SCALE,
        ip_adapter_scale=ADAPTER_SCALE,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE,
        width=IMG_SIZE,
        height=IMG_SIZE,
        generator=generator,
    ).images[0]
    result.save(dst_path, "JPEG", quality=94)
    return True, None


# %% CELL 8: Run the batch
# -----------------------------------------------------------------------------
gen_failures = []
t0 = time.time()
for i, (idx, src_path) in enumerate(downloads, 1):
    dst_path = f"{OUTPUT_DIR}/headshot_{idx:05d}.jpg"
    if os.path.exists(dst_path):
        continue  # already done — resumable
    try:
        ok, err = process_image(src_path, dst_path, seed=42 + idx)
        if not ok:
            gen_failures.append((idx, err))
    except Exception as e:
        gen_failures.append((idx, f"{type(e).__name__}: {e}"))
        traceback.print_exc()
    if i % 10 == 0 or i == len(downloads):
        elapsed = time.time() - t0
        rate = i / elapsed
        eta  = (len(downloads) - i) / rate if rate else 0
        print(f"[{i}/{len(downloads)}] elapsed {elapsed/60:.1f}m | "
              f"{rate*60:.1f} img/min | ETA {eta/60:.1f}m | "
              f"gen failures: {len(gen_failures)}")

print(f"Done. Generated {len(downloads) - len(gen_failures)} headshots.")
if gen_failures[:10]:
    print("First few generation failures:", gen_failures[:10])


# %% CELL 9: Zip outputs for download
# -----------------------------------------------------------------------------
zip_path = f"/kaggle/working/headshots_{STYLE_PRESET}_{SAMPLE_MODE}.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in sorted(Path(OUTPUT_DIR).glob("*.jpg")):
        z.write(p, arcname=p.name)
print(f"ZIP ready: {zip_path}")
print("Download from the right sidebar -> Output -> right-click the .zip -> Download.")


# %% CELL 10 (optional): Quick side-by-side preview grid for sample runs
# -----------------------------------------------------------------------------
from PIL import ImageDraw, ImageFont
def make_grid(pairs, cols=5, thumb=384):
    rows = (len(pairs) + cols - 1) // cols
    grid = Image.new("RGB", (cols*thumb*2, rows*thumb), "white")
    for i, (src, dst) in enumerate(pairs):
        r, c = divmod(i, cols)
        s = Image.open(src).convert("RGB").resize((thumb, thumb))
        d = Image.open(dst).convert("RGB").resize((thumb, thumb))
        grid.paste(s, (c*thumb*2, r*thumb))
        grid.paste(d, (c*thumb*2 + thumb, r*thumb))
    return grid

pairs = []
for idx, src in downloads[:10]:
    dst = f"{OUTPUT_DIR}/headshot_{idx:05d}.jpg"
    if os.path.exists(dst):
        pairs.append((src, dst))
if pairs:
    grid = make_grid(pairs)
    grid.save("/kaggle/working/preview_grid.jpg", quality=88)
    grid  # displays inline in Kaggle
