"""
FaceFoundry worker — runs ON Kaggle as a GPU script kernel.

This is the parameterized version of the original notebook.py. Instead of
hard-coded globals it reads a `job.json` (uploaded by the local orchestrator
alongside the source images in a private Kaggle dataset), processes every
image, and writes:
    /kaggle/working/headshots_out/*.jpg   the generated headshots
    /kaggle/working/results.json          structured per-image results
    /kaggle/working/headshots_<...>.zip   everything zipped for download

Input modes (set in job.json -> "input_mode"):
    "images" : process every image file found in the attached dataset(s)
    "urls"   : download from a urls.csv in the attached dataset, then process

job.json schema (all keys optional except input_mode):
{
  "job_id":        "abc123",
  "input_mode":    "images" | "urls",
  "url_column":    "url",                 # urls mode only
  "style_preset":  "corporate",
  "limit":         null,                  # cap number of images (null = all)
  "img_size":      1024,
  "num_steps":     30,
  "guidance":      5.0,
  "identity_scale":0.8,
  "adapter_scale": 0.8,
  "seed_base":     42
}

The worker is RESUMABLE: it skips any image whose output already exists.
"""

import io
import json
import os
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

# ----------------------------------------------------------------------------
# 0. Paths & shell helper
# ----------------------------------------------------------------------------
INPUT_ROOT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")   # ONLY this dir is pulled back by `kernels output`
TMP = Path("/kaggle/tmp")           # scratch: models/repo/zip live here, NOT downloaded

# Outputs the control panel needs (small, live in WORKING so they download back):
OUTPUT_DIR = WORKING / "headshots_out"
RESULTS_PATH = WORKING / "results.json"

# Heavy scratch (the 10GB models, InstantID repo, 360MB antelopev2 zip). Keeping
# these OUT of /kaggle/working means each job's result download is a few MB of
# JPEGs instead of dragging back hundreds of MB of models we already have.
DOWNLOAD_DIR = TMP / "downloads"
CHECKPOINTS_DIR = TMP / "checkpoints"
INSTANTID_DIR = TMP / "InstantID"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


RUN_LOG = WORKING / "run.log"


class _Tee:
    """Mirror stdout/stderr to a file so we get diagnostics even when Kaggle
    drops the kernel's captured log (empty .log on ERROR status)."""

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, data):
        self.stream.write(data)
        self.fh.write(data)
        self.fh.flush()

    def flush(self):
        self.stream.flush()
        self.fh.flush()


def sh(cmd: str, check: bool = True):
    """Run a shell command (script kernels can't use ! magics)."""
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"Command failed ({r.returncode}): {cmd}")


# ----------------------------------------------------------------------------
# 1. Load job config
# ----------------------------------------------------------------------------
def load_job() -> dict:
    """Find job.json anywhere under /kaggle/input, or fall back to defaults."""
    matches = list(INPUT_ROOT.glob("**/job.json"))
    if matches:
        cfg = json.loads(matches[0].read_text())
        print(f"[job] loaded {matches[0]}")
    else:
        cfg = {"input_mode": "images"}
        print("[job] no job.json found — using defaults (input_mode=images)")
    cfg.setdefault("job_id", "job")
    cfg.setdefault("input_mode", "images")
    cfg.setdefault("url_column", "url")
    cfg.setdefault("style_preset", "corporate")
    cfg.setdefault("limit", None)
    cfg.setdefault("img_size", 1024)          # SDXL native generation size
    cfg.setdefault("output_size", 1024)        # final saved size (upscaled if > img_size)
    cfg.setdefault("custom_prompt", "")        # optional extra prompt text appended to the style
    cfg.setdefault("background", "")           # optional background description (e.g. "soft teal")
    cfg.setdefault("face_enhance", False)      # optional GFPGAN face restoration (fail-safe)
    cfg.setdefault("white_background", False)  # optional rembg pass -> pure white bg (fail-safe)
    cfg.setdefault("num_steps", 30)
    cfg.setdefault("guidance", 5.0)
    cfg.setdefault("identity_scale", 0.8)
    cfg.setdefault("adapter_scale", 0.8)
    cfg.setdefault("seed_base", 42)
    # Multi-reference identity averaging: when true, average the face embeddings
    # from every source photo of the batch to build a more robust identity, then
    # use that averaged embedding for every generation. Reduces "one bad photo =
    # one bad output" drift for people who uploaded a small set of photos.
    cfg.setdefault("multi_reference", True)
    # Use Real-ESRGAN for hi-res upscale instead of Lanczos+UnsharpMask when
    # output_size > img_size. Falls back to Lanczos if unavailable.
    cfg.setdefault("realesrgan_upscale", True)
    return cfg


# ----------------------------------------------------------------------------
# Prompt hardening: strip user text before it joins the SDXL prompt so a photo
# batch with a malicious "background" or "custom_prompt" can't inject nsfw etc.
# ----------------------------------------------------------------------------
_PROMPT_BLOCKLIST = (
    "nsfw", "nude", "naked", "porn", "explicit", "sexy", "sexual",
    "gore", "violence", "blood", "weapon", "gun",
    "child", "kid", "minor", "loli", "shota",
)


def sanitize_user_text(s: str, max_len: int) -> str:
    if not s:
        return ""
    low = s.lower()
    if any(w in low for w in _PROMPT_BLOCKLIST):
        return ""
    # Keep it to letters/digits/space/punctuation — no prompt-splitting tokens.
    keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.-_+&"
    cleaned = "".join(c for c in s if c in keep).strip()
    return cleaned[:max_len]


STYLE_PRESETS = {
    "corporate": {
        "prompt": "professional corporate headshot, a person, neutral light gray studio background, tailored charcoal business suit, crisp white shirt, soft key light with subtle rim light, sharp focus on eyes, shallow depth of field, high detail skin texture, photorealistic, shot on 85mm lens",
        "negative": "cartoon, anime, illustration, painting, 3d render, blurry, low quality, distorted, deformed face, extra limbs, watermark, text, logo, oversaturated, harsh shadows",
    },
    "modern_tech": {
        "prompt": "professional headshot, a person, softly blurred modern office background with warm bokeh, smart casual attire crew neck sweater or open collar shirt, natural window light, friendly confident expression, photorealistic, shot on 50mm lens",
        "negative": "cartoon, anime, illustration, painting, 3d render, blurry face, low quality, distorted, deformed, extra limbs, watermark, text, oversaturated",
    },
    "warm_friendly": {
        "prompt": "warm approachable professional headshot, a person, soft cream beige background, cozy knit sweater in muted tone, golden hour lighting, genuine subtle smile, photorealistic, shot on 85mm lens",
        "negative": "cartoon, anime, illustration, painting, 3d render, blurry, low quality, distorted face, extra limbs, watermark, text, harsh light, cold tone",
    },
    "formal_executive": {
        "prompt": "formal executive portrait, a person, deep navy blue background with subtle vignette, tailored dark suit and tie, dramatic rembrandt lighting, confident authoritative expression, photorealistic, shot on 85mm lens, editorial quality",
        "negative": "cartoon, anime, illustration, painting, 3d render, blurry, low quality, distorted face, extra limbs, watermark, text, casual, oversaturated",
    },
    "linkedin_classic": {
        "prompt": "clean professional LinkedIn headshot, a person, smooth light blue-gray gradient background, business casual blazer over open collar shirt, bright even softbox lighting, approachable confident expression, sharp focus on eyes, photorealistic, shot on 85mm lens",
        "negative": "cartoon, anime, illustration, painting, 3d render, blurry, low quality, distorted face, extra limbs, watermark, text, harsh shadows, oversaturated",
    },
    "startup_casual": {
        "prompt": "modern startup team headshot, a person, bright airy white background, casual t-shirt or henley, natural daylight, relaxed genuine smile, clean and contemporary, photorealistic, shot on 50mm lens",
        "negative": "cartoon, anime, illustration, painting, 3d render, blurry, low quality, distorted face, extra limbs, watermark, text, formal suit, dark moody",
    },
    "healthcare": {
        "prompt": "professional healthcare portrait, a person, clean bright clinical white background, white medical coat over collared shirt, soft even lighting, warm trustworthy expression, photorealistic, shot on 85mm lens",
        "negative": "cartoon, anime, illustration, painting, 3d render, blurry, low quality, distorted face, extra limbs, watermark, text, casual, dark background",
    },
    "academic": {
        "prompt": "distinguished academic faculty portrait, a person, muted warm library background softly blurred, tweed jacket or smart cardigan, soft directional light, thoughtful composed expression, photorealistic, shot on 85mm lens",
        "negative": "cartoon, anime, illustration, painting, 3d render, blurry, low quality, distorted face, extra limbs, watermark, text, neon, oversaturated",
    },
}


# ----------------------------------------------------------------------------
# 2. Install deps + fetch models (idempotent; cached across a session)
# ----------------------------------------------------------------------------
def install_deps():
    print("[setup] installing python deps...", flush=True)
    # This InstantID stack (insightface 0.7.3, onnxruntime 1.17.1, opencv) was
    # built against NumPy 1.x. Kaggle's current base ships NumPy 2.x, which
    # breaks their compiled extensions on import ("numpy.core.multiarray failed
    # to import") — a hard crash with no Python traceback. Pin NumPy<2 first.
    sh("pip install -q 'numpy<2'")
    # CRITICAL: Kaggle's default PyTorch (a cu128 build) DROPPED Pascal support -
    # its lowest arch is sm_70. But Kaggle frequently assigns a Tesla P100 (sm_60),
    # so every diffusion step dies with `cudaErrorNoKernelImageForDevice`
    # ("P100 ... sm_60 is not compatible with the current PyTorch installation").
    # Install a torch build whose kernels include sm_60..sm_90, so generation
    # works on P100 AND T4 (sm_75) alike. This is the real fix for that CUDA error.
    sh("pip install -q torch==2.3.1 torchvision==0.18.1 "
       "--index-url https://download.pytorch.org/whl/cu121")
    # peft MUST be pinned alongside accelerate: diffusers builds the pipeline via
    # `from peft import PeftModel`, and peft>=0.13 imports accelerate's
    # `clear_device_cache` (only added in accelerate>=0.34). With accelerate
    # pinned to 0.29.2, Kaggle's newer base peft crashes on import. peft 0.11.1
    # is the last version compatible with this 2024-era stack. (confirmed bug)
    sh("pip install -q diffusers==0.27.2 transformers==4.39.3 accelerate==0.29.2 peft==0.11.1")
    sh("pip install -q insightface==0.7.3 onnxruntime-gpu==1.17.1")
    sh("pip install -q opencv-python-headless controlnet-aux==0.0.7 huggingface_hub==0.22.2")
    # Re-assert the NumPy pin last: some of the above may pull NumPy 2.x back in.
    sh("pip install -q 'numpy<2'")
    if not INSTANTID_DIR.exists():
        sh(f"git clone -q https://github.com/InstantID/InstantID.git {INSTANTID_DIR}")
    sys.path.append(str(INSTANTID_DIR))


def use_model_cache() -> bool:
    """If a Kaggle dataset of pre-downloaded models is attached (named
    'facefoundry-models'), point HuggingFace at it and stage antelopev2 so the
    big SDXL/InstantID downloads are skipped. Safe no-op when no cache is
    attached - the normal download path in fetch_models() then runs as usual."""
    caches = [p for p in INPUT_ROOT.glob("**/facefoundry-models") if p.is_dir()]
    cache = caches[0] if caches else None
    if not cache:
        print("[cache] no model cache attached - downloading models", flush=True)
        return False
    print(f"[cache] using attached model cache: {cache}", flush=True)
    hf = cache / "huggingface"
    if hf.is_dir():
        os.environ["HF_HOME"] = str(hf)          # SDXL/InstantID load from here
        os.environ["HF_HUB_OFFLINE"] = "0"
    ant = cache / "antelopev2"
    if ant.is_dir():
        dst = Path("/root/.insightface/models/antelopev2")
        dst.mkdir(parents=True, exist_ok=True)
        sh(f"cp -rn {ant}/*.onnx {dst}/ 2>/dev/null || true", check=False)
    ckpt = cache / "checkpoints"
    if ckpt.is_dir():
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        sh(f"cp -rn {ckpt}/* {CHECKPOINTS_DIR}/ 2>/dev/null || true", check=False)
    return True


def fetch_models():
    from huggingface_hub import hf_hub_download

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    print("[setup] downloading InstantID ControlNet + IP-Adapter...", flush=True)
    hf_hub_download("InstantX/InstantID", "ControlNetModel/config.json", local_dir=str(CHECKPOINTS_DIR))
    hf_hub_download("InstantX/InstantID", "ControlNetModel/diffusion_pytorch_model.safetensors", local_dir=str(CHECKPOINTS_DIR))
    hf_hub_download("InstantX/InstantID", "ip-adapter.bin", local_dir=str(CHECKPOINTS_DIR))

    print("[setup] downloading InsightFace antelopev2...", flush=True)
    # NOTE: antelopev2.zip already contains a top-level `antelopev2/` folder, so
    # it must be extracted into the MODELS dir (…/models/), which yields
    # …/models/antelopev2/*.onnx — the layout insightface expects. Extracting
    # into …/models/antelopev2/ would nest it as …/antelopev2/antelopev2/*.onnx
    # and FaceAnalysis(name="antelopev2") would find no models. (confirmed bug)
    models_root = Path("/root/.insightface/models")
    onnx_dir = models_root / "antelopev2"
    zip_path = TMP / "antelopev2.zip"  # scratch, so it isn't downloaded back
    if not list(onnx_dir.glob("*.onnx")):
        sh(f"mkdir -p {TMP}")
        sh(f"wget -q -O {zip_path} "
           "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip")
        sh(f"mkdir -p {models_root}")
        sh(f"unzip -qo {zip_path} -d {models_root}")
        n = len(list(onnx_dir.glob("*.onnx")))
        print(f"[setup] antelopev2 -> {onnx_dir} ({n} .onnx files)", flush=True)
        if n == 0:
            raise RuntimeError(f"antelopev2 models missing after unzip in {onnx_dir}")


# ----------------------------------------------------------------------------
# 3. Build the pipeline
# ----------------------------------------------------------------------------
def build_pipeline(cfg):
    import torch
    from insightface.app import FaceAnalysis
    from diffusers.models import ControlNetModel
    from pipeline_stable_diffusion_xl_instantid import (
        StableDiffusionXLInstantIDPipeline,
        draw_kps,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "GPU not detected — enable GPU accelerator on the kernel."

    # GPU / build diagnostics. `cudaErrorNoKernelImageForDevice` happens when the
    # installed CUDA kernels weren't compiled for this GPU's compute capability.
    # Log both so run.log pinpoints any torch<->GPU arch mismatch.
    cap = torch.cuda.get_device_capability(0)
    sm = f"sm_{cap[0]}{cap[1]}"
    arch_list = torch.cuda.get_arch_list()
    print(f"[pipeline] device={device} | {torch.cuda.get_device_name(0)} | {sm}", flush=True)
    print(f"[pipeline] torch={torch.__version__} cuda={torch.version.cuda} "
          f"arch_list={arch_list}", flush=True)
    if not any(str(cap[0]) + str(cap[1]) in a for a in arch_list):
        print(f"[pipeline] WARNING: installed torch {torch.__version__} has no kernels for "
              f"{sm} (arch_list={arch_list}). Generation will fail with 'no kernel image'. "
              f"The pinned torch==2.3.1+cu121 should include sm_60..sm_90 - if you see this, "
              f"that install did not take effect.", flush=True)

    face_app = FaceAnalysis(name="antelopev2", root="/root/.insightface",
                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    controlnet = ControlNetModel.from_pretrained(
        str(CHECKPOINTS_DIR / "ControlNetModel"), torch_dtype=torch.float16)

    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet, torch_dtype=torch.float16).to(device)
    pipe.load_ip_adapter_instantid(str(CHECKPOINTS_DIR / "ip-adapter.bin"))
    pipe.set_ip_adapter_scale(cfg["adapter_scale"])
    pipe.enable_vae_tiling()
    # NOTE: do NOT enable xformers. Kaggle's prebuilt xformers is frequently
    # compiled for a different GPU arch than the one assigned, so the diffusion
    # forward pass dies with `cudaErrorNoKernelImageForDevice` even though the
    # enable() call itself succeeds. Diffusers' default AttnProcessor2_0 uses
    # PyTorch SDPA (torch's own kernels, matched to the GPU) - reliable and fast.
    # (No attention slicing: SDXL fp16 fits in 16GB at 1024 and slicing would
    # only slow generation, which we want fast.)
    print("[pipeline] loaded (SDPA attention).", flush=True)
    return pipe, face_app, draw_kps, device


# ----------------------------------------------------------------------------
# 4. Gather source images
# ----------------------------------------------------------------------------
def gather_sources(cfg) -> list:
    """Return a list of (stem, local_path) source images to process."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    if cfg["input_mode"] == "urls":
        return _gather_from_urls(cfg)

    # images mode: only pull from the input dataset — the one that also carries
    # job.json. Skips the model-cache dataset (which may contain PNG/JPG assets)
    # and any other attached extras. Fallback to full scan only if job.json is
    # not found (legacy behavior).
    sources = []
    job_json_matches = list(INPUT_ROOT.glob("**/job.json"))
    if job_json_matches:
        input_dataset_dir = job_json_matches[0].parent
        print(f"[input] scoping to input dataset dir: {input_dataset_dir}")
        for p in sorted(input_dataset_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.name != "job.json":
                sources.append((p.stem, str(p)))
    else:
        # Legacy fallback — still exclude non-image files.
        for p in sorted(INPUT_ROOT.glob("**/*")):
            if p.suffix.lower() in IMAGE_EXTS and p.is_file():
                sources.append((p.stem, str(p)))
    if cfg["limit"]:
        sources = sources[: int(cfg["limit"])]
    print(f"[input] {len(sources)} source images (mode=images)")
    return sources


def _gather_from_urls(cfg) -> list:
    import csv
    import requests
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor

    csv_matches = list(INPUT_ROOT.glob("**/*.csv"))
    if not csv_matches:
        print("[input] urls mode but no CSV found in dataset.")
        return []
    csv_path = csv_matches[0]
    col = cfg["url_column"]
    with open(csv_path, encoding="utf-8") as f:
        urls = [r[col].strip() for r in csv.DictReader(f) if r.get(col, "").strip()]
    if cfg["limit"]:
        urls = urls[: int(cfg["limit"])]
    print(f"[input] {len(urls)} URLs from {csv_path.name}")

    def dl(item):
        i, url = item
        out = DOWNLOAD_DIR / f"src_{i:05d}.jpg"
        if out.exists():
            return (f"src_{i:05d}", str(out), None)
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            Image.open(io.BytesIO(r.content)).convert("RGB").save(out, "JPEG", quality=92)
            return (f"src_{i:05d}", str(out), None)
        except Exception as e:
            return (f"src_{i:05d}", None, f"{type(e).__name__}: {e}")

    sources, dl_failures = [], []
    with ThreadPoolExecutor(max_workers=16) as ex:
        for stem, path, err in ex.map(dl, enumerate(urls)):
            (dl_failures if err else sources).append((stem, err) if err else (stem, path))
    print(f"[input] downloaded {len(sources)}/{len(urls)} (failures: {len(dl_failures)})")
    return sources


# ----------------------------------------------------------------------------
# 5. Process one image
# ----------------------------------------------------------------------------
def _load_face_enhancer():
    """Return a GFPGAN restore function, or None if unavailable. Fully guarded:
    GFPGAN's deps (basicsr/facexlib) are fragile, so any failure here just means
    'no enhancement' - it must never crash a job."""
    try:
        sh("pip install -q gfpgan==1.3.8 realesrgan==0.3.0", check=False)
        # basicsr <1.4.2 imports torchvision.transforms.functional_tensor, removed
        # in torchvision 0.17+. Patch a shim so the import doesn't explode.
        import importlib.util, types, sys as _sys
        if importlib.util.find_spec("torchvision.transforms.functional_tensor") is None:
            import torchvision.transforms.functional as _F
            shim = types.ModuleType("torchvision.transforms.functional_tensor")
            shim.rgb_to_grayscale = _F.rgb_to_grayscale
            _sys.modules["torchvision.transforms.functional_tensor"] = shim
        from gfpgan import GFPGANer
        wpath = TMP / "GFPGANv1.4.pth"
        if not wpath.is_file():
            sh("wget -q -O {} https://github.com/TencentARC/GFPGAN/releases/download/"
               "v1.3.0/GFPGANv1.4.pth".format(wpath), check=False)
        gfp = GFPGANer(model_path=str(wpath), upscale=1, arch="clean", channel_multiplier=2)
        print("[enhance] GFPGAN loaded", flush=True)
        return gfp
    except Exception as e:
        print(f"[enhance] GFPGAN unavailable, skipping face restore: {e}", flush=True)
        return None


def _load_realesrgan(target_size: int):
    """Return an upsample(pil) -> pil callable, or None. `target_size` is the
    LONGEST edge we want; the upscaler outputs its native 4x then we resize
    down. Real-ESRGAN preserves detail much better than Lanczos at 2K/4K."""
    try:
        # realesrgan is already installed by _load_face_enhancer if GFPGAN ran;
        # install here too so the upscaler works even when face_enhance=False.
        sh("pip install -q realesrgan==0.3.0", check=False)
        import numpy as np
        from PIL import Image
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        wpath = TMP / "RealESRGAN_x4plus.pth"
        if not wpath.is_file():
            sh("wget -q -O {} https://github.com/xinntao/Real-ESRGAN/releases/download/"
               "v0.1.0/RealESRGAN_x4plus.pth".format(wpath), check=False)
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                        num_grow_ch=32, scale=4)
        up = RealESRGANer(scale=4, model_path=str(wpath), model=model,
                          tile=384, tile_pad=10, pre_pad=0, half=True)

        def apply(pil_img):
            arr = np.array(pil_img.convert("RGB"))
            out, _ = up.enhance(arr, outscale=4)
            up_img = Image.fromarray(out)
            w, h = up_img.size
            m = max(w, h)
            if m > target_size:
                s = target_size / m
                up_img = up_img.resize((int(w*s), int(h*s)), Image.LANCZOS)
            return up_img

        print(f"[upscale] Real-ESRGAN x4 loaded (target={target_size})", flush=True)
        return apply
    except Exception as e:
        print(f"[upscale] Real-ESRGAN unavailable, falling back to Lanczos: {e}", flush=True)
        return None


def _load_bg_remover():
    """Return a rembg session, or None if unavailable. Fully guarded - a pure
    white background is a nice-to-have, never a reason to fail a job."""
    try:
        sh("pip install -q rembg==2.0.57 pymatting==1.1.12", check=False)
        from rembg import new_session
        session = new_session("u2net")  # downloads the ~170MB u2net model once
        print("[whitebg] rembg loaded", flush=True)
        return session
    except Exception as e:
        print(f"[whitebg] rembg unavailable, keeping generated background: {e}", flush=True)
        return None


def make_processor(cfg, pipe, face_app, draw_kps, device):
    import cv2
    import numpy as np
    import torch
    from PIL import Image, ImageFilter

    preset = STYLE_PRESETS.get(cfg["style_preset"], STYLE_PRESETS["corporate"])
    prompt, negative = preset["prompt"], preset["negative"]
    # Sanitize user prompt tweaks before they touch the SDXL prompt — blocks
    # nsfw/violence injection via the "background" and "custom prompt" fields.
    bg_txt = sanitize_user_text(cfg.get("background", ""), 80)
    cust_txt = sanitize_user_text(cfg.get("custom_prompt", ""), 200)
    if bg_txt:
        prompt = f"{prompt}, {bg_txt} background"
    if cust_txt:
        prompt = f"{prompt}, {cust_txt}"
    # Reinforce negative prompt with common artifact terms.
    negative = negative + ", plastic skin, waxy skin, doll-like, uncanny, jpeg artifacts, mutated hands, extra fingers"

    gen_size = int(cfg["img_size"])          # SDXL renders best at its native 1024
    out_size = int(cfg.get("output_size", gen_size))
    enhancer = _load_face_enhancer() if cfg.get("face_enhance") else None
    bg_session = _load_bg_remover() if cfg.get("white_background") else None
    upscaler = _load_realesrgan(out_size) if (out_size > gen_size and cfg.get("realesrgan_upscale", True)) else None

    def process(src_path, dst_path, seed, avg_embedding=None):
        img = cv2.imread(src_path)
        if img is None:
            return False, "unreadable image"
        faces = face_app.get(img)
        if not faces:
            return False, "no face detected"
        face = sorted(faces, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
                      reverse=True)[0]
        kps = draw_kps(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), face["kps"])
        # Use the averaged batch embedding when caller supplied one — sharper
        # identity across a batch of the same person's photos.
        embedding = avg_embedding if avg_embedding is not None else face["embedding"]
        gen = torch.Generator(device=device).manual_seed(seed)
        result = pipe(
            prompt=prompt, negative_prompt=negative,
            image_embeds=embedding, image=kps,
            controlnet_conditioning_scale=cfg["identity_scale"],
            ip_adapter_scale=cfg["adapter_scale"],
            num_inference_steps=cfg["num_steps"], guidance_scale=cfg["guidance"],
            width=gen_size, height=gen_size, generator=gen,
        ).images[0]
        # Optional GFPGAN face restoration (sharper eyes/skin). Fully guarded.
        if enhancer is not None:
            try:
                bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
                _, _, restored = enhancer.enhance(bgr, has_aligned=False,
                                                  only_center_face=True, paste_back=True)
                if restored is not None:
                    result = Image.fromarray(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))
            except Exception as e:
                print(f"[enhance] restore failed for this image, using original: {e}", flush=True)
        # Optional pure-white background. Fully guarded.
        if bg_session is not None:
            try:
                from rembg import remove
                result = remove(result, session=bg_session,
                                bgcolor=(255, 255, 255, 255)).convert("RGB")
            except Exception as e:
                print(f"[whitebg] failed for this image, keeping background: {e}", flush=True)
        # Upscale to requested output resolution. Real-ESRGAN preserves detail
        # far better than plain Lanczos at 2K/4K; falls back to Lanczos+unsharp
        # if the upscaler wasn't loadable.
        if out_size > gen_size:
            if upscaler is not None:
                try:
                    result = upscaler(result)
                except Exception as e:
                    print(f"[upscale] Real-ESRGAN failed, using Lanczos: {e}", flush=True)
                    result = result.resize((out_size, out_size), Image.LANCZOS)
                    result = result.filter(ImageFilter.UnsharpMask(radius=2.2, percent=70, threshold=2))
            else:
                result = result.resize((out_size, out_size), Image.LANCZOS)
                result = result.filter(ImageFilter.UnsharpMask(radius=2.2, percent=70, threshold=2))
        result.save(dst_path, "JPEG", quality=95, subsampling=0)
        return True, None

    def build_avg_embedding(source_paths):
        """Return the mean face embedding across a list of source images. Skips
        images where no face is detected. Returns None on total failure."""
        embs = []
        for p in source_paths:
            im = cv2.imread(p)
            if im is None:
                continue
            fs = face_app.get(im)
            if not fs:
                continue
            fs.sort(key=lambda x: (x["bbox"][2]-x["bbox"][0])*(x["bbox"][3]-x["bbox"][1]),
                    reverse=True)
            embs.append(fs[0]["embedding"])
        if not embs:
            return None
        avg = np.mean(np.stack(embs, axis=0), axis=0)
        # ArcFace embeddings are typically L2-normalized; renormalize after mean.
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm
        print(f"[identity] averaged embedding across {len(embs)} photos", flush=True)
        return avg

    return process, build_avg_embedding


# ----------------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------------
def _run(t_start):
    cfg = load_job()
    print("[job] config:", json.dumps(cfg), flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    install_deps()
    use_model_cache()   # skip downloads if a models dataset is attached (safe no-op)
    fetch_models()
    pipe, face_app, draw_kps, device = build_pipeline(cfg)
    process, build_avg_embedding = make_processor(cfg, pipe, face_app, draw_kps, device)

    sources = gather_sources(cfg)
    # Build a shared "averaged" identity embedding across all input photos when
    # multi-reference is on and there are at least 2 photos. This makes every
    # output share one strong identity instead of drifting per-photo.
    avg_embedding = None
    if cfg.get("multi_reference") and len(sources) >= 2:
        avg_embedding = build_avg_embedding([p for _, p in sources])
    results = []
    done = 0
    for i, (stem, src_path) in enumerate(sources, 1):
        dst_path = OUTPUT_DIR / f"{stem}.jpg"
        seed = int(cfg["seed_base"]) + i
        if dst_path.exists():
            results.append({"stem": stem, "status": "ok", "output": dst_path.name, "seed": seed, "cached": True})
            done += 1
            continue
        try:
            ok, err = process(src_path, str(dst_path), seed, avg_embedding=avg_embedding)
            if ok:
                results.append({"stem": stem, "status": "ok", "output": dst_path.name, "seed": seed})
                done += 1
            else:
                results.append({"stem": stem, "status": "failed", "error": err, "seed": seed})
        except Exception as e:
            results.append({"stem": stem, "status": "error", "error": f"{type(e).__name__}: {e}", "seed": seed})
            traceback.print_exc()
        if i % 10 == 0 or i == len(sources):
            el = time.time() - t_start
            print(f"[{i}/{len(sources)}] ok={done} elapsed={el/60:.1f}m", flush=True)

    # results.json (structured, for the review UI)
    summary = {
        "job_id": cfg["job_id"],
        "style_preset": cfg["style_preset"],
        "total": len(sources),
        "ok": sum(r["status"] == "ok" for r in results),
        "failed": sum(r["status"] != "ok" for r in results),
        "config": cfg,
        "results": results,
    }
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))
    print(f"[done] ok={summary['ok']} failed={summary['failed']} -> {RESULTS_PATH}", flush=True)

    # zip outputs
    zip_path = WORKING / f"headshots_{cfg['style_preset']}_{cfg['job_id']}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(OUTPUT_DIR.glob("*.jpg")):
            z.write(p, arcname=p.name)
        if RESULTS_PATH.exists():
            z.write(RESULTS_PATH, arcname="results.json")
    print(f"[done] zip -> {zip_path} ({time.time()-t_start:.0f}s total)", flush=True)


def main():
    """Tee all output to run.log and guarantee a results.json is written even if
    setup crashes hard — so the orchestrator always gets a diagnosable artifact."""
    WORKING.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    log_fh = open(RUN_LOG, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)
    try:
        _run(t_start)
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, flush=True)
        # Always leave a machine-readable failure marker for the review UI.
        if not RESULTS_PATH.exists():
            RESULTS_PATH.write_text(json.dumps({
                "job_id": "unknown",
                "fatal": True,
                "error": f"{type(e).__name__}: {e}",
                "traceback": tb,
                "total": 0, "ok": 0, "failed": 0, "results": [],
            }, indent=2))
        raise
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_fh.close()


if __name__ == "__main__":
    main()
