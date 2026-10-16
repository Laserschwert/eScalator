"""
eScalator - High-performance AI upscaling tool
Backend: FastAPI + Spandrel (42+ architectures) + ONNX Runtime

Speed optimisations vs plain spandrel:
  • Models run on GPU (cuda) instead of CPU
  • fp16 (half precision) inference on CUDA — ~2× faster, half VRAM
  • tf32 enabled for Ampere+ GPUs — tensor-core convolutions at ~3× fp32 speed
  • cudnn.benchmark — picks fastest conv algorithm per input shape
  • Models cached in RAM/VRAM after first load — preloaded on selection
  • torch.compile() warm-up disabled (adds latency on first run, skip for UX)
"""

import base64
import io
import json
import threading
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, Response
from blend_modes import blend_images, BLEND_MODES
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Suppress harmless ONNX ScatterND warnings and torch.meshgrid deprecation
warnings.filterwarnings("ignore", message=".*ScatterND.*")
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
warnings.filterwarnings("ignore", message=".*DecompressionBomb.*")
Image.MAX_IMAGE_PIXELS = None  # allow very large images (user's own files)

DEVICE   = "cpu"
USE_FP16 = False
HAS_TORCH = False

# ── PyTorch + GPU setup ────────────────────────────────────────────────────────
try:
    import torch
    import spandrel
    HAS_TORCH = True

    if torch.cuda.is_available():
        # Defer cudnn/tf32 setup to first inference — avoid initialising a CUDA
        # context at import time which can interfere with other GPU apps (e.g. Blender OptiX)
        DEVICE   = "cuda"
        USE_FP16 = True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE   = "mps"
        USE_FP16 = False
    else:
        DEVICE   = "cpu"
        USE_FP16 = False
    print(f"[torch] device={DEVICE}  cuda={torch.cuda.is_available()}  version={torch.__version__}")

except ImportError as e:
    print(f"[torch] not available ({e})")

# ── ONNX Runtime ───────────────────────────────────────────────────────────────
try:
    import onnxruntime as ort
    HAS_ONNX = True
    _ORT_AVAILABLE_PROVIDERS = set(ort.get_available_providers())
except ImportError:
    HAS_ONNX = False
    _ORT_AVAILABLE_PROVIDERS = set()

def _onnx_providers() -> list:
    """Return best available ONNX execution providers in priority order."""
    if not HAS_ONNX:
        return ["CPUExecutionProvider"]
    avail = _ORT_AVAILABLE_PROVIDERS
    chain = []
    if "TensorrtExecutionProvider" in avail:
        chain.append("TensorrtExecutionProvider")
    if "CUDAExecutionProvider" in avail:
        chain.append("CUDAExecutionProvider")
    if "ROCMExecutionProvider" in avail:
        chain.append("ROCMExecutionProvider")
    if "DmlExecutionProvider" in avail:
        chain.append("DmlExecutionProvider")
    if "CoreMLExecutionProvider" in avail:
        chain.append("CoreMLExecutionProvider")
    chain.append("CPUExecutionProvider")
    return chain

app = FastAPI(title="eScalator API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR    = Path(__file__).parent
WORK_DIR    = BASE_DIR / "work"   # clipboard uploads + processed results
UPLOADS_DIR = WORK_DIR
OUTPUT_DIR  = WORK_DIR
STATIC_DIR  = BASE_DIR / "static"
try:
    APP_VERSION = (BASE_DIR / "VERSION").read_text(encoding="utf-8").strip()
except OSError:
    APP_VERSION = "?"
CONFIG_FILE = BASE_DIR / "config.json"

WORK_DIR.mkdir(exist_ok=True)

MODEL_EXTENSIONS = {".pth", ".safetensors", ".pt", ".onnx"}

# ── Config ─────────────────────────────────────────────────────────────────────
def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            pass
    return {"models_dir": str(BASE_DIR / "models")}

def save_config(cfg: dict):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))

PRESETS_FILE = BASE_DIR / "presets.json"

def load_presets() -> dict:
    if PRESETS_FILE.exists():
        try: return json.loads(PRESETS_FILE.read_text())
        except: pass
    return {}

def save_presets(p: dict):
    PRESETS_FILE.write_text(json.dumps(p, indent=2, ensure_ascii=False))

# ── Model metadata cache ────────────────────────────────────────────────────────
# Maps model filename (e.g. "Real_HAT_x4.pth") → {scale, type, …}
# Persisted so scale survives restarts without re-loading models.
MODEL_META_FILE = BASE_DIR / "model_meta.json"

def load_model_meta() -> dict:
    if MODEL_META_FILE.exists():
        try:  return json.loads(MODEL_META_FILE.read_text())
        except: pass
    return {}

def save_model_meta(meta: dict):
    MODEL_META_FILE.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

model_meta: dict = load_model_meta()   # {filename → {scale, …}}

def _auto_canvas_limit() -> int:
    """Default canvas preview limit — 8K is safe for all systems."""
    return 8192

cfg     = load_config()
# Apply canvas limit default if not set
if "canvas_limit" not in cfg:
    cfg["canvas_limit"] = _auto_canvas_limit()
    save_config(cfg)
elif cfg["canvas_limit"] > 16384:   # 32K option removed in v1.3 (beyond the browser's pixel limit)
    cfg["canvas_limit"] = 16384
    save_config(cfg)

presets = load_presets()
Path(cfg["models_dir"]).mkdir(parents=True, exist_ok=True)

# ── WebView window reference (set by app.py after window creation) ─────────────
_webview_window = None  # set to enable direct chunked pixel push

def _eval_js(js: str):
    """Thread-safe evaluate_js — handles WKWebView (macOS) and GTK (Linux) main-thread requirements."""
    if _webview_window is None:
        return
    try:
        _webview_window.evaluate_js(js)
    except Exception as e:
        log(f"[eval_js] {e}")

loaded_models: dict = {}   # path → model descriptor or ort session
load_lock = threading.Lock()  # prevents two threads loading the same model simultaneously

processing_status: dict = {
    "running": False, "progress": 0, "message": "",
    "step": "", "tile_current": 0, "tile_total": 0,
    "pipe_step": 0, "pipe_total": 0,
}
cancel_event = threading.Event()   # set to request cancellation

class ImageState:
    path: Optional[Path] = None
    source_path: Optional[Path] = None
    is_clipboard: bool = False

img_state = ImageState()
alpha_result_cache: Optional[bytes] = None  # PNG bytes of last upscaled alpha
result_raw_cache = None  # (straight RGBA bytes, w, h) of last preview result

import time

_server_log: list[str] = []   # rolling buffer of last 200 log lines

def log(msg: str):
    """Timestamped console output, also buffered for /api/debug-log."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    _server_log.append(line)
    if len(_server_log) > 200:
        _server_log.pop(0)


# ── Status helper ──────────────────────────────────────────────────────────────
def set_status(progress: int, message: str, step: str = "",
               tile_current: int = 0, tile_total: int = 0,
               pipe_step: int = 0, pipe_total: int = 0):
    processing_status.update({
        "running": True, "progress": progress, "message": message,
        "step": step, "tile_current": tile_current, "tile_total": tile_total,
        "pipe_step": pipe_step, "pipe_total": pipe_total,
    })

# ── Image helpers ──────────────────────────────────────────────────────────────
def _guess_scale_from_name(name: str) -> int:
    """Extract scale factor from a model filename. Recognises '4x_model', '4xNomos',
    '2x-Foo', 'RealESRGAN_x4plus', 'SwinIR_x2', 'model-x3' … → 4/4/2/4/2/3. 0 if none."""
    import re
    stem = Path(name).stem
    for pat in (r'(?<![0-9a-z])([1-8])x(?![0-9])',      # 4x, 4x_, 4xName
                r'(?<![0-9a-z])x([1-8])(?![0-9])',      # _x4, -x2, x4plus
                r'(?<![a-z])sr[_-]?x([1-8])(?![0-9])'):  # HAT/SwinIR style: SRx4, SR_x2
        m = re.search(pat, stem, re.IGNORECASE)
        if m and int(m.group(1)) in (1, 2, 3, 4, 8):
            return int(m.group(1))
    return 0


def _has_alpha(img: Image.Image) -> bool:
    """Detect alpha channel reliably, including TIFF unassociated-alpha edge cases."""
    if img.mode in ("RGBA", "LA", "PA"):
        return True
    if img.mode == "P" and "transparency" in img.info:
        return True
    if 'A' in img.getbands():
        return True
    return False


def to_rgb(img: Image.Image) -> Image.Image:
    """
    Robustly convert any Pillow image to 8-bit RGB.
    Handles TIFF edge cases: I, I;16, I;32, F, RGB;16 modes where
    plain .convert('RGB') maps values wrong (→ black image).
    """
    mode = img.mode
    log(f"[to_rgb] input mode={mode}")

    # Integer grayscale (16-bit or 32-bit) — normalise range to 0-255
    if mode in ("I", "I;16", "I;16B", "I;32", "I;32B"):
        arr = np.array(img, dtype=np.float32)
        lo, hi = arr.min(), arr.max()
        log(f"[to_rgb] I-mode  min={lo:.1f}  max={hi:.1f}")
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
        return img.convert("RGB")

    # Float grayscale
    if mode == "F":
        arr = np.array(img, dtype=np.float32)
        lo, hi = arr.min(), arr.max()
        log(f"[to_rgb] F-mode  min={lo:.1f}  max={hi:.1f}")
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
        return img.convert("RGB")

    # 16-bit RGB/RGBA — shift down to 8-bit
    if mode in ("RGB;16", "RGB;16B", "RGBA;16", "RGBA;16B"):
        arr = np.array(img, dtype=np.uint16)
        log(f"[to_rgb] 16-bit RGB  min={arr.min()}  max={arr.max()}")
        arr8 = (arr >> 8).astype(np.uint8)
        if arr8.ndim == 3 and arr8.shape[2] == 4:
            return Image.fromarray(arr8, mode="RGBA").convert("RGB")
        return Image.fromarray(arr8[:, :, :3], mode="RGB")

    # Anything exotic (CMYK, LAB, YCbCr…)
    if mode not in ("RGB", "RGBA", "L", "LA", "P", "PA"):
        log(f"[to_rgb] unusual mode {mode} → converting via RGBA/RGB")
        img = img.convert("RGBA" if "A" in mode else "RGB")

    result = img.convert("RGB")
    log(f"[to_rgb] done  {mode} → RGB")
    return result


def to_rgba(img: Image.Image) -> Image.Image:
    """Convert to RGBA, preserving or adding alpha channel."""
    if img.mode == "RGBA":
        return img
    if img.mode in ("LA", "PA"):
        return img.convert("RGBA")
    return to_rgb(img).convert("RGBA")


def pil_to_np(img: Image.Image) -> np.ndarray:
    img = to_rgb(img)
    return (np.array(img, dtype=np.float32) / 255.0)[:, :, ::-1]   # HWC BGR

def np_to_pil(arr: np.ndarray) -> Image.Image:
    # float32 conversion + nan/inf guard before uint8 cast
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    return Image.fromarray((np.clip(arr, 0.0, 1.0)[:, :, ::-1] * 255).astype(np.uint8))

def np_to_tensor(arr: np.ndarray):
    """HWC BGR float32 → 1CHW RGB float32 tensor on device."""
    t = torch.from_numpy(arr[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0)
    t = t.to(DEVICE)
    if USE_FP16:
        t = t.half()
    return t

def tensor_to_np(t) -> np.ndarray:
    """1CHW RGB tensor → HWC BGR float32 numpy."""
    return t.squeeze(0).permute(1, 2, 0).float().cpu().numpy()[:, :, ::-1].copy()

# ── Model loading ──────────────────────────────────────────────────────────────
def ensure_loaded(path: Path, model_idx: int = 0, model_total: int = 1):
    key = str(path)
    # Fast path: already cached (no lock needed for read)
    if key in loaded_models:
        log(f"[cache hit] {path.name}")
        return loaded_models[key]

    # Slow path: load under lock so two threads don't load the same model simultaneously
    with load_lock:
        # Re-check inside lock — another thread may have loaded it while we waited
        if key in loaded_models:
            log(f"[cache hit after lock] {path.name}")
            return loaded_models[key]

        log(f"[load] START  {path.name}  ({path.stat().st_size/1024/1024:.1f} MB)")
        set_status(10 + model_idx * 60 // model_total,
                   f"Loading model {model_idx+1}/{model_total}: {path.name}",
                   step="load_model")

        suffix = path.suffix.lower()
        if suffix in (".pth", ".safetensors", ".pt"):
            t0 = time.perf_counter()
            md = spandrel.ModelLoader(device=DEVICE).load_from_file(str(path))
            log(f"[load] spandrel.load_from_file  {time.perf_counter()-t0:.2f}s")
            t0 = time.perf_counter()
            md.model.eval()
            log(f"[load] model.eval()             {time.perf_counter()-t0:.2f}s")
            if USE_FP16:
                try:
                    t0 = time.perf_counter()
                    md.model.half()
                    test_dtype = next(md.model.parameters()).dtype
                    dummy = torch.zeros(1, 3, 32, 32, dtype=test_dtype, device=DEVICE)
                    with torch.no_grad():
                        md.model(dummy)
                    log(f"[load] model.half() OK          {time.perf_counter()-t0:.2f}s")
                except Exception as e:
                    log(f"[load] model.half() failed ({e}) → falling back to fp32")
                    md.model.float()
            loaded_models[key] = md
            log(f"[load] DONE  {path.name}  arch={getattr(md,'architecture','?')}  scale={getattr(md,'scale','?')}  dtype={next(md.model.parameters()).dtype}")
            # Persist scale so future cold-starts don't need to reload the model
            sc = int(getattr(md, "scale", 0) or 0)
            if sc and path.name not in model_meta or model_meta.get(path.name, {}).get("scale") != sc:
                model_meta[path.name] = {**model_meta.get(path.name, {}), "scale": sc}
                save_model_meta(model_meta)
        elif suffix == ".onnx":
            t0 = time.perf_counter()
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = _onnx_providers()
            log(f"[load] ONNX providers: {providers}")
            # Grow the CUDA memory arena only as needed instead of doubling (less VRAM waste)
            providers = [("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})
                         if pv == "CUDAExecutionProvider" else pv for pv in providers]
            sess = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
            loaded_models[key] = sess
            log(f"[load] ONNX session ready  {time.perf_counter()-t0:.2f}s")
        else:
            raise ValueError(f"Unsupported model format: {suffix}")

        return loaded_models[key]

# ── Tile inference ─────────────────────────────────────────────────────────────
_active_ort_runs: set = set()          # RunOptions of running ONNX sessions (for terminate)
_ort_runs_lock = threading.Lock()

def release_vram_cache():
    """Return PyTorch's cached-but-unused VRAM (tile/output buffers) to the driver.
    Loaded models stay in VRAM, so the next run starts without reloading."""
    if not HAS_TORCH:
        return
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        log(f"[vram] release failed: {e}")

def check_cancel():
    """Raise CancelledError if the user pressed Cancel. Call between expensive steps."""
    if cancel_event.is_set():
        raise CancelledError("Cancelled by user")

def request_cancel():
    """Set the cancel flag and abort running ONNX inferences immediately."""
    cancel_event.set()
    with _ort_runs_lock:
        for ro in list(_active_ort_runs):
            try: ro.terminate = True
            except Exception: pass

class CancelledError(Exception):
    pass

def tile_upscale(md, img_tensor, tile_size=512, overlap=32,
                 model_idx=0, model_total=1,
                 pipe_step=0, pipe_total=0):
    _, c, h, w = img_tensor.shape
    sc     = md.scale
    stride = tile_size - overlap

    ys = list(range(0, h, stride))
    xs = list(range(0, w, stride))
    total = len(ys) * len(xs)
    done  = 0

    pct_start = 10 + model_idx * 55 // model_total
    pct_end   = 10 + (model_idx + 1) * 55 // model_total

    model_dtype = next(md.model.parameters()).dtype
    try:
        # Always accumulate in fp32 for precision, regardless of model dtype
        output = torch.zeros(1, c, h*sc, w*sc, dtype=torch.float32, device=DEVICE)
        weight = torch.zeros(1, 1, h*sc, w*sc, dtype=torch.float32, device=DEVICE)
    except (RuntimeError, MemoryError) as e:
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise RuntimeError(
            f"Out of memory allocating output buffer ({w*sc}×{h*sc}). "
            f"Try a smaller tile size or use a model with a lower upscale factor."
        ) from e

    for y in ys:
        for x in xs:
            check_cancel()

            y1 = min(y, h - tile_size) if h > tile_size else 0
            x1 = min(x, w - tile_size) if w > tile_size else 0
            y2, x2 = min(y1+tile_size, h), min(x1+tile_size, w)
            try:
                with torch.no_grad():
                    out_t = md(img_tensor[:, :, y1:y2, x1:x2])
                    out_t = torch.nan_to_num(out_t, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
            except (RuntimeError, MemoryError) as e:
                if "out of memory" in str(e).lower() or isinstance(e, MemoryError):
                    if HAS_TORCH and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise RuntimeError(
                        f"CUDA out of memory on tile {done+1}/{total} "
                        f"(tile_size={tile_size}). Reduce tile size and try again."
                    ) from e
                raise
            oy1,ox1,oy2,ox2 = y1*sc,x1*sc,y2*sc,x2*sc
            output[:,:,oy1:oy2,ox1:ox2] += out_t.float()
            weight[:,:,oy1:oy2,ox1:ox2] += 1
            done += 1
            pct = pct_start + (pct_end - pct_start) * done // total
            set_status(pct,
                       f"tile {done}/{total}",
                       step="inference", tile_current=done, tile_total=total,
                       pipe_step=pipe_step, pipe_total=pipe_total)

    set_status(pct_end,
               f"merging {total} tiles…",
               step="merge", tile_current=total, tile_total=total,
               pipe_step=pipe_step, pipe_total=pipe_total)
    result = output / weight.clamp(min=1)
    return result

# ── Per-model runners ──────────────────────────────────────────────────────────
def run_spandrel(path: Path, img: Image.Image, tile_size=512,
                 model_idx=0, model_total=1,
                 pipe_step=0, pipe_total=0) -> Image.Image:
    t_total = time.perf_counter()
    check_cancel()
    md  = ensure_loaded(path, model_idx, model_total)
    check_cancel()
    arr = pil_to_np(img)
    t0 = time.perf_counter()
    t   = np_to_tensor(arr)
    model_dtype = next(md.model.parameters()).dtype
    if t.dtype != model_dtype:
        t = t.to(model_dtype)
    log(f"[infer] np_to_tensor (H={arr.shape[0]} W={arr.shape[1]})  {time.perf_counter()-t0:.3f}s  dtype={t.dtype}")
    h, w = arr.shape[:2]

    pct_base = 10 + model_idx * 60 // model_total
    if h <= tile_size and w <= tile_size:
        set_status(pct_base + 2,
                   f"inferencing…",
                   step="inference", tile_current=1, tile_total=1,
                   pipe_step=pipe_step, pipe_total=pipe_total)
        log(f"[infer] START single tile  device={DEVICE}  fp16={USE_FP16}")
        t0 = time.perf_counter()
        with torch.no_grad():
            out = md(t)
            out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        log(f"[infer] forward pass DONE  {time.perf_counter()-t0:.3f}s")
        set_status(10 + (model_idx+1)*60//model_total,
                   f"done",
                   step="inference", tile_current=1, tile_total=1,
                   pipe_step=pipe_step, pipe_total=pipe_total)
    else:
        log(f"[infer] tiled  tile_size={tile_size}  img={w}×{h}")
        out = tile_upscale(md, t, tile_size,
                           model_idx=model_idx, model_total=model_total,
                           pipe_step=pipe_step, pipe_total=pipe_total)
    check_cancel()
    t0 = time.perf_counter()
    result = np_to_pil(tensor_to_np(out))
    log(f"[infer] tensor_to_np + to_pil  {time.perf_counter()-t0:.3f}s")
    log(f"[infer] TOTAL run_spandrel  {time.perf_counter()-t_total:.3f}s")
    return result

# ── ONNX runner ───────────────────────────────────────────────────────────────
# Per-model calibration, decided on the first tile and cached for the session:
#   layout   "NCHW" | "NHWC"
#   in_mul   1.0 (model expects 0..1) or 255.0 (model expects 0..255)
#   out_div  1.0 or 255.0 (model outputs 0..255)
#   scale    integer upscale factor
_onnx_profiles: dict = {}

class OnnxModelError(RuntimeError):
    """Model-specific problem with a user-facing explanation."""

def _is_oom(e: Exception) -> bool:
    m = str(e).lower()
    return ("failed to allocate" in m or "out of memory" in m or "bfcarena" in m
            or "cudnn_status_alloc_failed" in m or "cuda_error_out_of_memory" in m
            or isinstance(e, MemoryError))

def _onnx_input_info(sess):
    """Return (layout, static_h, static_w, np_dtype) of the model's image input."""
    inp = sess.get_inputs()[0]
    shp = list(inp.shape) + [None] * (4 - len(inp.shape))
    dims = [d if isinstance(d, int) and d > 0 else None for d in shp[:4]]
    if dims[1] in (1, 3, 4):
        layout, sh, sw = "NCHW", dims[2], dims[3]
    elif dims[3] in (1, 3, 4):
        layout, sh, sw = "NHWC", dims[1], dims[2]
    else:
        layout, sh, sw = "NCHW", dims[2], dims[3]
    np_dtype = np.float16 if "float16" in inp.type else np.float32
    return layout, sh, sw, np_dtype

def _onnx_forward(sess, tile_rgb: np.ndarray, layout: str, in_mul: float, np_dtype):
    """tile_rgb: HWC RGB float32 0..1 → HWC RGB float32 (raw model range). Terminable."""
    x = tile_rgb * in_mul
    x = x.transpose(2, 0, 1)[np.newaxis] if layout == "NCHW" else x[np.newaxis]
    x = np.ascontiguousarray(x, dtype=np_dtype)
    run_opts = ort.RunOptions()
    with _ort_runs_lock:
        _active_ort_runs.add(run_opts)
    try:
        if cancel_event.is_set():
            run_opts.terminate = True
        out = sess.run(None, {sess.get_inputs()[0].name: x}, run_opts)[0]
    except Exception as e:
        if cancel_event.is_set():
            raise CancelledError("Cancelled by user") from e
        raise
    finally:
        with _ort_runs_lock:
            _active_ort_runs.discard(run_opts)
    out = np.asarray(out, dtype=np.float32)
    if out.ndim == 4:
        out = out[0]
    if out.ndim == 3 and out.shape[0] in (1, 3, 4) and out.shape[2] not in (1, 3, 4):
        out = out.transpose(1, 2, 0)          # CHW → HWC
    elif out.ndim == 3 and out.shape[0] in (1, 3, 4) and layout == "NCHW":
        out = out.transpose(1, 2, 0)
    if out.ndim == 2:
        out = out[:, :, None]
    if out.shape[2] == 1:
        out = np.repeat(out, 3, axis=2)
    return out[:, :, :3]

def _pad_tile(t: np.ndarray, th: int, tw: int) -> np.ndarray:
    ph, pw = th - t.shape[0], tw - t.shape[1]
    if ph <= 0 and pw <= 0:
        return t
    mode = "reflect" if t.shape[0] > ph and t.shape[1] > pw else "edge"
    return np.pad(t, ((0, max(ph, 0)), (0, max(pw, 0)), (0, 0)), mode=mode)

def _calibrate_onnx(sess, sample: np.ndarray, layout: str, np_dtype, name: str) -> dict:
    """Run one sample tile and pick input/output value ranges. Raises OnnxModelError."""
    def stats(o):
        return float(np.nanmin(o)), float(np.nanmax(o)), float(np.nanmean(o))
    in_max = float(sample.max())
    out = _onnx_forward(sess, sample, layout, 1.0, np_dtype)
    if not np.isfinite(out).all():
        raise OnnxModelError(
            f"{name}: model output contains NaN/Inf (typical for fp16 models that overflow). "
            f"Try the fp32 version of this model.")
    mn, mx, mean = stats(out)
    log(f"[onnx] calibrate {name}: in 0..1 → out min={mn:.4f} max={mx:.4f} mean={mean:.4f}")
    in_mul, out_div = 1.0, 1.0
    if mx > 2.0:
        out_div = 255.0                    # model outputs 0..255
    elif in_max > 0.1 and mx < 0.02:
        # Near-black output for a non-black input: model probably expects 0..255 input
        out2 = _onnx_forward(sess, sample, layout, 255.0, np_dtype)
        if np.isfinite(out2).all():
            mn2, mx2, mean2 = stats(out2)
            log(f"[onnx] calibrate {name}: in 0..255 → out min={mn2:.4f} max={mx2:.4f} mean={mean2:.4f}")
            if mx2 > 2.0:
                in_mul, out_div = 255.0, 255.0
            elif mx2 > 0.1:
                in_mul = 255.0
        if in_mul == 1.0:
            raise OnnxModelError(
                f"{name}: model returns an (almost) black image for both 0..1 and 0..255 input "
                f"(max output {mx:.4f}). The model may be incompatible or need a different input format.")
    return {"in_mul": in_mul, "out_div": out_div}

def _probe_onnx_scale(path: Path) -> int:
    """Determine an ONNX model's upscale factor with a small CPU inference. 0 if unknown."""
    if not HAS_ONNX:
        return 0
    try:
        sess = loaded_models.get(str(path)) or ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"])
        layout, sh, sw, np_dtype = _onnx_input_info(sess)
        h, w = (sh, sw) if (sh and sw) else (32, 32)
        x = np.full((1, 3, h, w) if layout == "NCHW" else (1, h, w, 3), 0.5, dtype=np_dtype)
        out = sess.run(None, {sess.get_inputs()[0].name: x})[0]   # not cancel-aware on purpose
        oh = out.shape[2] if (layout == "NCHW" and out.ndim == 4) else out.shape[1] if out.ndim == 4 else out.shape[-2]
        return max(1, round(oh / h))
    except Exception as e:
        log(f"[model-scale] ONNX probe failed for {path.name}: {e}")
        return 0

def run_onnx(path: Path, img: Image.Image,
             model_idx=0, model_total=1,
             pipe_step=0, pipe_total=0, tile_size=512) -> Image.Image:
    """Tiled ONNX inference with overlap blending, auto value-range calibration,
    NCHW/NHWC detection, static-shape support and automatic tile shrinking on OOM."""
    t_total = time.perf_counter()
    check_cancel()
    sess = ensure_loaded(path, model_idx, model_total)
    check_cancel()
    key = str(path)
    layout, sh, sw, np_dtype = _onnx_input_info(sess)

    rgb = np.asarray(to_rgb(img), dtype=np.float32) / 255.0      # HWC RGB 0..1
    H, W = rgb.shape[:2]

    prof = _onnx_profiles.get(key)
    if prof is None:
        cs = min(64, H, W) if not (sh and sw) else None
        if sh and sw:
            sample = _pad_tile(rgb[:sh, :sw], sh, sw)
        else:
            sample = rgb[:cs, :cs]
            m = 16
            sample = _pad_tile(sample, -(-cs // m) * m, -(-cs // m) * m)
        prof = _calibrate_onnx(sess, sample, layout, np_dtype, path.name)
        test = _onnx_forward(sess, sample, layout, prof["in_mul"], np_dtype)
        prof["scale"] = max(1, round(test.shape[0] / sample.shape[0]))
        prof["layout"] = layout
        _onnx_profiles[key] = prof
        if model_meta.get(path.name, {}).get("scale") != prof["scale"]:
            model_meta[path.name] = {**model_meta.get(path.name, {}), "scale": prof["scale"]}
            save_model_meta(model_meta)
        log(f"[onnx] profile {path.name}: {prof}  input={sess.get_inputs()[0].shape} {sess.get_inputs()[0].type}")
    sc, in_mul, out_div = prof["scale"], prof["in_mul"], prof["out_div"]

    # Tile geometry: static models dictate the tile; dynamic ones use the user's tile size
    if sh and sw:
        tiles_hw = [(sh, sw)]
    else:
        ts = max(64, int(tile_size))
        tiles_hw = [(t, t) for t in (ts, ts // 2, ts // 4, ts // 8) if t >= 64] or [(64, 64)]

    pct_start = 10 + model_idx * 55 // model_total
    pct_end   = 10 + (model_idx + 1) * 55 // model_total
    last_err = None
    for th, tw in tiles_hw:
        try:
            out = _onnx_tiled(sess, rgb, th, tw, sc, layout, in_mul, out_div, np_dtype,
                              pct_start, pct_end, pipe_step, pipe_total, static=bool(sh and sw))
            break
        except CancelledError:
            raise
        except Exception as e:
            if not _is_oom(e) or (sh and sw):
                raise
            last_err = e
            log(f"[onnx] out of memory at tile {th}×{tw} — retrying with smaller tiles")
            set_status(pct_start, f"Out of memory — retrying with smaller tiles…", step="inference",
                       pipe_step=pipe_step, pipe_total=pipe_total)
    else:
        raise OnnxModelError(
            f"{path.name}: out of memory even with 64 px tiles. This model needs more VRAM/RAM "
            f"than available. ({last_err})")

    set_status(pct_end, "done", step="inference")
    log(f"[onnx] TOTAL {path.name}  {W}×{H} → {out.shape[1]}×{out.shape[0]}  {time.perf_counter()-t_total:.2f}s")
    return Image.fromarray((np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="RGB")

def _onnx_tiled(sess, rgb, th, tw, sc, layout, in_mul, out_div, np_dtype,
                pct_start, pct_end, pipe_step, pipe_total, static=False):
    H, W = rgb.shape[:2]
    ov = min(16, th // 8, tw // 8)                        # overlap in input px
    sy, sx = max(1, th - 2 * ov), max(1, tw - 2 * ov)
    ys = list(range(0, max(H - th, 0) + 1, sy)) or [0]
    xs = list(range(0, max(W - tw, 0) + 1, sx)) or [0]
    if ys[-1] + th < H: ys.append(H - th)
    if xs[-1] + tw < W: xs.append(W - tw)
    ys = sorted(set(max(0, y) for y in ys)); xs = sorted(set(max(0, x) for x in xs))
    acc = np.zeros((H * sc, W * sc, 3), np.float32)
    wgt = np.zeros((H * sc, W * sc, 1), np.float32)
    total, done = len(ys) * len(xs), 0
    for y in ys:
        for x in xs:
            check_cancel()
            tile = rgb[y:y + th, x:x + tw]
            h0, w0 = tile.shape[:2]
            if static:
                ph, pw = th, tw
            else:
                ph, pw = -(-h0 // 16) * 16, -(-w0 // 16) * 16    # multiple of 16 (window sizes)
            o = _onnx_forward(sess, _pad_tile(tile, ph, pw), layout, in_mul, np_dtype)
            if not np.isfinite(o).all():
                raise OnnxModelError("Model output contains NaN/Inf (fp16 overflow?). "
                                     "Try the fp32 version of this model.")
            o = o[:h0 * sc, :w0 * sc] / out_div
            # Feathered weights: inner pixels count more than tile borders
            fy = np.minimum(np.arange(h0 * sc) + 1, np.arange(h0 * sc)[::-1] + 1).astype(np.float32)
            fx = np.minimum(np.arange(w0 * sc) + 1, np.arange(w0 * sc)[::-1] + 1).astype(np.float32)
            wmap = np.minimum.outer(np.minimum(fy, ov * sc + 1), np.minimum(fx, ov * sc + 1))[:, :, None]
            acc[y * sc:y * sc + h0 * sc, x * sc:x * sc + w0 * sc] += o * wmap
            wgt[y * sc:y * sc + h0 * sc, x * sc:x * sc + w0 * sc] += wmap
            done += 1
            set_status(pct_start + (pct_end - pct_start) * done // total, f"tile {done}/{total}",
                       step="inference", tile_current=done, tile_total=total,
                       pipe_step=pipe_step, pipe_total=pipe_total)
    return acc / np.maximum(wgt, 1e-6)

# ── Pipeline ───────────────────────────────────────────────────────────────────
def apply_action(img: Image.Image, action: dict, step_idx: int, total: int) -> Image.Image:
    """Apply a single action step to img. Returns modified image."""
    atype = action.get("type", "")
    pct_base = 10 + step_idx * 55 // total
    set_status(pct_base, f"Action: {atype}…", step="action")

    # Helper: apply fn to RGB channels only, preserve alpha if present
    def apply_rgb(fn):
        nonlocal img
        has_a = _has_alpha(img)
        if has_a:
            rgba = img.convert("RGBA")
            r, g, b, a = rgba.split()
            rgb_img = Image.merge("RGB", (r, g, b))
            result_rgb = fn(rgb_img)
            r2, g2, b2 = result_rgb.convert("RGB").split()
            img = Image.merge("RGBA", (r2, g2, b2, a))
        else:
            img = fn(img.convert("RGB"))

    if atype == "resize":
        mode_str = action.get("mode", "percent")
        channels = action.get("channels", "all")
        filt_map = {"lanczos": Image.LANCZOS, "bicubic": Image.BICUBIC,
                    "nearest": Image.NEAREST, "bilinear": Image.BILINEAR}
        filt = filt_map.get(action.get("filter", "lanczos"), Image.LANCZOS)
        w, h = img.size
        if mode_str == "percent":
            pct = float(action.get("value", 100)) / 100.0
            nw, nh = max(1, int(w * pct)), max(1, int(h * pct))
        elif mode_str == "pixels":
            nw = int(action.get("width", w))
            nh = int(action.get("height", h))
        elif mode_str == "longest_side":
            longest = int(action.get("value", max(w, h)))
            scale = longest / max(w, h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        elif mode_str == "shortest_side":
            shortest = int(action.get("value", min(w, h)))
            scale = shortest / min(w, h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        else:
            nw, nh = w, h
        # channels='alpha': only the alpha track resizes; RGB track skips
        if channels != "alpha":
            if _has_alpha(img):
                img = img.convert("RGBA").resize((nw, nh), filt)
            else:
                img = img.resize((nw, nh), filt)

    elif atype == "flip":
        if action.get("flip_x"): img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if action.get("flip_y"): img = img.transpose(Image.FLIP_TOP_BOTTOM)

    elif atype == "crop":
        mode_str = action.get("mode", "percent")
        w, h = img.size
        if mode_str == "percent":
            l = int(w * float(action.get("left",   0)) / 100)
            t = int(h * float(action.get("top",    0)) / 100)
            r = int(w * float(action.get("right",  0)) / 100)
            b = int(h * float(action.get("bottom", 0)) / 100)
        else:
            l = int(action.get("left",   0))
            t = int(action.get("top",    0))
            r = int(action.get("right",  0))
            b = int(action.get("bottom", 0))
        box = (l, t, max(l+1, w-r), max(t+1, h-b))
        img = img.crop(box)

    elif atype == "hsl":
        h_shift = float(action.get("hue",        0)) / 360.0
        s_shift = float(action.get("saturation", 0)) / 100.0
        l_shift = float(action.get("lightness",  0)) / 100.0
        def _hsl(rgb_img):
            # Fast path: OpenCV is ~10× faster than numpy for colour-space ops
            try:
                import cv2 as _cv2
                arr = np.array(rgb_img, dtype=np.uint8)
                hls = _cv2.cvtColor(arr, _cv2.COLOR_RGB2HLS).astype(np.float32)
                # cv2 HLS ranges: H [0,180], L [0,255], S [0,255]
                hls[:,:,0] = ((hls[:,:,0] / 180.0 + h_shift) % 1.0) * 180.0
                hls[:,:,1] = np.clip(hls[:,:,1] / 255.0 + l_shift, 0.0, 1.0) * 255.0
                hls[:,:,2] = np.clip(hls[:,:,2] / 255.0 + s_shift, 0.0, 1.0) * 255.0
                return Image.fromarray(
                    _cv2.cvtColor(np.round(hls).clip(0, 255).astype(np.uint8),
                                  _cv2.COLOR_HLS2RGB))
            except ImportError:
                pass
            # Pure-numpy fallback: vectorised RGB→HLS→RGB (no Python loops)
            arr = np.array(rgb_img, dtype=np.float32) / 255.0
            r, g, b = arr[...,0], arr[...,1], arr[...,2]
            cmax = np.maximum(r, np.maximum(g, b))
            cmin = np.minimum(r, np.minimum(g, b))
            delta = cmax - cmin
            # Lightness + saturation
            l = (cmax + cmin) * 0.5
            denom = 1.0 - np.abs(2.0*l - 1.0)
            s = np.where(denom > 1e-7, delta / np.maximum(denom, 1e-7), 0.0).clip(0.0, 1.0)
            # Hue — safe reciprocal, no masked per-pixel work
            inv_d = np.where(delta > 1e-7, 1.0 / np.maximum(delta, 1e-7), 0.0)
            h = np.where(delta > 1e-7,
                    np.where(cmax == r, ((g - b) * inv_d) % 6.0,
                    np.where(cmax == g,  (b - r) * inv_d + 2.0,
                                         (r - g) * inv_d + 4.0)) / 6.0,
                    0.0)
            # Apply shifts
            h = (h + h_shift) % 1.0
            s = (s + s_shift).clip(0.0, 1.0)
            l = (l + l_shift).clip(0.0, 1.0)
            # HLS→RGB via np.choose — much faster than np.select (direct indexing)
            c = (1.0 - np.abs(2.0*l - 1.0)) * s
            m = l - c * 0.5
            h6  = h * 6.0
            h6i = np.floor(h6).astype(np.int32) % 6
            x   = c * (1.0 - np.abs(h6 % 2.0 - 1.0))
            z   = np.zeros_like(c)
            r2  = np.choose(h6i, [c, x, z, z, x, c]) + m
            g2  = np.choose(h6i, [x, c, c, x, z, z]) + m
            b2  = np.choose(h6i, [z, z, x, c, c, x]) + m
            return Image.fromarray(
                (np.stack([r2, g2, b2], axis=-1).clip(0.0, 1.0) * 255.0).astype(np.uint8))
        apply_rgb(_hsl)

    elif atype == "levels":
        in_black  = float(action.get("in_black",  0))   / 255.0
        in_white  = float(action.get("in_white",  255)) / 255.0
        out_black = float(action.get("out_black", 0))   / 255.0
        out_white = float(action.get("out_white", 255)) / 255.0
        gamma     = float(action.get("gamma", 1.0))
        def _levels(rgb_img):
            arr = np.array(rgb_img, dtype=np.float32) / 255.0
            arr = (arr - in_black) / max(in_white - in_black, 1e-6)
            arr = np.clip(arr, 0, 1) ** (1.0 / max(gamma, 0.01))
            arr = arr * (out_white - out_black) + out_black
            return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
        apply_rgb(_levels)

    elif atype == "brightness_contrast":
        brightness = float(action.get("brightness", 0)) / 100.0
        contrast   = float(action.get("contrast",   0)) / 100.0
        channel    = action.get("channel", "rgb")

        if channel == "alpha":
            # Apply only to alpha channel
            if _has_alpha(img):
                rgba = img.convert("RGBA")
                r, g, b, a = rgba.split()
                arr = np.array(a, dtype=np.float32) / 255.0
                arr = arr + brightness
                arr = (arr - 0.5) * (1.0 + contrast) + 0.5
                a2 = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L")
                img = Image.merge("RGBA", (r, g, b, a2))
            # if no alpha, do nothing
        else:
            # Apply to RGB, preserve alpha
            def _bc(rgb_img):
                arr = np.array(rgb_img, dtype=np.float32) / 255.0
                arr = arr + brightness
                arr = (arr - 0.5) * (1.0 + contrast) + 0.5
                return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
            apply_rgb(_bc)

    elif atype == "color_temperature":
        temp = float(action.get("temperature", 0))
        r_adj = temp / 100.0 * 0.15
        b_adj = -temp / 100.0 * 0.15
        def _temp(rgb_img):
            arr = np.array(rgb_img, dtype=np.float32) / 255.0
            arr[:,:,0] = np.clip(arr[:,:,0] + r_adj, 0, 1)
            arr[:,:,2] = np.clip(arr[:,:,2] + b_adj, 0, 1)
            return Image.fromarray((arr * 255).astype(np.uint8))
        apply_rgb(_temp)

    elif atype == "invert":
        def _inv(rgb_img):
            return Image.fromarray(255 - np.array(rgb_img, dtype=np.uint8))
        apply_rgb(_inv)

    elif atype == "grayscale":
        method = action.get("method", "luminosity")
        def _gray(rgb_img):
            arr = np.array(rgb_img, dtype=np.float32)
            if method == "luminosity":
                gray = 0.2126*arr[:,:,0] + 0.7152*arr[:,:,1] + 0.0722*arr[:,:,2]
            elif method == "average":
                gray = arr.mean(axis=2)
            else:  # lightness
                gray = (arr.max(axis=2) + arr.min(axis=2)) / 2
            gray8 = np.clip(gray, 0, 255).astype(np.uint8)
            return Image.fromarray(np.stack([gray8]*3, axis=2))
        apply_rgb(_gray)

    elif atype == "alpha_threshold":
        if _has_alpha(img):
            thresh = int(action.get("threshold", 128))
            arr = np.array(img.convert("RGBA"))
            alpha = arr[:,:,3]
            alpha[alpha <  thresh] = 0
            alpha[alpha >= thresh] = 255
            arr[:,:,3] = alpha
            img = Image.fromarray(arr, mode="RGBA")

    elif atype == "color_extend":
        # Extend RGB colours into transparent regions orthogonally to the alpha
        # edge — each transparent pixel gets the colour of its nearest opaque
        # neighbour (Voronoi / nearest-neighbour inpainting).
        # Technique: https://shahriyarshahrabi.medium.com/padding-transparent-textures-fir-mips-and-game-engines-c71c085142fe
        #
        # radius=0 → fill ALL transparent pixels (unlimited).
        # radius>0 → only fill pixels within `radius` px of an opaque edge.
        #
        # Implementation uses scipy.ndimage.distance_transform_edt which computes,
        # for every transparent pixel, both the euclidean distance to the nearest
        # opaque pixel AND the coordinates of that pixel — in a single fast pass.
        if _has_alpha(img):
            from scipy.ndimage import distance_transform_edt
            radius = int(action.get("radius", 0))
            arr   = np.array(img.convert("RGBA"), dtype=np.uint8)
            alpha = arr[:, :, 3]
            opaque = alpha > 0          # mask of pixels with valid colour

            # distance_transform_edt on the *inverted* mask gives, for each
            # transparent pixel, the distance to the nearest opaque pixel and
            # (via return_indices) the row/col of that nearest opaque pixel.
            distances, indices = distance_transform_edt(~opaque, return_indices=True)

            # Build output: start with original, then overwrite transparent pixels
            out = arr.copy()
            if radius > 0:
                fill_mask = (~opaque) & (distances <= radius)
            else:
                fill_mask = ~opaque

            rows = indices[0][fill_mask]
            cols = indices[1][fill_mask]
            out[fill_mask, 0] = arr[rows, cols, 0]
            out[fill_mask, 1] = arr[rows, cols, 1]
            out[fill_mask, 2] = arr[rows, cols, 2]
            # Alpha always preserved verbatim
            out[:, :, 3] = alpha
            img = Image.fromarray(out, mode="RGBA")
        # No alpha channel → no-op

    return img


def _run_model_file(mp: Path, rgb_img: Image.Image, tile_size: int,
                    model_idx: int, model_total: int, pipe_step: int, pipe_total: int) -> Image.Image:
    """Run any supported model file (PyTorch via spandrel, or ONNX) on an RGB image."""
    sfx = mp.suffix.lower()
    if sfx in (".pth", ".safetensors", ".pt"):
        return run_spandrel(mp, rgb_img, tile_size, model_idx=model_idx, model_total=model_total,
                            pipe_step=pipe_step, pipe_total=pipe_total)
    if sfx == ".onnx":
        return run_onnx(mp, rgb_img, model_idx=model_idx, model_total=model_total,
                        pipe_step=pipe_step, pipe_total=pipe_total, tile_size=tile_size)
    raise ValueError(f"Unsupported model format: {sfx}")

def process_pipeline(img: Image.Image, pipeline: list, pre_scale: float,
                     post_scale: float, tile_size: int) -> Image.Image:
    """
    pipeline: list of dicts, each either:
      {"type":"model", "path": "/abs/path/model.pth"}  — run upscale model
      {"type":"resize"|"flip"|..., ...params}           — run action
    pre_scale / post_scale are legacy global scales applied first/last.
    """
    t_pipe = time.perf_counter()

    # Count total steps for progress
    total = len(pipeline) + (1 if pre_scale != 1.0 else 0) + (1 if post_scale != 1.0 else 0)
    step = 0

    if pre_scale != 1.0:
        w, h = img.size
        set_status(4, f"Pre-scaling {pre_scale}×…", step="prescale",
                   pipe_step=1, pipe_total=total)
        t0 = time.perf_counter()
        img = img.resize((int(w*pre_scale), int(h*pre_scale)), Image.LANCZOS)
        log(f"[pipe] pre-scale {pre_scale}×  {time.perf_counter()-t0:.3f}s")
        step += 1

    # Count only models for model-index tracking (tile progress)
    model_steps = [p for p in pipeline if p.get("type") == "model"]
    model_total_count = (len(model_steps)
                         + sum(2 for p in pipeline if p.get("type") == "interpolate")
                         + sum(1 for p in pipeline if p.get("type") in ("discrete", "overlay")))
    model_idx = 0

    for i, p in enumerate(pipeline):
        check_cancel()
        _FILT = {"lanczos":Image.LANCZOS,"bicubic":Image.BICUBIC,
                 "bilinear":Image.BILINEAR,"nearest":Image.NEAREST}

        ptype = p.get("type", "model")
        if ptype == "interpolate":
            path_a = p.get("model_a", "")
            path_b = p.get("model_b", "")
            factor = float(p.get("factor", 0.5))
            ch     = p.get("channels", "all")
            of     = _FILT.get(p.get("other_filter", "lanczos"), Image.LANCZOS)
            scale_f= float(p.get("scale", 0) or 0)
            if not path_a or not path_b:
                log(f"[pipe] interpolate: skipping — missing models")
            elif p.get("invalid"):
                log(f"[pipe] interpolate: skipping — incompatible models")
            elif ch == "alpha":
                # Alpha-only: resize RGB to expected output size; alpha track runs the model
                if scale_f > 0:
                    nw, nh = max(1,round(img.width*scale_f)), max(1,round(img.height*scale_f))
                    alpha_ch2 = img.getchannel("A") if img.mode=="RGBA" else None
                    img = to_rgb(img).resize((nw, nh), of)
                    if alpha_ch2:
                        img = Image.merge("RGBA", (*img.split(), alpha_ch2.resize((nw,nh),of)))
            else:
                mp_a, mp_b = Path(path_a), Path(path_b)
                log(f"[pipe] interpolate {mp_a.name} ×{1-factor:.2f} + {mp_b.name} ×{factor:.2f}")
                set_status(10 + step * 55 // max(total,1), "Interpolating…",
                           step="interpolate", pipe_step=step+1, pipe_total=total)
                alpha_ch = img.getchannel("A") if img.mode == "RGBA" else None
                rgb_img = to_rgb(img)
                arr_a_pil = run_spandrel(mp_a, rgb_img, tile_size,
                    model_idx=model_idx, model_total=model_total_count,
                    pipe_step=step+1, pipe_total=total)
                model_idx += 1
                arr_b_pil = run_spandrel(mp_b, rgb_img, tile_size,
                    model_idx=model_idx, model_total=model_total_count,
                    pipe_step=step+1, pipe_total=total)
                model_idx += 1
                import numpy as _np
                arr_a = _np.array(arr_a_pil, dtype=_np.float32)
                arr_b = _np.array(arr_b_pil, dtype=_np.float32)
                blended = _np.clip(arr_a*(1.0-factor) + arr_b*factor, 0, 255).astype(_np.uint8)
                img = Image.fromarray(blended)
                if alpha_ch and ch == "all":
                    img = Image.merge("RGBA", (*img.split(), alpha_ch.resize(img.size, of)))
        elif ptype == "model":
            mp = Path(p["path"])
            s  = mp.suffix.lower()
            ch     = p.get("channels", "all")
            of     = _FILT.get(p.get("other_filter", "lanczos"), Image.LANCZOS)
            scale_f= float(p.get("scale", 0) or 0)
            if ch == "alpha":
                # Alpha-only: resize RGB to expected output size; alpha track runs the model
                if scale_f > 0:
                    nw, nh = max(1,round(img.width*scale_f)), max(1,round(img.height*scale_f))
                    alpha_ch2 = img.getchannel("A") if img.mode=="RGBA" else None
                    img = to_rgb(img).resize((nw, nh), of)
                    if alpha_ch2:
                        img = Image.merge("RGBA", (*img.split(), alpha_ch2.resize((nw,nh),of)))
                model_idx += 1
            else:
                log(f"[pipe] model {model_idx+1}/{len(model_steps)}: {mp.name}")
                alpha_ch = img.getchannel("A") if img.mode == "RGBA" else None
                rgb_img = to_rgb(img)
                if s in (".pth", ".safetensors", ".pt"):
                    img = run_spandrel(mp, rgb_img, tile_size,
                                       model_idx=model_idx, model_total=len(model_steps),
                                       pipe_step=step+1, pipe_total=total)
                elif s == ".onnx":
                    img = run_onnx(mp, rgb_img, model_idx=model_idx, model_total=len(model_steps),
                                   pipe_step=step+1, pipe_total=total, tile_size=tile_size)
                else:
                    raise ValueError(f"Unsupported: {s}")
                if alpha_ch and ch == "all":
                    img = Image.merge("RGBA", (*img.split(), alpha_ch.resize(img.size, of)))
                model_idx += 1
        elif ptype == "overlay":
            # Run a model on the previous step's result and blend its output over that result
            # (previous result = base, resized with other_filter to the model's output size).
            path_m = p.get("model", "")
            ch     = p.get("channels", "all")
            of     = _FILT.get(p.get("other_filter", "lanczos"), Image.LANCZOS)
            scale_f= float(p.get("scale", 0) or 0)
            mode_b = p.get("blend_mode", "normal")
            opac   = float(p.get("opacity", 100)) / 100.0
            if not path_m:
                log(f"[pipe] overlay: skipping — no model assigned")
            elif ch == "alpha":
                # Alpha-only: RGB is just resized; the alpha track runs model + blend
                if scale_f > 0:
                    nw, nh = max(1,round(img.width*scale_f)), max(1,round(img.height*scale_f))
                    alpha_ch2 = img.getchannel("A") if img.mode=="RGBA" else None
                    img = to_rgb(img).resize((nw, nh), of)
                    if alpha_ch2:
                        img = Image.merge("RGBA", (*img.split(), alpha_ch2.resize((nw,nh),of)))
                model_idx += 1
            else:
                mp = Path(path_m)
                log(f"[pipe] overlay {mp.name}  mode={mode_b}  opacity={opac:.2f}"
                    f"{'  (alpha track)' if p.get('_alpha_track') else ''}")
                set_status(10 + step * 55 // max(total,1), f"Overlay: {mp.name}…",
                           step="overlay", pipe_step=step+1, pipe_total=total)
                alpha_ch = img.getchannel("A") if img.mode == "RGBA" else None
                base_rgb = to_rgb(img)
                top = _run_model_file(mp, base_rgb, tile_size, model_idx, model_total_count, step+1, total)
                check_cancel()
                if base_rgb.size != top.size:
                    base_rgb = base_rgb.resize(top.size, of)
                # Alpha track: the blend mode is ignored, alpha is simply cross-faded by opacity
                if p.get("_alpha_track"):
                    mode_b = "normal"
                img = blend_images(base_rgb, top, mode_b, opac)
                if alpha_ch and ch == "all":
                    img = Image.merge("RGBA", (*img.split(), alpha_ch.resize(img.size, of)))
                model_idx += 1
        elif ptype == "discrete":
            # Dual: run model_rgb on RGB channel only; alpha model is handled in the alpha track
            path_rgb = p.get("model_rgb", "")
            if path_rgb:
                mp = Path(path_rgb)
                s  = mp.suffix.lower()
                log(f"[pipe] discrete RGB model: {mp.name}")
                set_status(10 + step * 55 // max(total,1), f"Dual RGB: {mp.name}…",
                           step="discrete_rgb", pipe_step=step+1, pipe_total=total)
                alpha_ch = img.getchannel("A") if img.mode == "RGBA" else None
                rgb_img  = to_rgb(img)
                if s in (".pth", ".safetensors", ".pt"):
                    img = run_spandrel(mp, rgb_img, tile_size,
                                       model_idx=model_idx, model_total=model_total_count,
                                       pipe_step=step+1, pipe_total=total)
                elif s == ".onnx":
                    img = run_onnx(mp, rgb_img, model_idx=model_idx, model_total=model_total_count,
                                   pipe_step=step+1, pipe_total=total, tile_size=tile_size)
                else:
                    raise ValueError(f"Unsupported model format: {s}")
                # Do NOT reattach alpha — the alpha track runs its own model separately
                log(f"[pipe] discrete RGB done  {img.size}")
            model_idx += 1
        else:
            if p.get("type") == "color_extend" and not _has_alpha(img):
                # RGB-only track: we need to reattach the current alpha before dilating.
                # alpha_cur is injected into the action dict by process_image for this case.
                alpha_cur = p.get("_alpha_proxy")  # mode-L image or None
                if alpha_cur is not None:
                    # Resize alpha proxy to match current img size if they differ
                    # (pre-scale may have changed dimensions already)
                    if alpha_cur.size != img.size:
                        alpha_cur = alpha_cur.resize(img.size, Image.LANCZOS)
                    rgba_tmp = Image.merge("RGBA", (*img.split(), alpha_cur))
                    rgba_out = apply_action(rgba_tmp, p, step, total)
                    # Strip alpha back off — alpha track handles it separately
                    img = rgba_out.convert("RGB")
                else:
                    img = apply_action(img, p, step, total)
            else:
                img = apply_action(img, p, step, total)
        step += 1

    if post_scale != 1.0:
        w, h = img.size
        set_status(76, f"Post-scaling {post_scale}×…", step="postscale")
        t0 = time.perf_counter()
        img = img.resize((int(w*post_scale), int(h*post_scale)), Image.LANCZOS)
        log(f"[pipe] post-scale {post_scale}×  {time.perf_counter()-t0:.3f}s")

    log(f"[pipe] TOTAL pipeline  {time.perf_counter()-t_pipe:.3f}s")
    return img

def _warmup_model(md):
    """Run a small forward pass to trigger cuDNN kernel search.
    Uses 32×32 (same as the fp16 compatibility test in ensure_loaded)."""
    if not HAS_TORCH or DEVICE == "cpu":
        log("[warmup] skipped (CPU mode)")
        return
    try:
        log("[warmup] START dummy forward pass")
        t0 = time.perf_counter()
        model_dtype = next(md.model.parameters()).dtype
        dummy = torch.zeros(1, 3, 32, 32, dtype=model_dtype, device=DEVICE)
        with torch.no_grad():
            md.model(dummy)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        log(f"[warmup] DONE  {time.perf_counter()-t0:.2f}s  dtype={model_dtype}")
    except Exception as e:
        log(f"[warmup] failed (non-fatal): {e}")


# ── Background preload ─────────────────────────────────────────────────────────
def preload_models_bg(paths: list):
    def _load():
        for i, p in enumerate(paths):
            key = str(p)
            already_cached = key in loaded_models
            try:
                md = ensure_loaded(p, i, len(paths))
                # Warmup: trigger CUDA context + cuDNN kernel search now
                if not already_cached and hasattr(md, 'model'):
                    n = len(paths)
                    processing_status.update({
                        "running": True,
                        "progress": 10 + (i * 60 // n) + 30,
                        "message": f"Warming up {p.name}…",
                        "step": "warmup", "tile_current": 0, "tile_total": 0,
                    })
                    _warmup_model(md)
            except Exception:
                pass
        processing_status.update({"running": False, "progress": 0, "message": "",
                                   "step": "", "tile_current": 0, "tile_total": 0})
    threading.Thread(target=_load, daemon=True).start()

# ── Model tree ─────────────────────────────────────────────────────────────────
def build_tree(root: Path) -> dict:
    if not root.exists():
        return {"name": root.name, "path": str(root),
                "children": [], "models": [], "exists": False}
    def scan(p: Path) -> dict:
        node = {"name": p.name, "path": str(p), "children": [], "models": []}
        try:
            entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
        except PermissionError:
            return node
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                child = scan(entry)
                if child["models"] or child["children"]:
                    node["children"].append(child)
            elif entry.is_file() and entry.suffix.lower() in MODEL_EXTENSIONS:
                node["models"].append({
                    "name": entry.name,
                    "path": str(entry),
                    "size_mb": round(entry.stat().st_size / 1024 / 1024, 1),
                    "type": "onnx" if entry.suffix.lower() == ".onnx" else "pytorch",
                    "cached": str(entry) in loaded_models,
                })
        return node
    return scan(root)

# ── API ────────────────────────────────────────────────────────────────────────

@app.get("/api/config")
def get_config():
    return {**cfg, "device": DEVICE, "fp16": USE_FP16, "version": APP_VERSION,
            "canvas_limit": cfg.get("canvas_limit", 8192)}

@app.post("/api/open-folder")
async def open_folder(path: str = Form(...)):
    """Open a folder in the system file manager."""
    import platform
    p = Path(path)
    if p.is_file():
        p = p.parent
    if not p.exists():
        raise HTTPException(404, "Path not found")
    try:
        if platform.system() == "Windows":
            import os; os.startfile(str(p))
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p)])
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/config/canvas-limit")
async def set_canvas_limit(limit: int = Form(...)):
    valid = [4096, 8192, 12288, 16384]   # pixel budget = value²; 16384² is Chromium's canvas maximum
    limit = min(valid, key=lambda x: abs(x - limit))
    cfg["canvas_limit"] = limit
    save_config(cfg)
    return {"canvas_limit": limit}


@app.post("/api/config/quick-model")
async def set_quick_model_cfg(path: str = Form("")):
    cfg["last_quick_model"] = path
    save_config(cfg)
    return {"ok": True, "path": path}

@app.post("/api/save-as")
async def save_as(file: UploadFile = File(...), dest_path: str = Form(...)):
    """Write an uploaded blob to a user-chosen absolute path."""
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = await file.read()
    dest.write_bytes(data)
    log(f"[save-as] {dest}  ({len(data)//1024} KB)")
    return {"saved_to": str(dest)}

@app.get("/api/presets")
def get_presets():
    return {"presets": presets}

@app.post("/api/presets/{name}")
def save_preset(name: str, data: str = Form(...)):
    global presets
    presets[name] = json.loads(data)
    save_presets(presets)
    return {"ok": True, "names": list(presets.keys())}

@app.delete("/api/presets/{name}")
def delete_preset(name: str):
    global presets
    presets.pop(name, None)
    save_presets(presets)
    return {"ok": True, "names": list(presets.keys())}

# ── Batch processing ────────────────────────────────────────────────────────────
batch_status: dict = {"running": False, "total": 0, "done": 0, "current": "", "errors": []}

@app.get("/api/batch/status")
def get_batch_status():
    return batch_status

@app.post("/api/batch/run")
async def run_batch(
    image_paths:   str  = Form(...),   # newline-separated absolute paths
    model_paths:   str  = Form(...),
    pre_scale:     float = Form(1.0),
    post_scale:    float = Form(1.0),
    tile_size:     int   = Form(512),
    output_format: str   = Form("PNG"),
    upscale_alpha: bool  = Form(True),
    output_dir:    str   = Form(""),   # empty = use source image dir
    use_src_paths: bool  = Form(True),
):
    global batch_status
    if batch_status["running"]:
        raise HTTPException(400, "Batch already running")

    imgs  = [p.strip() for p in image_paths.splitlines() if p.strip()]
    paths = [Path(p.strip()) for p in model_paths.split("||") if p.strip()]
    out_base = Path(output_dir) if output_dir and not use_src_paths else None

    batch_status = {"running": True, "total": len(imgs), "done": 0,
                    "current": "", "errors": []}

    def _run():
        global batch_status
        for img_path_str in imgs:
            ip = Path(img_path_str)
            batch_status["current"] = ip.name
            try:
                src = Image.open(ip)
                has_alpha = _has_alpha(src)
                alpha_ch  = src.convert("RGBA").getchannel("A") if has_alpha and upscale_alpha else None
                img = to_rgb(src); src.close()
                result = process_pipeline(img, paths, pre_scale, post_scale, tile_size)

                if alpha_ch and upscale_alpha:
                    alpha_rgb = Image.merge("RGB",[alpha_ch]*3)
                    ar = process_pipeline(alpha_rgb, paths, pre_scale, post_scale, tile_size)
                    ua = ar.getchannel("R")
                else:
                    ua = None

                from datetime import datetime
                model_steps = [s for s in pipeline_json if s.get("type") == "model"]
                non_model   = [s for s in pipeline_json if s.get("type") not in ("model",)]
                if len(model_steps) == 1 and not non_model:
                    ms = Path(model_steps[0]["path"]).stem
                else:
                    ms = "pipeline_" + datetime.now().strftime("%Y%m%d%H%M%S")
                fname = f"{ip.stem}__{ms}.{output_format.lower()}"

                if use_src_paths and not ip_is_clipboard(ip):
                    dest = ip.parent / fname
                else:
                    dest = (out_base or OUTPUT_DIR) / fname

                dest.parent.mkdir(parents=True, exist_ok=True)
                fmt = "JPEG" if output_format.upper()=="JPG" else output_format.upper()
                save_img = Image.merge("RGBA",(*result.split(),ua)) if ua and fmt in ("PNG","TIFF","WEBP") else result
                save_img.save(dest, format=fmt, **({"quality":95} if fmt=="JPEG" else {}))

            except Exception as e:
                batch_status["errors"].append({"file": ip.name, "error": str(e)})
            batch_status["done"] += 1

        batch_status["running"] = False
        batch_status["current"] = ""

    threading.Thread(target=_run, daemon=True).start()
    return {"started": True, "total": len(imgs)}

def ip_is_clipboard(p: Path) -> bool:
    return p.stem.startswith("clipboard_") or str(OUTPUT_DIR) in str(p)

@app.post("/api/batch/stop")
def stop_batch():
    # Soft stop — set flag, thread checks it
    batch_status["running"] = False
    return {"ok": True}

@app.post("/api/config/models-dir")
def set_models_dir(path: str = Form(...)):
    p = Path(path).expanduser().resolve()
    if not p.exists(): raise HTTPException(400, f"Does not exist: {p}")
    if not p.is_dir(): raise HTTPException(400, f"Not a directory: {p}")
    cfg["models_dir"] = str(p)
    save_config(cfg)
    return {"models_dir": str(p)}

@app.get("/api/models/tree")
def models_tree():
    root = Path(cfg["models_dir"])
    return {"tree": build_tree(root), "models_dir": str(root)}

@app.get("/api/models")
def list_models_flat():
    root = Path(cfg["models_dir"])
    models = []
    if root.exists():
        for ext in ("*.pth","*.safetensors","*.pt","*.onnx"):
            for p in root.rglob(ext):
                if not any(part.startswith('.') for part in p.parts):
                    fname = p.name
                    # Scale priority: persisted model_meta → in-memory cache → filename heuristic
                    cached_scale = (model_meta.get(fname, {}).get("scale")
                                    or getattr(loaded_models.get(str(p)), "scale", 0)
                                    or _guess_scale_from_name(p.name)
                                    or 0)
                    models.append({
                        "name": fname, "path": str(p),
                        "rel": str(p.relative_to(root)),
                        "size_mb": round(p.stat().st_size/1024/1024, 1),
                        "type": "onnx" if p.suffix==".onnx" else "pytorch",
                        "cached": str(p) in loaded_models,
                        "scale": cached_scale,
                    })
    return {"models": models}

@app.get("/api/model-scale")
def get_model_scale(path: str):
    """Return the scale factor of a model (lightweight — no GPU needed).
    Result is persisted to model_meta.json so future lookups are instant."""
    try:
        p = Path(path)
        fname = p.name
        # 1. Already in persistent cache
        if fname in model_meta and model_meta[fname].get("scale"):
            return {"scale": int(model_meta[fname]["scale"])}
        # 2. Already loaded in-memory this session
        if path in loaded_models:
            sc = int(getattr(loaded_models[path], "scale", 0) or 0)
            if sc:
                model_meta[fname] = {**model_meta.get(fname, {}), "scale": sc}
                save_model_meta(model_meta)
            return {"scale": sc}
        if not p.exists():
            return {"scale": 0, "error": "File not found"}
        # 3. ONNX: filename first (instant), tiny probe inference on CPU only if that fails
        if p.suffix.lower() == ".onnx":
            sc = _guess_scale_from_name(fname)
            if sc:
                log(f"[model-scale] {fname}: scale {sc} from filename")
                return {"scale": sc}         # not persisted — the first real run verifies and saves it
            sc = _probe_onnx_scale(p)
            if sc:
                model_meta[fname] = {**model_meta.get(fname, {}), "scale": sc}
                save_model_meta(model_meta)
                log(f"[model-meta] persisted ONNX scale={sc} for {fname}")
            return {"scale": sc}
        # 4. Load via spandrel on CPU to inspect architecture
        if HAS_SPANDREL:
            import spandrel as _sp
            desc = _sp.ModelLoader(device="cpu").load_from_file(str(p))
            sc = int(getattr(desc, "scale", 0) or 0)
            loaded_models[path] = desc          # cache in-memory
            model_meta[fname] = {**model_meta.get(fname, {}), "scale": sc}
            save_model_meta(model_meta)
            log(f"[model-meta] persisted scale={sc} for {fname}")
            return {"scale": sc}
        return {"scale": 0}
    except Exception as e:
        log(f"[model-scale] {e}")
        return {"scale": 0, "error": str(e)}

def preload_models(model_paths: str = Form(...)):
    paths = [Path(p.strip()) for p in model_paths.split("||")
             if p.strip() and Path(p.strip()).exists()]
    not_cached = [p for p in paths if str(p) not in loaded_models]
    if not_cached:
        processing_status.update({
            "running": True, "progress": 1,
            "message": f"Pre-loading {len(not_cached)} model(s) on {DEVICE.upper()}…",
            "step": "preload", "tile_current": 0, "tile_total": 0,
        })
        preload_models_bg(not_cached)
    return {"queued": len(not_cached), "cached": len(paths) - len(not_cached)}

def _make_preview(img: Image.Image, max_px: int = 2048) -> tuple[str, int, int]:
    """
    Fast downsampled preview for the canvas.
    Uses BOX resampling (fastest) and low-quality WEBP (method=0).
    """
    full_w, full_h = img.size
    if max(full_w, full_h) > max_px:
        img = img.copy()
        img.thumbnail((max_px, max_px), Image.BOX)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="WEBP", quality=72, method=0)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode(), full_w, full_h


def _open_fast(path) -> Image.Image:
    """
    Open an image as fast as possible.
    For JPEG: use draft() to decode at 1/8 resolution directly in libjpeg.
    For TIFF with multiple pages: only read first page.
    For everything else: normal open + load.
    """
    img = Image.open(path)
    fmt = (img.format or "").upper()

    if fmt == "JPEG":
        # libjpeg can decode at 1/2, 1/4, or 1/8 natively — much faster
        img.draft("RGB", (2048, 2048))
        img.load()
    elif fmt in ("TIFF", "TIF"):
        # Seek to first page only, don't load all frames
        try:
            img.seek(0)
        except EOFError:
            pass
        img.load()
    else:
        img.load()

    return img


# Formats the browser/WebView2 can display natively
BROWSER_NATIVE = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif"}

@app.get("/api/image-source")
def get_image_source():
    """Serve the current source image for canvas display.
    Native formats (JPEG/PNG/WEBP) are served directly.
    Non-native formats (TIFF, BMP…) are converted to JPEG on the fly.
    The original file is never modified — conversion is only for display.
    """
    if img_state.path is None or not img_state.path.exists():
        raise HTTPException(404, "No image loaded")

    suffix = img_state.path.suffix.lower()

    if suffix in BROWSER_NATIVE:
        # Serve directly — zero re-encoding cost
        media = {".jpg":"image/jpeg",".jpeg":"image/jpeg",
                 ".png":"image/png",".webp":"image/webp",
                 ".gif":"image/gif",".avif":"image/avif"}[suffix]
        return FileResponse(img_state.path, media_type=media,
                            headers={"Cache-Control":"no-store"})

    # Non-native (TIFF, BMP, etc.) — convert to PNG (lossless) for display only
    # Preserve alpha channel if present
    t0 = time.perf_counter()
    img = _open_fast(img_state.path)
    if _has_alpha(img):
        display = img.convert("RGBA")
    else:
        display = to_rgb(img)
    img.close()
    buf = io.BytesIO()
    display.save(buf, format="PNG", compress_level=1)
    buf.seek(0)
    log(f"[image-source] converted {img_state.path.suffix} → PNG {display.mode} {display.width}×{display.height}  {time.perf_counter()-t0:.2f}s")
    return Response(content=buf.getvalue(), media_type="image/png",
                             headers={"Cache-Control":"no-store"})


@app.get("/api/image-source-raw")
def get_image_source_raw():
    """Serve the current source image as raw straight-alpha RGBA bytes.
    Browsers decode images premultiplied, which destroys RGB under alpha=0;
    the "Raw RGB" view needs the real values. Headers carry the dimensions."""
    if img_state.path is None or not img_state.path.exists():
        raise HTTPException(404, "No image loaded")
    img = _open_fast(img_state.path)
    try:
        if not _has_alpha(img):
            raise HTTPException(404, "Image has no alpha channel")
        rgba = img.convert("RGBA")
    finally:
        img.close()
    data = rgba.tobytes()
    return Response(content=data, media_type="application/octet-stream",
                    headers={"Cache-Control": "no-store",
                             "X-Width": str(rgba.width), "X-Height": str(rgba.height),
                             "Access-Control-Expose-Headers": "X-Width, X-Height"})


@app.post("/api/load-image-path")
async def load_image_by_path(path: str = Form(...)):
    """Desktop mode: load image directly from a local file path (no upload needed)."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(400, f"File not found: {path}")

    t0 = time.perf_counter()
    img_state.path        = p
    img_state.is_clipboard = False
    img_state.source_path  = p

    # Just read dimensions — no pixel decode, no re-encode
    with Image.open(p) as img:
        full_w, full_h = img.size
        has_alpha = _has_alpha(img)
        log(f"[load-image] mode={img.mode}  size={full_w}×{full_h}  format={img.format}  alpha={has_alpha}  {p.name}")

    log(f"[load-image] path  {p.name}  {full_w}×{full_h}  {time.perf_counter()-t0:.3f}s")
    return {
        "filename": p.name,
        "width": full_w, "height": full_h,
        "source_url": "/api/image-source",
        "is_clipboard": False,
        "source_path": str(p),
        "has_alpha": has_alpha,
    }


@app.post("/api/load-image")
async def load_image(
    file: UploadFile = File(...),
    source_path: str = Form(""),
    is_clipboard: bool = Form(False),
):
    t0 = time.perf_counter()
    data = await file.read()
    dest = UPLOADS_DIR / file.filename
    # Remove previous upload if it exists and differs (keep disk clean)
    if img_state.path and img_state.path != dest:
        try:
            if str(img_state.path).startswith(str(UPLOADS_DIR)):
                img_state.path.unlink(missing_ok=True)
        except Exception:
            pass
    dest.write_bytes(data)
    img_state.path = dest
    img_state.is_clipboard = is_clipboard
    img_state.source_path = Path(source_path) if source_path else None

    with Image.open(dest) as img:
        full_w, full_h = img.size
        has_alpha = _has_alpha(img)
        log(f"[load-image] mode={img.mode}  size={full_w}×{full_h}  format={img.format}  alpha={has_alpha}  {file.filename}")

    log(f"[load-image] upload  {file.filename}  {full_w}×{full_h}  {time.perf_counter()-t0:.3f}s")
    return {
        "filename": file.filename,
        "width": full_w, "height": full_h,
        "source_url": "/api/image-source",
        "is_clipboard": is_clipboard,
        "source_path": source_path,
        "has_alpha": has_alpha,
    }

def _predict_sizes(pipeline, src_w, src_h, pre_scale, post_scale):
    """
    Simulate both tracks through the pipeline and return predicted final sizes.
    Returns ((rgb_w, rgb_h), (alpha_w, alpha_h), predictable).
    predictable=False means at least one step has an unknown scale factor —
    we can't determine final sizes with certainty.
    """
    def rs(w, h):
        return max(1, round(w)), max(1, round(h))

    rw, rh = rs(src_w * pre_scale, src_h * pre_scale)
    aw, ah = rw, rh

    for p in pipeline:
        ptype = p.get("type", "model")

        if ptype == "overlay" and not p.get("model"):
            continue                                  # no model assigned → step is skipped
        if ptype in ("model", "interpolate", "overlay"):
            sf = float(p.get("scale", 0) or p.get("model_a_scale", 0) or 0)
            if sf <= 0:
                return (rw, rh), (aw, ah), False  # scale unknown
            # Both RGB and Alpha tracks end up at the same size:
            # - channels='all'/'rgb'/'alpha': the *other* channel is also resized by sf
            rw, rh = rs(rw * sf, rh * sf)
            aw, ah = rs(aw * sf, ah * sf)

        elif ptype == "discrete":
            rsf = float(p.get("model_rgb_scale", 0) or 0)
            asf = float(p.get("model_alpha_scale", 0) or 0)
            if p.get("model_rgb"):
                if rsf <= 0:
                    return (rw, rh), (aw, ah), False
                rw, rh = rs(rw * rsf, rh * rsf)
            if p.get("model_alpha"):
                if asf <= 0:
                    return (rw, rh), (aw, ah), False
                aw, ah = rs(aw * asf, ah * asf)

        elif ptype == "resize":
            ch   = p.get("channels", "all")
            mode = p.get("mode", "percent")

            def _rsz(w, h):
                if mode == "percent":
                    pct = float(p.get("value", 100)) / 100.0
                    return rs(w * pct, h * pct)
                elif mode == "pixels":
                    return int(p.get("width", w)), int(p.get("height", h))
                elif mode == "longest_side":
                    lng = float(p.get("value", max(w, h)))
                    sc  = lng / max(w, h)
                    return rs(w * sc, h * sc)
                elif mode == "shortest_side":
                    sht = float(p.get("value", min(w, h)))
                    sc  = sht / min(w, h)
                    return rs(w * sc, h * sc)
                return w, h

            if ch != "alpha": rw, rh = _rsz(rw, rh)
            if ch != "rgb":   aw, ah = _rsz(aw, ah)

        elif ptype == "crop":
            mode = p.get("mode", "percent")
            if mode == "percent":
                lr = float(p.get("left", 0)) + float(p.get("right",  0))
                tb = float(p.get("top",  0)) + float(p.get("bottom", 0))
                rw = max(1, round(rw * (1 - lr / 100)))
                rh = max(1, round(rh * (1 - tb / 100)))
                aw = max(1, round(aw * (1 - lr / 100)))
                ah = max(1, round(ah * (1 - tb / 100)))
            else:
                lr = int(p.get("left", 0)) + int(p.get("right",  0))
                tb = int(p.get("top",  0)) + int(p.get("bottom", 0))
                rw = max(1, rw - lr); rh = max(1, rh - tb)
                aw = max(1, aw - lr); ah = max(1, ah - tb)
        # flip, alpha_threshold, hsl, levels, brightness_contrast, etc.: no size change

    rw, rh = rs(rw * post_scale, rh * post_scale)
    aw, ah = rs(aw * post_scale, ah * post_scale)
    return (rw, rh), (aw, ah), True


@app.post("/api/process")
def process_image(
    model_paths:   str   = Form(""),     # legacy: "||"-separated paths (kept for compat)
    pipeline_json: str   = Form(""),     # new: JSON array of pipeline steps
    pre_scale:     float = Form(1.0),
    post_scale:    float = Form(1.0),
    tile_size:     int   = Form(512),
    output_format: str   = Form("PNG"),
    mode:          str   = Form("preview_full"),
    upscale_alpha: bool  = Form(True),
    region_x:      int   = Form(0),
    region_y:      int   = Form(0),
    region_w:      int   = Form(0),
    region_h:      int   = Form(0),
    out_dir:       str   = Form(""),     # custom output directory for batch
):
    global processing_status, alpha_result_cache, result_raw_cache
    alpha_result_cache = None
    result_raw_cache = None
    cancel_event.clear()
    if img_state.path is None or not img_state.path.exists():
        raise HTTPException(400, "No image loaded")

    # Build pipeline: prefer pipeline_json, fall back to legacy model_paths
    if pipeline_json:
        pipeline = json.loads(pipeline_json)
    else:
        pipeline = [{"type":"model","path":p.strip()}
                    for p in model_paths.split("||") if p.strip()]

    if not pipeline:
        raise HTTPException(400, "No pipeline steps selected")
    # Validate model paths exist
    for step in pipeline:
        if step.get("type") == "model":
            p = Path(step["path"])
            if not p.exists():
                raise HTTPException(404, f"Model not found: {p}")

    # For log and preload compatibility
    model_names = [Path(s["path"]).name for s in pipeline if s.get("type")=="model"]

    processing_status.update({"running": True, "progress": 2, "message": "Loading image…",
                         "step": "load_image", "tile_current": 0, "tile_total": 0,
                         "pipe_step": 0, "pipe_total": 0})
    log(f"[process] START  mode={mode}  pipeline={[s.get('type','?')+'('+Path(s['path']).name+')' if s.get('type')=='model' else s.get('type','?') for s in pipeline]}")
    t_req = time.perf_counter()
    try:
        t0 = time.perf_counter()
        src = Image.open(img_state.path)
        has_alpha = _has_alpha(src)
        log(f"[process] image open+convert  {time.perf_counter()-t0:.3f}s  size={src.width}×{src.height}  mode was: {src.mode}  has_alpha={has_alpha}")

        # Pre-flight note: the client performs a size mismatch check before submitting.
        # If we somehow reach here with a mismatch (e.g. unknown scale at prediction time),
        # the reconciliation code below will handle it.
        if has_alpha and upscale_alpha and len(pipeline) > 0:
            rgb_sz, alpha_sz, predictable = _predict_sizes(
                pipeline, src.width, src.height, pre_scale, post_scale)
            if predictable and rgb_sz != alpha_sz:
                log(f"[process] WARNING: predicted size mismatch RGB={rgb_sz} α={alpha_sz} "
                    f"(client check may have missed this — reconciliation will handle it)")

        alpha_channel = None
        src_alpha = None   # original alpha for upscale_alpha=False pipeline path
        if has_alpha:
            src_rgba = src.convert("RGBA")
            _a = src_rgba.getchannel("A")
            if upscale_alpha:
                alpha_channel = _a   # will be upscaled separately
            else:
                src_alpha = _a       # will be passed into pipeline with the image

        img = to_rgb(src)
        src.close()

        if mode == "preview_cutout" and region_w > 0 and region_h > 0:
            proc_img = img.crop((region_x, region_y, region_x+region_w, region_y+region_h))
            if alpha_channel:
                alpha_channel = alpha_channel.crop(
                    (region_x, region_y, region_x+region_w, region_y+region_h))
            if src_alpha:
                src_alpha = src_alpha.crop(
                    (region_x, region_y, region_x+region_w, region_y+region_h))
        else:
            proc_img = img

        # If image has alpha and we're not upscaling it separately, pass RGBA into pipeline
        # so actions can preserve/operate on the alpha channel.
        # Use the real original alpha (src_alpha) — NOT proc_img.convert("RGBA") which
        # would produce an all-white (255) alpha because proc_img is already RGB.
        if has_alpha and alpha_channel is None:
            if src_alpha is not None:
                src_for_pipe = Image.merge("RGBA", (*proc_img.split(), src_alpha))
            else:
                src_for_pipe = proc_img.convert("RGBA")  # fallback (should not happen)
        else:
            src_for_pipe = proc_img  # RGB (alpha stripped, handled separately)

        # For color_extend steps in the RGB-only track: inject the current alpha
        # as a proxy so the action can dilate RGB guided by the alpha mask.
        # We use src_alpha (original alpha before any pipeline processing) as a
        # best approximation — it matches the source geometry at this point.
        pipeline_for_run = pipeline
        # For color_extend: inject alpha proxy regardless of whether alpha is
        # upscaled separately (alpha_channel) or passed inline (src_alpha).
        _alpha_for_ce = alpha_channel if alpha_channel is not None else src_alpha
        if _alpha_for_ce is not None:
            has_ce = any(p.get("type") == "color_extend" for p in pipeline)
            if has_ce:
                pipeline_for_run = [
                    dict(p, _alpha_proxy=_alpha_for_ce) if p.get("type") == "color_extend" else p
                    for p in pipeline
                ]

        try:
            result = process_pipeline(src_for_pipe, pipeline_for_run, pre_scale, post_scale, tile_size)
        except MemoryError as e:
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()
            log(f"[process] OUT OF MEMORY: {e}")
            raise HTTPException(500,
                "Out of memory. Try: reduce tile size, use a smaller image, "
                "or close other applications to free RAM/VRAM.")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if HAS_TORCH and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                log(f"[process] CUDA OOM: {e}")
                raise HTTPException(500,
                    f"CUDA out of memory. Reduce tile size and try again. ({e})")
            raise

        # Alpha track: apply the full pipeline to the alpha channel,
        # but skip colour-only actions (HSL, levels, etc.) that should not
        # affect transparency.  Geometric actions (resize/flip/crop) and
        # alpha_threshold are applied directly on the mode-L alpha image;
        # model / interpolate steps run through process_pipeline on a
        # greyscale-as-RGB proxy exactly as before.
        upscaled_alpha = None
        if alpha_channel and upscale_alpha:
            cur_alpha = alpha_channel.copy()   # mode L
            alpha_recon_filt = Image.LANCZOS   # updated by discrete steps if present
            # Pre-scale (mirrors the pre-scale in the RGB pipeline)
            if pre_scale != 1.0:
                aw, ah = cur_alpha.size
                cur_alpha = cur_alpha.resize(
                    (max(1, int(aw * pre_scale)), max(1, int(ah * pre_scale))),
                    Image.LANCZOS)
            has_model_step = any(p.get("type","model") in ("model","interpolate")
                                 for p in pipeline)
            if has_model_step:
                set_status(72, "Upscaling alpha channel…", step="alpha")
            for step in pipeline:
                check_cancel()
                ptype = step.get("type", "model")
                if ptype in ("model", "interpolate", "overlay"):
                    ch      = step.get("channels", "all")
                    of_key  = step.get("other_filter", "lanczos")
                    _of_map = {"lanczos":Image.LANCZOS,"bicubic":Image.BICUBIC,
                               "bilinear":Image.BILINEAR,"nearest":Image.NEAREST}
                    of_filt = _of_map.get(of_key, Image.LANCZOS)
                    scale_f = float(step.get("scale", 0) or 0)
                    if ch == "rgb":
                        # Model runs on RGB; alpha track just resizes to match
                        if scale_f > 0:
                            aw, ah = cur_alpha.size
                            cur_alpha = cur_alpha.resize(
                                (max(1,round(aw*scale_f)), max(1,round(ah*scale_f))), of_filt)
                        # else: scale unknown, keep alpha size
                    else:
                        # ch == "all" or "alpha": run model on alpha track.
                        # Override channels to "all" so process_pipeline actually runs
                        # the model (not the resize-bypass intended for the RGB track).
                        alpha_step = dict(step, channels="all", _alpha_track=True)
                        alpha_rgb = Image.merge("RGB", [cur_alpha] * 3)
                        alpha_out = process_pipeline(alpha_rgb, [alpha_step],
                                                     1.0, 1.0, tile_size)
                        cur_alpha = alpha_out.getchannel("R")
                elif ptype == "discrete":
                    # Discrete: run model_alpha on the alpha channel
                    _of_map2 = {"lanczos":Image.LANCZOS,"bicubic":Image.BICUBIC,
                                "bilinear":Image.BILINEAR,"nearest":Image.NEAREST}
                    af_key = step.get("alpha_resize", "lanczos")
                    # None means: keep alpha at its own size even if it differs from RGB
                    alpha_recon_filt = None if af_key == "none" else _of_map2.get(af_key, Image.LANCZOS)
                    path_alpha = step.get("model_alpha", "")
                    if path_alpha:
                        log(f"[process] discrete alpha model: {Path(path_alpha).name}")
                        alpha_rgb = Image.merge("RGB", [cur_alpha] * 3)
                        alpha_out = process_pipeline(alpha_rgb,
                                                     [{"type":"model","path":path_alpha}],
                                                     1.0, 1.0, tile_size)
                        cur_alpha = alpha_out.getchannel("R")
                        log(f"[process] discrete alpha done  {cur_alpha.size}")
                elif ptype == "resize":
                    # channels='rgb': only RGB gets resized; alpha track skips
                    if step.get("channels", "all") != "rgb":
                        mode_str = step.get("mode", "percent")
                        filt_map = {"lanczos": Image.LANCZOS, "bicubic": Image.BICUBIC,
                                    "nearest": Image.NEAREST, "bilinear": Image.BILINEAR}
                        filt = filt_map.get(step.get("filter", "lanczos"), Image.LANCZOS)
                        aw, ah = cur_alpha.size
                        if mode_str == "percent":
                            pct = float(step.get("value", 100)) / 100.0
                            nw, nh = max(1, int(aw*pct)), max(1, int(ah*pct))
                        elif mode_str == "pixels":
                            nw, nh = int(step.get("width", aw)), int(step.get("height", ah))
                        elif mode_str == "longest_side":
                            lng = int(step.get("value", max(aw, ah)))
                            sc  = lng / max(aw, ah)
                            nw, nh = max(1, int(aw*sc)), max(1, int(ah*sc))
                        elif mode_str == "shortest_side":
                            sht = int(step.get("value", min(aw, ah)))
                            sc  = sht / min(aw, ah)
                            nw, nh = max(1, int(aw*sc)), max(1, int(ah*sc))
                        else:
                            nw, nh = aw, ah
                        cur_alpha = cur_alpha.resize((nw, nh), filt)
                elif ptype == "flip":
                    if step.get("flip_x"):
                        cur_alpha = cur_alpha.transpose(Image.FLIP_LEFT_RIGHT)
                    if step.get("flip_y"):
                        cur_alpha = cur_alpha.transpose(Image.FLIP_TOP_BOTTOM)
                elif ptype == "crop":
                    mode_str = step.get("mode", "percent")
                    aw, ah = cur_alpha.size
                    if mode_str == "percent":
                        cl = int(aw * float(step.get("left",   0)) / 100)
                        ct = int(ah * float(step.get("top",    0)) / 100)
                        cr = int(aw * float(step.get("right",  0)) / 100)
                        cb = int(ah * float(step.get("bottom", 0)) / 100)
                    else:
                        cl = int(step.get("left",   0)); ct = int(step.get("top",    0))
                        cr = int(step.get("right",  0)); cb = int(step.get("bottom", 0))
                    cur_alpha = cur_alpha.crop(
                        (cl, ct, max(cl+1, aw-cr), max(ct+1, ah-cb)))
                elif ptype == "alpha_threshold":
                    thresh = int(step.get("threshold", 128))
                    arr = np.array(cur_alpha, dtype=np.uint8)
                    cur_alpha = Image.fromarray(
                        np.where(arr < thresh, np.uint8(0), np.uint8(255)),
                        mode="L")
                elif ptype == "brightness_contrast" and step.get("channel") == "alpha":
                    # brightness_contrast targeting the alpha channel is applied here
                    brightness = float(step.get("brightness", 0)) / 100.0
                    contrast   = float(step.get("contrast",   0)) / 100.0
                    arr = np.array(cur_alpha, dtype=np.float32) / 255.0
                    arr = arr + brightness
                    arr = (arr - 0.5) * (1.0 + contrast) + 0.5
                    cur_alpha = Image.fromarray(
                        (np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L")
                # colour-only actions (hsl, levels, brightness_contrast on RGB,
                # color_temperature, invert, grayscale) and color_extend
                # (operates on RGB guided by alpha; alpha itself is unchanged) are skipped
            # Post-scale
            if post_scale != 1.0:
                aw, ah = cur_alpha.size
                cur_alpha = cur_alpha.resize(
                    (max(1, int(aw * post_scale)), max(1, int(ah * post_scale))),
                    Image.LANCZOS)
            upscaled_alpha = cur_alpha
            abuf = io.BytesIO()
            upscaled_alpha.save(abuf, format="PNG", compress_level=1)
            alpha_result_cache = abuf.getvalue()
            log(f"[process] alpha track done  {upscaled_alpha.width}×{upscaled_alpha.height}")

        # Size reconciliation: safety net for when scale was unpredictable (scale_f=0).
        # If the pre-flight check ran and passed, sizes already match and this is a no-op.
        if upscaled_alpha is not None and result.size != upscaled_alpha.size:
            rp = result.width * result.height
            ap = upscaled_alpha.width * upscaled_alpha.height
            log(f"[process] size mismatch RGB={result.size} α={upscaled_alpha.size} — reconciling (unpredictable scale)")
            if alpha_recon_filt is None:
                log("[process] alpha_recon_filt=None, keeping sizes as-is (alpha_resize=none)")
            elif rp >= ap:
                upscaled_alpha = upscaled_alpha.resize(result.size, alpha_recon_filt)
            else:
                result = result.convert("RGB").resize(upscaled_alpha.size, Image.LANCZOS)
                log(f"[process] resized RGB result to {upscaled_alpha.size}")

        check_cancel()
        set_status(80, "Encoding result…", step="encode")
        t0 = time.perf_counter()

        # For save modes: compose RGBA if we have an upscaled alpha
        if upscaled_alpha and mode not in ("preview_full", "preview_cutout"):
            if result.size == upscaled_alpha.size:
                result_save = Image.merge("RGBA", (*result.split(), upscaled_alpha))
            else:
                log(f"[process] merge skipped: size mismatch {result.size} vs {upscaled_alpha.size}")
                result_save = result
        else:
            result_save = result

        # has_alpha_out: True if result should be displayed with alpha channel
        has_alpha_out = (upscaled_alpha is not None) or (has_alpha and not upscale_alpha)

        if mode in ("preview_full", "preview_cutout"):
            # Raw push: keep the result as straight RGBA bytes; the frontend
            # fetches them with one binary request (/api/result-raw). Much faster
            # than base64 tiles through evaluate_js, and works without pywebview.
            try:
                push_img = result
                if upscaled_alpha is not None:
                    if result.size == upscaled_alpha.size:
                        r, g, b = result.convert("RGB").split()
                        push_img = Image.merge("RGBA", (r, g, b, upscaled_alpha))
                    else:
                        log(f"[process] merge skipped: size mismatch (unpredictable scale)")
                        push_img = result.convert("RGB")
                pw, ph = push_img.size
                cx = region_x; cy = region_y
                cw = region_w if region_w > 0 else img.width
                ch = region_h if region_h > 0 else img.height
                t1 = time.perf_counter()
                result_raw_cache = (push_img.convert("RGBA").tobytes(), pw, ph)
                log(f"[process] raw result {pw}×{ph}  {len(result_raw_cache[0])/1e6:.1f} MB  "
                    f"{time.perf_counter()-t1:.3f}s")
                log(f"[process] TOTAL request  {time.perf_counter()-t_req:.3f}s")
                processing_status.update({"running": False, "progress": 100, "message": "Done",
                                      "step": "done", "tile_current": 0, "tile_total": 0})
                return {"raw_push": True,
                        "width": result.width, "height": result.height,
                        "push_w": pw, "push_h": ph,
                        "crop_x": cx, "crop_y": cy, "crop_w": cw, "crop_h": ch,
                        "has_alpha": has_alpha_out}
            except Exception as e:
                result_raw_cache = None
                log(f"[process] raw push failed: {e} — falling back to PNG")
            # Fallback: encode PNG and stream
            buf = io.BytesIO()
            result.save(buf, format="PNG", compress_level=1)
            buf.seek(0)
            log(f"[process] encode PNG  {time.perf_counter()-t0:.3f}s  result={result.width}×{result.height}")
            log(f"[process] TOTAL request  {time.perf_counter()-t_req:.3f}s")
            processing_status.update({"running": False, "progress": 100, "message": "Done",
                                  "step": "done", "tile_current": 0, "tile_total": 0})
            return Response(content=buf.getvalue(), media_type="image/png", headers={
                "X-Result-Width":  str(result.width),
                "X-Result-Height": str(result.height),
                "X-Crop-X": str(region_x),
                "X-Crop-Y": str(region_y),
                "X-Crop-W": str(region_w if region_w>0 else img.width),
                "X-Crop-H": str(region_h if region_h>0 else img.height),
                "X-Has-Alpha":     "1" if has_alpha_out else "0",
            })
        else:
            check_cancel()
            set_status(88, "Saving…", step="save")
            def pipeline_suffix(pipeline):
                from datetime import datetime
                model_steps = [s for s in pipeline if s.get("type") == "model"]
                non_model   = [s for s in pipeline if s.get("type") not in ("model",)]
                # Single model, no actions → Quick Mode: use model name only
                if len(model_steps) == 1 and not non_model:
                    return Path(model_steps[0]["path"]).stem
                # Pipeline: use timecode
                ts = datetime.now().strftime("%Y%m%d%H%M%S")
                return f"pipeline_{ts}"

            ms = pipeline_suffix(pipeline)
            if out_dir and out_dir.strip():
                # Custom output directory (batch mode)
                dest = Path(out_dir.strip())
                dest.mkdir(parents=True, exist_ok=True)
                filename = f"{img_state.path.stem}__{ms}.{output_format.lower()}"
                out_path = dest / filename
            elif mode == "process_save" and not img_state.is_clipboard and img_state.source_path:
                filename = f"{img_state.source_path.stem}__{ms}.{output_format.lower()}"
                out_path = img_state.source_path.parent / filename
            else:
                filename = f"{img_state.path.stem}__{ms}.{output_format.lower()}"
                out_path = OUTPUT_DIR / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fmt = "JPEG" if output_format.upper()=="JPG" else output_format.upper()
            save_img = result_save if fmt in ("PNG","TIFF","WEBP") else result
            set_status(90, "Saving file…", step="save")
            save_img.save(out_path, format=fmt, **({"quality":95} if fmt=="JPEG" else {}))
            processing_status.update({"running": False, "progress": 100, "message": "Saved",
                                  "step": "done", "tile_current": 0, "tile_total": 0})
            return {"filename": filename, "saved_to": str(out_path),
                    "width": result.width, "height": result.height,
                    "has_alpha": has_alpha_out}

    except CancelledError:
        processing_status.update({"running": False, "progress": 0, "message": "Cancelled",
                             "step": "cancelled", "tile_current": 0, "tile_total": 0})
        raise HTTPException(499, "Cancelled by user")
    except Exception as e:
        processing_status.update({"running": False, "progress": 0, "message": f"Error: {e}",
                              "step": "error", "tile_current": 0, "tile_total": 0})
        raise HTTPException(500, str(e))
    finally:
        release_vram_cache()

@app.get("/api/result-raw")
def get_result_raw():
    """Last preview result as straight RGBA bytes (see raw push in process_image)."""
    if result_raw_cache is None:
        raise HTTPException(404, "No raw result")
    data, w, h = result_raw_cache
    return Response(content=data, media_type="application/octet-stream",
                    headers={"Cache-Control": "no-store",
                             "X-Width": str(w), "X-Height": str(h)})

@app.post("/api/clear-result")
def clear_result():
    """Drop the cached preview result (raw RGBA + alpha PNG) and release cached VRAM."""
    global result_raw_cache, alpha_result_cache
    freed = (len(result_raw_cache[0]) if result_raw_cache else 0) + len(alpha_result_cache or b"")
    result_raw_cache = None
    alpha_result_cache = None
    release_vram_cache()
    log(f"[clear-result] freed {freed/1e6:.1f} MB result cache")
    return {"freed_mb": round(freed / 1e6, 1)}

@app.post("/api/cancel")
async def cancel_processing():
    """Stop the running job: between tiles/steps, or immediately for ONNX runs."""
    request_cancel()
    processing_status.update({"message": "Cancelling…"})
    return {"ok": True}

@app.get("/api/alpha-result")
def get_alpha_result():
    """Return the last upscaled alpha channel as greyscale PNG."""
    if alpha_result_cache is None:
        raise HTTPException(404, "No alpha result cached")
    return Response(content=alpha_result_cache, media_type="image/png",
                    headers={"Cache-Control": "no-store"})

@app.get("/api/debug-log")
def get_debug_log():
    return {"lines": list(_server_log)}

@app.post("/api/vram/clear")
def clear_vram():
    """Unload all cached models and clear CUDA VRAM."""
    global loaded_models
    loaded_models.clear()
    if HAS_TORCH:
        try:
            import torch as _t
            if _t.cuda.is_available():
                _t.cuda.empty_cache()
                _t.cuda.ipc_collect()
        except Exception:
            pass
    log("[vram] cleared by user")
    return {"cleared": True}

@app.get("/api/vram")
def get_vram():
    try:
        import torch as _torch
        if not _torch.cuda.is_available():
            return {"available_gb": None, "total_gb": None, "auto_tile": 512}
        free, total = _torch.cuda.mem_get_info(0)
        free_gb = free / 1024**3
        if free_gb >= 6:   auto_tile = 1024
        elif free_gb >= 4: auto_tile = 768
        elif free_gb >= 2: auto_tile = 512
        elif free_gb >= 1: auto_tile = 384
        else:              auto_tile = 256
        auto_tile = round(auto_tile / 128) * 128
        return {
            "available_gb": round(free_gb, 1),
            "total_gb":     round(total / 1024**3, 1),
            "auto_tile":    auto_tile,
        }
    except Exception:
        return {"available_gb": None, "total_gb": None, "auto_tile": 512}

@app.get("/api/status")
async def get_status():
    return processing_status

@app.get("/api/download/{filename}")
def download_file(filename: str):
    p = OUTPUT_DIR / filename
    if not p.exists(): raise HTTPException(404, "Not found")
    return FileResponse(p, filename=filename)

@app.get("/api/browse")
def browse_dir(path: str = ""):
    if not path:
        import platform
        if platform.system() == "Windows":
            import string
            roots = [f"{d}:\\" for d in string.ascii_uppercase if Path(f"{d}:\\").exists()]
        else:
            roots = ["/"]
        return {"path":"","parent":"","dirs":[{"name":r,"path":r} for r in roots]}
    p = Path(path).expanduser().resolve()
    if not p.exists() or not p.is_dir(): raise HTTPException(400,"Invalid path")
    try:
        dirs = sorted(
            [{"name":e.name,"path":str(e)} for e in p.iterdir()
             if e.is_dir() and not e.name.startswith(".")],
            key=lambda d: d["name"].lower())
    except PermissionError:
        dirs = []
    return {"path":str(p),"parent":str(p.parent) if p.parent!=p else "","name":p.name,"dirs":dirs}

import os
import subprocess, sys, importlib.metadata

# Only add --break-system-packages if NOT in a venv (fallback for system Python)
_IN_VENV = hasattr(sys, 'real_prefix') or (
    hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
_PIP_FLAGS = [] if _IN_VENV else ["--break-system-packages"]

# ── Dependency info ────────────────────────────────────────────────────────────
DEPS = [
    ("fastapi",            "fastapi",    ">=0.100", "Web API framework",                         "~0.1 MB"),
    ("uvicorn",            "uvicorn",    ">=0.20",  "ASGI server",                               "~0.5 MB"),
    ("python-multipart",   "multipart",  ">=0.0.5", "File upload support",                       "~0.1 MB"),
    ("Pillow",             "PIL",        ">=9.0",   "Image processing",                          "~4 MB"),
    ("numpy",              "numpy",      ">=1.23",  "Array operations",                          "~18 MB"),
    ("torch",              "torch",      ">=2.0",   "PyTorch — install this first (CPU / CUDA / MPS)", "~800 MB CPU / ~2.5 GB CUDA"),
    ("spandrel",           "spandrel",   ">=0.3",   "42+ model architectures (requires torch)",  "~0.5 MB"),
    ("onnxruntime",        "onnxruntime",">=1.15",  "ONNX inference (CPU)",                      "~11 MB"),
    ("onnxruntime-gpu",    "onnxruntime",">=1.15",  "ONNX inference (CUDA GPU)",                 "~650 MB"),
    ("onnxruntime-directml","onnxruntime",">=1.15", "ONNX inference (DirectML — Windows AMD/Intel)", "~350 MB"),
    ("onnxruntime-silicon","onnxruntime",">=1.15",  "ONNX inference (CoreML — Apple Silicon)",   "~25 MB"),
    ("pywebview",          "webview",    ">=4.0",   "Desktop window",                            "~2 MB"),
]

def check_dep(import_name: str):
    """Check if a package is installed. Uses metadata first (reliable after pip install)."""
    # Map import name → pip package name for metadata lookup
    pip_name_map = {
        "PIL": "Pillow", "multipart": "python-multipart",
        "webview": "pywebview", "spandrel": "spandrel",
        "fastapi": "fastapi", "uvicorn": "uvicorn",
        "numpy": "numpy", "torch": "torch", "onnxruntime": "onnxruntime",
    }
    pip_name = pip_name_map.get(import_name, import_name)
    # For onnxruntime, any installed variant counts (gpu/directml/silicon all provide it)
    candidates = (["onnxruntime-gpu", "onnxruntime-directml",
                   "onnxruntime-silicon", "onnxruntime"]
                  if import_name == "onnxruntime"
                  else [pip_name, pip_name.replace("-","_"), import_name])
    for pkg in candidates:
        try:
            ver = importlib.metadata.version(pkg)
            return {"installed": True, "version": ver}
        except importlib.metadata.PackageNotFoundError:
            continue
    # Fall back to actual import (clears stale cache entry first)
    sys.modules.pop(import_name, None)
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "?")
        return {"installed": True, "version": ver}
    except ImportError:
        return {"installed": False, "version": None}

def get_cuda_info():
    info = {"available": False, "version": None, "device_name": None, "vram_gb": None}
    try:
        import torch
        if torch.cuda.is_available():
            info["available"] = True
            info["device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_gb"] = round(props.total_memory / 1024**3, 1)
            info["version"] = torch.version.cuda
    except Exception:
        pass
    return info

def get_current_device():
    """Return the active inference device based on current torch state."""
    try:
        import torch
        if torch.cuda.is_available(): return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    except Exception:
        pass
    return DEVICE

install_log: list[str] = []
install_running: bool = False
# Structured progress for a real progress bar in the UI (separate from the ASCII log)
install_progress: dict = {
    "phase": "", "pct": 0, "label": "",
    "speed": "", "eta": "", "done_mb": 0.0, "total_mb": 0.0, "file": "",
}

import re as _re
_PROG_RE  = _re.compile(r'([\d.]+)\s*/\s*([\d.]+)\s*(MB|kB|KB|GB|MiB|GiB|B)\b', _re.I)
_SPEED_RE = _re.compile(r'([\d.]+\s*(?:[MkKG]i?B)/s)', _re.I)
_ETA_RE   = _re.compile(r'(?:eta|ETA)\s+([\d:]+)')
_DLFILE_RE= _re.compile(r'Downloading\s+(\S+)', _re.I)
_UNIT_MB  = {"b": 1e-6, "kb": 1e-3, "kib": 1e-3, "mb": 1.0, "mib": 1.0,
             "gb": 1000.0, "gib": 1000.0}

def _reset_install_progress():
    install_progress.update({"phase": "", "pct": 0, "label": "",
                             "speed": "", "eta": "", "done_mb": 0.0,
                             "total_mb": 0.0, "file": ""})

def _parse_progress_line(line: str):
    """Extract structured download/install progress from a pip output line.
    Updates install_progress in place. Returns True if it was a progress line."""
    low = line.lower()
    # Phase transitions
    if "installing collected packages" in low:
        install_progress.update({"phase": "installing", "pct": 0,
                                 "label": "Installing packages…",
                                 "speed": "", "eta": ""})
        return False
    if "building wheel" in low or "preparing metadata" in low or "building wheels" in low:
        install_progress.update({"phase": "building",
                                 "label": "Building wheels…", "pct": 0,
                                 "speed": "", "eta": ""})
        return False
    if low.startswith("collecting") or low.startswith("requirement already") \
       or "resolving dependencies" in low:
        install_progress.update({"phase": "resolving",
                                 "label": line[:80], "pct": 0})
        return False
    # Download start: "Downloading torch-...whl (670.2 MB)"
    mdl = _DLFILE_RE.search(line)
    pm  = _PROG_RE.search(line)
    if pm:
        done = float(pm.group(1)); total = float(pm.group(2))
        unit = pm.group(3).lower()
        f = _UNIT_MB.get(unit, 1.0)
        done_mb, total_mb = done * f, total * f
        pct = int(min(100, round(done / total * 100))) if total > 0 else 0
        sp = _SPEED_RE.search(line); eta = _ETA_RE.search(line)
        install_progress.update({
            "phase": "downloading", "pct": pct,
            "done_mb": round(done_mb, 1), "total_mb": round(total_mb, 1),
            "speed": sp.group(1).replace(" ", "") if sp else "",
            "eta": eta.group(1) if eta else "",
            "label": (install_progress.get("file") or "Downloading…"),
        })
        return True
    if mdl:
        install_progress.update({"phase": "downloading", "file": mdl.group(1),
                                 "label": mdl.group(1), "pct": 0,
                                 "speed": "", "eta": ""})
        return False
    return False

@app.get("/api/settings/deps")
def get_deps():
    result = []
    for pip_name, import_name, req, desc, dl_size in DEPS:
        status = check_dep(import_name)
        result.append({
            "pip_name": pip_name,
            "import_name": import_name,
            "description": desc,
            "required": req,
            "dl_size": dl_size,
            "installed": status["installed"],
            "version": status["version"],
        })
    return {
        "deps": result,
        "cuda": get_cuda_info(),
        "python": sys.version,
        "platform": sys.platform,
        "device": get_current_device(),
        "onnx_providers": sorted(_ORT_AVAILABLE_PROVIDERS),
        "active_onnx_providers": _onnx_providers() if HAS_ONNX else [],
        "is_mac": sys.platform == "darwin",
        "is_windows": sys.platform == "win32",
        "is_linux": sys.platform.startswith("linux"),
        "is_apple_silicon": sys.platform == "darwin" and __import__("platform").machine() == "arm64",
        "mps_available": bool(
            HAS_TORCH and
            getattr(getattr(sys.modules.get("torch"), "backends", None), "mps", None) and
            getattr(sys.modules["torch"].backends.mps, "is_available", lambda: False)()
        ),
    }

@app.post("/api/settings/install")
async def install_package(pip_name: str = Form(...), extra_index: str = Form(""),
                          force_reinstall: bool = Form(False)):
    global install_running, install_log
    if install_running:
        raise HTTPException(400, "Install already running")
    install_running = True
    _reset_install_progress()
    packages = [p for p in pip_name.strip().split() if p]
    install_log = [f"$ pip install {' '.join(packages)}" +
                   (" --force-reinstall" if force_reinstall else "")]

    def _run():
        global install_running
        is_torch = any(p.startswith('torch') for p in packages)

        if force_reinstall and is_torch:
            install_log.append("[*] Uninstalling existing torch/torchvision…")
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y",
                 "torch", "torchvision", "torchaudio", *_PIP_FLAGS],
                capture_output=True)
            install_log.append("[*] Installing GPU version…")

        torch_pkgs = [p for p in packages if p.startswith('torch')]
        other_pkgs = [p for p in packages if not p.startswith('torch')]

        def _stream_pip(pkgs, use_index):
            cmd = [sys.executable, "-m", "pip", "install", *pkgs,
                   "--progress-bar", "on", "--no-color", "--no-cache-dir", *_PIP_FLAGS]
            if use_index and extra_index:
                cmd += ["--index-url", extra_index]
            if force_reinstall and is_torch:
                cmd.append("--upgrade")
            elif force_reinstall:
                cmd.append("--force-reinstall")

            env = {**os.environ,
                   "PYTHONUNBUFFERED": "1",
                   "PIP_PROGRESS_BAR": "on",
                   "COLUMNS": "120",
                   "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
                   "DISPLAY": os.environ.get("DISPLAY", ":0")}

            # On Windows, also open a visible cmd window showing the full pip output
            # (ASCII progress bars render correctly in a real terminal).
            # The window uses /c so it closes automatically when pip finishes.
            if sys.platform == "win32":
                try:
                    # Build a cmd string without --no-color so the terminal looks nicer
                    term_cmd = [sys.executable, "-m", "pip", "install", *pkgs,
                                "--progress-bar", "on", *_PIP_FLAGS]
                    if use_index and extra_index:
                        term_cmd += ["--index-url", extra_index]
                    if force_reinstall and is_torch:
                        term_cmd.append("--upgrade")
                    elif force_reinstall:
                        term_cmd.append("--force-reinstall")
                    subprocess.Popen(
                        ["cmd", "/c", " ".join(f'"{a}"' if " " in a else a for a in term_cmd)
                         + " & echo. & echo Done — you can close this window. & pause"],
                        creationflags=subprocess.CREATE_NEW_CONSOLE,
                        env=env, close_fds=True)
                    log("[install] opened separate cmd window for live progress")
                except Exception as _te:
                    log(f"[install] could not open terminal window: {_te}")

            import codecs, threading as _th, time as _time
            _decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
            _buf = []  # character buffer for current line

            def _flush_buf():
                line = ''.join(_buf).strip()
                _buf.clear()
                if not line: return
                # Update structured progress (drives the real progress bar in the UI)
                is_progress = _parse_progress_line(line)
                # Maintain the ASCII text log as before
                if is_progress or (("/" in line) and any(u in line for u in
                        ["MB","kB","KB","MiB","GiB","GB"]) and ("━" in line or "─" in line or "/s" in line)):
                    if install_log and install_log[-1].startswith("↓"):
                        install_log[-1] = "↓ " + line
                    else:
                        install_log.append("↓ " + line)
                elif "Installing collected" in line or "Building wheel" in line:
                    install_log.append("⚙ " + line)
                else:
                    install_log.append(line)

            spinner = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']

            def _heartbeat(t0, i):
                elapsed = int(_time.time() - t0)
                mins, secs = divmod(elapsed, 60)
                last = install_log[-1] if install_log else ''
                if not last.startswith('↓') and not last.startswith('⚙'):
                    hb = f"{spinner[i % len(spinner)]} Working… {mins}:{secs:02d}"
                    if install_log and install_log[-1] and install_log[-1][0] in '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏':
                        install_log[-1] = hb
                    else:
                        install_log.append(hb)

            def _ingest_text(text):
                for ch in text:
                    if ch == '\r' or ch == '\n':
                        _flush_buf()
                    else:
                        _buf.append(ch)

            rc = None
            # --- Unix: run pip inside a pseudo-terminal so its Rich progress bar
            #     renders the granular "x/y MB · speed · eta" lines we parse. ---
            if os.name == "posix":
                try:
                    import pty, select
                    master, slave = pty.openpty()
                    proc = subprocess.Popen(cmd, stdout=slave, stderr=slave,
                                            stdin=slave, env=env, close_fds=True)
                    os.close(slave)
                    t0 = _time.time(); i = 0
                    while True:
                        try:
                            r, _, _ = select.select([master], [], [], 0.3)
                        except (OSError, ValueError):
                            break
                        if r:
                            try:
                                data = os.read(master, 4096)
                            except OSError:
                                break
                            if not data:
                                break
                            _ingest_text(_decoder.decode(data))
                        else:
                            if proc.poll() is not None:
                                break
                            _heartbeat(t0, i); i += 1
                    _flush_buf()
                    try: os.close(master)
                    except OSError: pass
                    proc.wait()
                    rc = proc.returncode
                except Exception as _pty_err:
                    log(f"[install] pty unavailable ({_pty_err}); using pipe")
                    rc = None

            # --- Windows / pty fallback: pipe-based byte reader ---
            if rc is None:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, bufsize=0, env=env)
                def _read_stream(stream):
                    while True:
                        raw = stream.read(1)
                        if not raw: break
                        _ingest_text(_decoder.decode(raw))
                    _flush_buf()
                t_read = _th.Thread(target=_read_stream, args=(proc.stdout,), daemon=True)
                t_read.start()
                t0 = _time.time(); i = 0
                while proc.poll() is None:
                    _time.sleep(0.3)
                    _heartbeat(t0, i); i += 1
                t_read.join(timeout=2)
                proc.wait()
                rc = proc.returncode

            # Remove trailing heartbeat line
            if install_log and install_log[-1] and any(install_log[-1].startswith(s) for s in spinner):
                install_log.pop()
            return rc

        try:
            if torch_pkgs and other_pkgs:
                install_log.append(f"[1/2] Installing {' '.join(torch_pkgs)}…")
                rc = _stream_pip(torch_pkgs, use_index=True)
                if rc != 0:
                    install_log.append(f"❌ Failed (exit {rc})")
                    install_running = False; return
                install_log.append(f"[2/2] Installing {' '.join(other_pkgs)}…")
                rc = _stream_pip(other_pkgs, use_index=False)
            else:
                rc = _stream_pip(packages, use_index=True)

            if rc == 0:
                install_log.append("✅ Done.")
                # Try live-refresh for onnxruntime (doesn't need restart)
                needs_restart = True
                if any(p.startswith("onnxruntime") for p in packages):
                    try:
                        sys.modules.pop("onnxruntime", None)
                        import onnxruntime as ort_new
                        global HAS_ONNX, _ORT_AVAILABLE_PROVIDERS
                        HAS_ONNX = True
                        _ORT_AVAILABLE_PROVIDERS = set(ort_new.get_available_providers())
                        install_log.append(f"  ONNX providers: {sorted(_ORT_AVAILABLE_PROVIDERS)}")
                        needs_restart = False
                    except Exception:
                        pass
                if needs_restart:
                    install_log.append("⚠ Please restart eScalator to apply changes.")
            else:
                install_log.append(f"❌ Failed (exit {rc})")
        except Exception as e:
            install_log.append(f"❌ Error: {e}")
        finally:
            install_running = False
            install_progress.update({"phase": "done", "pct": 100,
                                     "label": "Done", "speed": "", "eta": ""})

    threading.Thread(target=_run, daemon=True).start()
    return {"started": True}

@app.post("/api/restart")
async def restart_app():
    """Restart eScalator using the same venv Python."""
    import threading, platform
    def _do_restart():
        import time, subprocess
        time.sleep(0.5)
        base = Path(__file__).parent
        if platform.system() == "Windows":
            venv_py = base / ".venv" / "Scripts" / "python.exe"
        else:
            venv_py = base / ".venv" / "bin" / "python"
        # Fall back to current interpreter if venv not found
        py = str(venv_py) if venv_py.exists() else sys.executable
        subprocess.Popen([py, str(base / "app.py")], cwd=str(base))
        os._exit(0)
    threading.Thread(target=_do_restart, daemon=True).start()
    return {"restarting": True}

@app.get("/api/settings/install-log")
def get_install_log():
    return {"running": install_running, "lines": install_log,
            "progress": install_progress}

app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    print("=" * 60)
    print("  eScalator  –  http://localhost:7860")
    print(f"  Device  : {DEVICE}  |  fp16: {USE_FP16}")
    if HAS_TORCH:
        print(f"  CUDA ok : {torch.cuda.is_available()}  |  torch {torch.__version__}")
    if HAS_TORCH and torch.cuda.is_available():
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM    : {torch.cuda.get_device_properties(0).total_memory//1024//1024} MB")
        print(f"  tf32    : {torch.backends.cudnn.allow_tf32}")
        print(f"  cuDNN   : {torch.backends.cudnn.version()}")
    print(f"  Torch   : {torch.__version__ if HAS_TORCH else 'not installed'}")
    if HAS_ONNX:
        print(f"  ONNX    : {sorted(_ORT_AVAILABLE_PROVIDERS)}")
    print(f"  Models  : {cfg['models_dir']}")
    print(f"  Output  : {OUTPUT_DIR.resolve()}")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="warning")
