# eScalator v1.3

A fast, AI-powered image upscaling and processing tool with a native desktop UI. No compilation required — double-click to run.

## Quick Start

**Windows:** Double-click `start.bat`  
**macOS / Linux:** `chmod +x start.sh && ./start.sh`

On first run, a local `.venv` is created and base dependencies are installed automatically. Nothing touches your system Python. Subsequent launches reuse the venv and start in seconds.

If a required GPU package is missing, the app opens on the **Settings** tab.

---

## Dependencies

The Settings tab shows all packages with status and download sizes. Install in order:

1. **PyTorch** — CPU, CUDA, or ROCm (Linux AMD). Installs the correct build from the PyTorch index; removes any conflicting CPU version first.
2. **Spandrel** — model loader, unlocks after PyTorch.
3. **ONNX Runtime** — optional, for `.onnx` files. Correct variant shown automatically.

A **Restart Now** button appears after each install. Click it to restart — the app reopens automatically.

During download and installation a live progress bar shows the current phase (resolving / downloading / installing), percentage, transfer speed and ETA, alongside the full text log. On Windows a separate **cmd window** also opens, showing the full pip output with native ASCII progress bars; it closes automatically when pip finishes.

ONNX Execution Providers are selected automatically: TensorRT → CUDA → ROCm → DirectML → CoreML → CPU.

---

## Modes

### Quick Mode (default)
One active model at a time. Click any model to select it — the name appears in the Models panel header and persists across restarts. Output filename: `image__ModelName.ext`

### Pipeline Mode
Chain models and actions in any order. Output filename: `image__pipeline_YYYYMMDDHHMMSS.ext`

Drag chips into the pipeline (top 25% = insert before, middle = replace, bottom 25% = insert after). Drag existing steps to reorder. Both modes preserve their state independently when switching.

Each step has a checkbox on its left (enabled by default). Uncheck it to skip that step without removing it from the pipeline — useful for A/B testing a single action or model. Disabled steps are dimmed and struck through, are excluded from previews, processing, batch and the predicted output size, and their on/off state is saved with presets.

**Per-model channel settings** are configurable in the Model Info inspector (click a model step). The model name is shown right-aligned in the **Model Info** header:
- **Channels** — `RGB with Alpha` (default): model processes all channels; `RGB only`: model processes RGB, the alpha channel is resized with the chosen filter; `Alpha only`: model processes the alpha channel, RGB is resized.
- **Upscale other channels** — filter used to resize whichever channels the model does *not* process (Lanczos / Bicubic / Bilinear / Nearest).

**Pre/Post-Scale labels** show *before model* / *after model* in Quick mode and *before pipeline* / *after pipeline* in Pipeline mode.

The **Upscale Alpha** checkbox (Processing panel) is only shown and relevant in Quick mode; Pipeline mode uses the per-model channel settings instead.

**Missing models:** if a pipeline references a model whose file no longer exists — deleted, or a preset opened on another machine — the step is kept as a red, locked-off entry showing the model name (so you know exactly which file is missing). It cannot be enabled and is always skipped. Clicking it shows a **Model File Not Found** notice with the model name and path in the inspector. If the file reappears (e.g. after refreshing the model folder), the step recovers automatically and can be re-enabled.

---

## Actions

Actions are colour-coded in the pipeline and the action chip list: **Blend, Discrete and Overlay** are violet (same as model steps); **Resize, Flip and Crop** are blue; all other actions are green.

| Action | Description |
|--------|-------------|
| Blend | Run two models and blend outputs at a configurable ratio |
| Discrete | Run separate models for the RGB and Alpha channels independently |
| Overlay | Run a model and layer its output over the previous step with opacity and a Photoshop blend mode |
| Resize | Scale by %, absolute px, longest side, or shortest side |
| Flip | Horizontal and/or vertical |
| Crop | By percent or pixels (top / bottom / left / right) |
| HSL | Hue (±180°), Saturation (±100%), Lightness (±100%) |
| Levels | Input/output black & white points + gamma |
| Brightness/Contrast | ±100% each, RGB or Alpha channel |
| Color Temperature | Warm/cool shift |
| Invert | Invert all color channels |
| Grayscale | Luminosity, Average, or Lightness method |
| Alpha Threshold | Binarise alpha channel at a threshold |

**Blend action** — drop two models into the Model A / Model B slots and blend them at the Factor setting, shown as `A X% / Y% B` with live percentage update. Incompatible scale factors are detected immediately and shown as a red **Incompatible models** warning. The Blend action also has its own **Channels** and **Upscale other channels** settings (same as model steps).

**Overlay action** — one model slot plus **Opacity** (0–100 %) and a **Blend mode** dropdown with all 27 Photoshop modes (Normal, Dissolve, Darken, Multiply, Color Burn, Linear Burn, Darker Color, Lighten, Screen, Color Dodge, Linear Dodge (Add), Lighter Color, Overlay, Soft Light, Hard Light, Vivid Light, Linear Light, Pin Light, Hard Mix, Difference, Exclusion, Subtract, Divide, Hue, Saturation, Color, Luminosity). The model runs on the previous step's result; that result is resized to the model's output size with the **Other Resize** filter and used as the base layer, the model output is the top layer. If **Channels** includes Alpha, the alpha channel is upscaled by the model as well and simply cross-faded by Opacity — the blend mode applies to RGB only. A reset button appears next to opacity and blend mode when they differ from the default.

**Discrete action** — two separate model slots: one for the **RGB** channel and one for the **Alpha** channel. Each model processes its respective channel independently. An **Alpha Resize** dropdown determines how the alpha channel is reconciled if the two models produce different output sizes (Lanczos / Bicubic / Bilinear / Nearest / **No resize**). With *No resize* the sizes must match at the end of the pipeline; if they do not, a pre-flight dialog warns before processing.

**Resize action** — the **Channels** dropdown controls which track is resized (`RGB with Alpha` / `RGB only` / `Alpha only`), enabling fine-grained control when mixing differently-scaled channels in one pipeline. The percent slider snaps to specific positions (0 · 10 · 25 · 50 · 75 · 100 · 150 · 200 · 300 · 400 · 500 · 600 · 700 · 800 %). Any other value can be entered manually via the text field.

**RGB/Alpha size mismatch pre-flight** — before any processing starts, eScalator predicts the final output size for both the RGB and Alpha tracks. If a mismatch is detected and no automatic reconciliation would fix it (e.g. a Discrete step with *No resize* selected), a dialog appears: *"The current pipeline results in a size mismatch between the RGB and Alpha channels. The Alpha channel will be ignored."* Clicking **OK** proceeds with `upscale_alpha` disabled; **Cancel** aborts. Model scale factors are resolved and cached automatically whenever a model is added, so the pre-flight check is reliable without a prior run.

**Alpha handling:** geometric actions (Resize, Flip, Crop) and Alpha Threshold are applied to the alpha channel as well as RGB, so transparency stays in sync with the image. Colour-only actions (HSL, Levels, Brightness/Contrast, Color Temperature, Invert, Grayscale) leave the alpha channel untouched. With *Upscale Alpha* enabled, the alpha is run through the model steps too; otherwise it is scaled with LANCZOS.

---

## Inference

- 42+ architectures via Spandrel; FP16 ONNX models supported
- Apple Silicon MPS via standard PyTorch
- fp16 on CUDA, fp32 on MPS/CPU
- Tiled inference with configurable tile size; Auto calculates from VRAM
- Models stay loaded between runs; background pre-loading available
- CUDA OOM shows a clear error popup and frees VRAM without crashing

---

## Canvas & Previews

The canvas updates **only** when using Preview Full or Preview Cutout. Process & Save and Batch mode do not affect the canvas — they save to disk and update the info panel only.

| Button | What it does |
|--------|-------------|
| **Preview (Full)** | Upscale entire image → display in canvas |
| **Preview (Cutout)** | Upscale visible viewport region only → display in canvas. The predicted cutout size in the info panel matches the actual processed region exactly. |
| **Process & Save** | Upscale entire image → save to disk, no canvas update |

Results are transferred to the canvas as raw straight-alpha RGBA bytes in a single binary request (`/api/result-raw`) — no PNG encoding, no base64.

A **Clear Preview** button is centered at the top of the canvas. It discards the current result from memory and returns to the original-only view, exactly as if the image had just been loaded (the source image stays loaded).

The **Zoom Filter** toggle (Bicubic / Nearest) sits at the left end of the canvas hint bar. A single click anywhere on it toggles between the two filters.

### Navigation
- **Scroll** — zoom
- **LMB hold** — compare result with original (preview modes only)
- **MMB drag / LMB drag** — pan
- **RMB hold** — pixel inspector (RGBA tooltip, nearest-neighbor)
- **F** — fit to window
- **1** — zoom to 100%

### Alpha Channel
Select display mode from the dropdown in the canvas toolbar:
- **RGB with Alpha** — transparency as checkerboard
- **Alpha** — alpha channel as greyscale
- **Raw RGB** — RGB without alpha pre-multiplication

Hover any dropdown (here or in Settings) and scroll the mouse wheel to cycle options without opening it.

### Canvas Settings (in Settings tab)
- **Max preview size** — pixel budget for the canvas preview: 16.8 / 67 (default) / 151 / 268 MP (= 4096² … 16384²). The aspect ratio is free, e.g. 268 MP allows 32768×8192. 268 MP is the browser's hard canvas limit regardless of VRAM. Display only; Process & Save is not limited and always writes full resolution.

---

## Info Panel

Top-right of canvas. Updates live on every pan and zoom.

| Row | Dimensions | Cutout | Zoom |
|-----|-----------|--------|------|
| **Source** | Full image | Current viewport size | Current canvas zoom — *click to set 100%* |
| **Last** | Last upscaled result | — | Pixel-exact zoom for result — *click to match* |
| **Output** | Predicted output (shows `?` if model scale not yet known) | Predicted cutout output (red if over canvas limit) | Predicted pixel-exact zoom — *click to match* |
| **Saved** | Last saved filename | | |
| **VRAM** | Free / Total GB — *hover to show Clear VRAM, click to unload all models* | | |

VRAM color: green ≥50% free · yellow <50% · orange <25% · red <10%

---

## Saving & Output

- **Process & Save** — saves next to source file, or to a custom output folder
- **Save As…** — saves the current preview result via native file dialog (no re-processing)
- **Open folder** button (next to Cancel) — opens the folder of the last saved file
- **Copy to Clipboard** — three equal buttons: **RGBA** (full image with alpha), **RGB** (no alpha channel), **Alpha** (alpha channel as greyscale). All three are always visible and accessible regardless of the current view mode.

---

## Batch

Drop files or use the file picker. Uses the current Quick / Pipeline configuration.

- Per-item progress bar while processing
- **Cancel** stops after the current image
- Cancelled items re-queue on the next run
- Output to source folders or a custom directory
- Folder icon per item — opens the result folder (active after successful processing)
- Click a filename in the list — opens its source folder

---

## Presets

The **Presets** panel header contains a **+ Add Preset** button. Below it, the dropdown (65 % of sidebar width) and three buttons (Update / Reset / Delete) share the full row.

- **Update** — saves current settings into the active preset; active only when the preset is modified (marked with `*`)
- **Reset** — discards unsaved changes and reloads the saved version; active only when modified
- **Delete** — first click shows **Sure?** inline (2.5 s timeout); second click deletes

Switching presets in the dropdown is always free — no prompt even when the current preset is modified.

A preset stores mode (Quick / Pipeline), pre/post scale, tile size, output format, filter mode, and the full pipeline — every model and action with all parameters including each step's enabled/disabled state. In Quick mode the upscale-alpha setting is also saved; in Pipeline mode it is omitted (per-model channel settings replace it).

---

## Models

Set any folder via the folder button or by clicking the path field. The tree shows all `.pth`, `.safetensors`, and `.onnx` files. Scale factors are parsed from filenames and cached in `model_meta.json`. Whenever a model is added to the pipeline — by click, drag, drop into a Blend or Discrete slot, or preset load — the scale is resolved immediately via a lightweight CPU load if not already known. This means the predicted output size and the RGB/Alpha mismatch pre-flight are always accurate without having to run the model first. The active model and its folder highlight persist across restarts; on startup the folder containing the restored Quick model is expanded automatically and scrolled into view.

Recommended sources: [OpenModelDB](https://openmodeldb.info) · [chaiNNer model hub](https://github.com/chaiNNer-org/spandrel)

---

## Platform Notes

**Windows** — WebView2 built-in on Win10/11. AMD/Intel ONNX acceleration via `onnxruntime-directml`.

**macOS** — `pyobjc` bridge installed automatically. Apple Silicon uses MPS; Intel Mac uses CPU only.

**Linux** — requires WebKitGTK (system package, install with sudo):
```
Ubuntu 22.04:  sudo apt install python3-gi python3-gi-cairo gir1.2-webkit2-4.0 gir1.2-gtk-3.0
Ubuntu 24.04+: sudo apt install python3-gi python3-gi-cairo gir1.2-webkit2-4.1 gir1.2-gtk-3.0
Fedora:        sudo dnf install python3-gobject python3-gobject-devel webkit2gtk4.1
Arch:          sudo pacman -S python-gobject webkit2gtk-4.1
```
Wayland: `GDK_BACKEND=x11` and `PYWEBVIEW_GUI=gtk` are set automatically.

---

## File Structure

```
eScalator/
├── app.py                  # pywebview entry point
├── server.py               # FastAPI backend
├── start.bat / start.sh    # Launchers
├── requirements-base.txt   # Base dependencies
├── config.json             # Settings (auto-created)
├── presets.json            # Presets (auto-created)
├── model_meta.json         # Model scale cache (auto-created)
├── models/                 # Default model folder
├── work/                   # Clipboard + fallback output
└── static/index.html       # Complete frontend
```

To update: replace all files except `.venv`. Installed packages (torch, spandrel, onnxruntime) are preserved.

---

## Attributions

| Project | License |
|---------|---------|
| [Spandrel](https://github.com/chaiNNer-org/spandrel) | MIT |
| [PyTorch](https://pytorch.org) | BSD-3 |
| [ONNX Runtime](https://onnxruntime.ai) | MIT |
| [FastAPI](https://fastapi.tiangolo.com) | MIT |
| [Pillow](https://python-pillow.org) | HPND |
| [NumPy](https://numpy.org) | BSD-3 |
| [pywebview](https://pywebview.flowrl.com) | BSD-3 |
| [uvicorn](https://www.uvicorn.org) | BSD-3 |
| [Heroicons](https://heroicons.com) | MIT |
| [Iconoir](https://iconoir.com) | MIT |
| [Tabler Icons](https://tabler.io/icons) | MIT |
| [Boxicons](https://boxicons.com) | MIT |
| [Fluent UI System Icons](https://github.com/microsoft/fluentui-system-icons) | MIT |
| [IBM Plex fonts](https://www.ibm.com/plex/) | SIL OFL 1.1 |

Inspired by [Cupscale](https://github.com/DrPleaseRespect/cupscale) and [chaiNNer](https://github.com/chaiNNer-org/chaiNNer).
