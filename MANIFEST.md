# eScalator — MANIFEST

Working notes for continuing development in a fresh chat. Current version: **v1.3**.

## Architecture

| File | Role |
|---|---|
| `app.py` | Launcher: starts uvicorn + pywebview window (WebView2 on Windows). |
| `server.py` | FastAPI backend: model loading (spandrel / ONNX), tiled upscaling, pipeline + actions, alpha track, save, batch, settings/installer. |
| `static/index.html` | Entire frontend (HTML + CSS + JS in one file): canvas viewer, pipeline editor, actions library, inspector, batch, settings. |
| `start.bat` / `start.sh` | Portable setup: embedded Python + venv, dependency install, launch. |
| `VERSION` | Single source of the version number: read by app.py, server.py (`/api/config` → UI label `#tb-ver`), start.bat, start.sh. Only README title and MANIFEST must be bumped by hand. |
| `requirements-base.txt` | Base deps (scipy is required by Color Extend). |

### Processing flow
1. `/api/load-image(-path)` stores the source in `img_state`; the frontend displays `/api/image-source` via `<img>` (`iSrc`).
2. `/api/process` runs `process_pipeline()` on RGB (or RGBA when alpha is not upscaled separately).
   With `upscale_alpha=true` the alpha channel runs in its own **alpha track** (same steps, colour-only actions skipped).
3. Preview modes: the result is stored as straight RGBA bytes in `result_raw_cache`; the response is
   JSON `{raw_push:true, push_w, push_h, ...}`. The frontend's `loadRawResult()` fetches `/api/result-raw`
   (one binary request), keeps the buffer as `iRes._raw` and builds `iRes._bitmap` via
   `createImageBitmap(new ImageData(...))`.
4. Save mode: PIL writes the file server-side (RGBA merged from the alpha track).

### Actions
`apply_action()` in `server.py`; definitions/defaults in `ACTION_DEFS` and inspector UI in `index.html`.
Color Extend (`color_extend`): `scipy.ndimage.distance_transform_edt` nearest-opaque fill, `radius=0` = unlimited.
In the RGB-only track it receives the alpha via an injected `_alpha_proxy`.

### ONNX models
`run_onnx()` → tiled inference (`_onnx_tiled`) with feathered overlap blending, using the UI tile size.
On first use per model `_calibrate_onnx()` runs a 64 px sample and caches a profile in `_onnx_profiles`:
layout NCHW/NHWC (from input shape), input range 0..1 or 0..255, output range 0..1 or 0..255, scale.
Static-shape models (e.g. 1×3×64×64) are tiled at exactly that size. Dynamic tiles are padded to a multiple of 16.
On out-of-memory the tile size is halved (down to 64) automatically. NaN/Inf output or an all-black output raise
`OnnxModelError` with a readable message instead of returning a black image. CUDA EP uses
`arena_extend_strategy=kSameAsRequested`.

### Overlay action
`type:"overlay"` with `model`, `model_name`, `scale`, `channels`, `other_filter`, `opacity` (0–100), `blend_mode`.
`process_pipeline`: base = previous result resized to the model output size with `other_filter`, top = model output,
`blend_modes.blend_images()` (separate module, 27 Photoshop modes, row-chunked float32, deterministic Dissolve).
Alpha track runs the same branch with `_alpha_track=True` → blend mode ignored, alpha is cross-faded by opacity
(mode forced to "normal"). Frontend: `BLEND_MODES_UI` mirrors `BLEND_MODES`; model slot reuses the discrete drop handlers
(`slot='overlay'`); reset buttons via `resetStepValue()`, live label/reset sync in `_overlayUiSync()`.
Verified pixel-exact against an independent Python reference (RGB and alpha).

### Cancel
`/api/cancel` → `request_cancel()`: sets `cancel_event` and sets `terminate=True` on all running ONNX `RunOptions`.
`check_cancel()` is called between tiles, before/after model load, between pipeline steps, in the alpha track,
before encode and before save. Not interruptible: a single torch forward pass (one tile, or a whole image ≤ tile size)
and a single PIL action (e.g. one large resize) — cancel takes effect right after it.

### VRAM
`release_vram_cache()` runs after every `/api/process` (finally): frees PyTorch's cached tile/output buffers,
models stay loaded. "Clear VRAM" (`/api/vram/clear`) additionally unloads all models; the canvas is unaffected
(the preview lives in the browser and `result_raw_cache` in system RAM).

## Pitfalls

- **Canvas premultiplies alpha.** Anything drawn into a 2D canvas (`drawImage`, `putImageData`, `<img>` decode)
  loses RGB where alpha=0, and `canvas.convertToBlob()` writes those pixels as 0,0,0,0.
  Never derive "Raw RGB" or export data from a canvas. Use the straight buffers instead:
  - result: `iRes._raw` (from `/api/result-raw`, or the legacy tile push copy)
  - source: `iSrc._raw` (from `/api/image-source-raw`, loaded by `loadSrcRaw()` for images with alpha)
  - PNG export of preview results: `encodePngStraight()` (own PNG encoder using `CompressionStream('deflate')`)
- Clipboard: always call `navigator.clipboard.write()` synchronously in the click handler; pass the blob as a Promise.
- "Copy RGBA" to clipboard still loses RGB under alpha=0 — browser clipboard limitation, not fixable client-side.
- Reset `iRes._raw` everywhere `iRes._bitmap` is reset (`onImgLoaded`, `clearPreview`, start of preview run).
- **Canvas limit is a pixel budget, not a side length.** Chromium/Skia silently fail (black canvas) above
  16384² ≈ 268 MP, independent of VRAM; per side up to 65535 px works (verified, Chromium 141).
  `CANVAS_LIMIT` = N means W·H ≤ N²; all checks go through `fitsCanvas(w,h)`. Process & Save is not affected.
- Raw buffers cost 4 bytes/pixel in the frontend (8K×8K ≈ 256 MB, plus the bitmap).

## Packaging

Release ZIP must NOT contain user data — it would overwrite the user's settings on extract.
Exclude: `config.json`, `presets.json`, `model_meta.json`, `work/`, `.venv/`/`venv/`, `__pycache__/`, model files (`*.pth`, `*.onnx`, `*.safetensors`).
All of these are created by the app on demand (`load_config()` etc. fall back to defaults when missing).

## Fixed bugs

- v1.3: Color Extend seemed to do nothing — backend was correct, but the result display/export went through the
  premultiplied canvas. Fixed with straight-alpha buffers (see Pitfalls).
- v1.3: Source "Raw RGB" view showed black under alpha=0 (premultiplied `<img>` decode) → `/api/image-source-raw`.
- v1.3: Viewport transfer was very slow (base64 tiles via `evaluate_js`, per-byte `atob` decode: ~6 s JS-only for
  4096², plus IPC). Replaced by a single binary fetch: ~0.8 s for 4096² in headless Chromium.
- v1.3: Status bar stuck at "Sending to canvas…" after push — `showProg(false)` never resets the text; push watcher now sets "Preview W×H".
- v1.3: Copy-to-clipboard failed in the app — `clipboard.write()` was called only after several seconds of encoding,
  when the click's user activation had expired. Now `write()` is called synchronously in the click handler with a
  Promise-valued `ClipboardItem`. Buttons show feedback (hourglass after 200 ms, green check on success, red x on error).
- v1.3: `StreamingResponse(io.BytesIO(...))` iterates binary data line by line (split at random `\n` bytes) →
  `/api/alpha-result` took ~9 s for 11.7 MB. Replaced by plain `Response(content=...)` (14 ms). Never stream a BytesIO.
- v1.3: Cancel only worked between torch tiles; actions, ONNX models, alpha track, encode/save ignored it
  (job ran to completion with status 200). Now cancels within ~0.3–0.7 s in tests (see Cancel).
- v1.3: Starting a preview cleared the old result immediately; after Cancel/error the canvas was empty. Now the old
  preview stays until the new data is complete (`loadRawResult` fetches first, then swaps; PNG fallback swaps in onload).
- v1.3: After "Process & Save", `S.outputFilename` stayed set, so "Copy RGBA" after a later preview copied the saved
  file instead of the preview. Cleared when a new result is swapped in.
- v1.3: Status stuck at "Cancelling…" — the job's 499 arrived before `/api/cancel` returned; text is now set before the request.
- v1.3: ONNX models ran untiled on the whole image → OOM with transformer models (SwinIR/HAT/DAT attention);
  models expecting 0..255 input returned black; NHWC models failed; NaN output became black silently.
  New tiled/calibrated runner (see ONNX models). Verified with synthetic test models for each case.
- v1.3: "Max preview size" was never saved — `set_canvas_limit` had no `@app.post` decorator (404, error swallowed
  by the frontend). Decorator added; frontend now reports save errors and restores the dropdown on startup.
- v1.3: Removed obsolete "Display tile size" setting (UI, endpoint, config key).
- v1.3: Canvas limit checked side lengths only (e.g. 32768×5000 blocked at 16K although it fits; 32K setting
  produced silently black results). Now a pixel budget (`fitsCanvas`), 32K option removed (old configs clamped
  to 16384), messages show the real limit instead of a hard-coded 8192, ⊘/⚠ glyphs replaced by a tabler icon.
- v1.3: "Clear Preview" only cleared the browser side; the server kept the raw result (~1 GB at 16K) and alpha PNG.
  New `/api/clear-result` drops both and releases cached VRAM; status shows the freed MB.
- v1.3: ONNX models always reported scale 0 → output size shown as "?". `/api/model-scale` now probes ONNX models
  with a 32 px CPU inference (`_probe_onnx_scale`, not cancel-aware), and `run_onnx` persists the scale to model_meta.
  Order for ONNX: model_meta → filename (`_guess_scale_from_name`: 4x…, …x4…, SRx4; not persisted) → probe (persisted).
  A filename guess is replaced by the verified scale after the first real run.
- v1.3: Output-size estimate and preview-button limits did not update when changing action values (e.g. Resize
  slider) or when panning — only on zoom. Now `scheduleSizeRefresh()` (one run per animation frame) is triggered by
  every `input`/`change` event in the document, by `render()` (pan/zoom) and by the typed-value commits.
- v1.3: Size estimate read pre/post scale from the snapped slider, the run used the exact typed value (e.g. 130 %
  vs 125 %). Estimate now uses `getScaleValue()` like `runProcess`.
- v1.3: Version said 1.2 in README, app.py, start.bat, start.sh and the UI label → now central `VERSION` file.

## Open issues / ideas

- `/api/batch/run` backend is broken dead code (`pipeline_json` undefined, model paths passed as Path objects);
  the frontend batch runs `/api/process` per file instead. Remove or fix.
- `/api/preload` is called by the frontend (`triggerPreload`) but `preload_models` has no `@app.post` decorator, so
  model preloading never ran. Not enabled yet: its background thread resets `processing_status` when done and could
  clobber the status of a job started meanwhile — fix that before registering the route.
- Remaining unicode glyphs used as icons: `#err-icon` (⚠), "⚠ Restart Now" button, install-log line prefix check.
- Overlay with `channels:"rgb"` and inline alpha (upscale_alpha off) drops alpha — same pre-existing behaviour as plain model steps with `rgb`.
- PNG fallback path (when raw push fails) still loads the result via `<img>` → Raw RGB view premultiplied there.
- Legacy `_pixelPushTile` / `_pixelPushBatch` functions are unused but harmless.
- Not yet verified in the real Windows app (WebView2) with model pipelines and very large images.
