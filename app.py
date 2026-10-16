"""
eScalator – Desktop App Entry Point
Starts FastAPI in a background thread, then opens a native pywebview window.
No browser needed. Uses the system WebView2 renderer on Windows (always available
on Win10/11), WebKitGTK on Linux, WKWebView on macOS.
"""

import sys
import os
import subprocess
import socket
import threading
import time
import tempfile
import shutil
import atexit
from pathlib import Path

# ── Ensure base dependencies are installed (catches gaps after venv recreation) ─
def _ensure_base_deps():
    base = Path(__file__).parent
    req = base / "requirements-base.txt"
    if not req.exists():
        return
    try:
        import fastapi, uvicorn, multipart, webview, PIL, numpy, packaging  # noqa
    except ImportError:
        print("[*] Installing missing base dependencies…")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
                       check=False)
        # macOS: ensure pyobjc bridge for WKWebView is present
        if sys.platform == "darwin":
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "pyobjc-core", "pyobjc-framework-Cocoa",
                            "pyobjc-framework-WebKit"], check=False)

_ensure_base_deps()

# ── Pick a free port ───────────────────────────────────────────────────────────
def find_free_port(preferred: int = 7860) -> int:
    try:
        with socket.socket() as s:
            s.bind(("127.0.0.1", preferred))
            return preferred
    except OSError:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

PORT = find_free_port()

# WebView2 stores its user-data (cache, cookies, GPU shader cache) in a
# subfolder of the app directory by default — this holds file locks that
# prevent deleting the folder after exit. Put it in the system temp dir
# and clean it up on exit instead.
_webview2_data_dir = tempfile.mkdtemp(prefix="eScalator_webview2_")

def _cleanup():
    """Called at exit: free VRAM, remove WebView2 temp profile."""
    # 1. Free VRAM
    try:
        import server as srv
        if srv.loaded_models:
            print("  Unloading models from VRAM…")
            srv.loaded_models.clear()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("  VRAM cleared.")
        except Exception:
            pass
    except Exception:
        pass

    # 2. Remove WebView2 user-data dir (releases all file locks)
    try:
        shutil.rmtree(_webview2_data_dir, ignore_errors=True)
    except Exception:
        pass

atexit.register(_cleanup)

# ── Start FastAPI server in background ────────────────────────────────────────
def start_server():
    import uvicorn
    import server as srv  # noqa — work/ dirs already configured in server.py
    uvicorn.run(srv.app, host="127.0.0.1", port=PORT, log_level="warning")

server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()

# Wait until the server is actually accepting connections
def wait_for_server(timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.05)
    return False

print("\n\n" + "=" * 60)
APP_VERSION = (Path(__file__).parent / "VERSION").read_text(encoding="utf-8").strip() \
    if (Path(__file__).parent / "VERSION").exists() else "?"
print(f"  eScalator v{APP_VERSION}")
print(f"  Starting server on port {PORT}…")

if not wait_for_server():
    print("[ERROR] Server failed to start — check the log above.")
    sys.exit(1)

print("  Server ready — opening window")
print("=" * 60)

# ── Open native desktop window via pywebview ──────────────────────────────────
import webview

# Expose a minimal JS API so the frontend can call native file dialogs
class NativeAPI:
    """Methods exposed to JS as window.pywebview.api.*"""

    def open_file_dialog(self):
        """Open a native file-open dialog, return chosen path or None."""
        result = window.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=False,
            file_types=("Image files (*.png;*.jpg;*.jpeg;*.webp;*.tif;*.tiff;*.bmp)",
                        "All files (*.*)")
        )
        return result[0] if result else None

    def open_files_dialog(self):
        """Open a native file-open dialog, return list of paths."""
        result = window.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=True,
            file_types=("Image files (*.png;*.jpg;*.jpeg;*.webp;*.tif;*.tiff;*.bmp)",
                        "All files (*.*)")
        )
        return list(result) if result else []

    def open_file_dialog_multi(self):
        return self.open_files_dialog()

    def save_file_dialog(self, suggested_name: str = "upscaled.png"):
        """Open a native save dialog, return chosen path or None."""
        result = window.create_file_dialog(
            webview.SAVE_DIALOG,
            save_filename=suggested_name,
            file_types=("PNG image (*.png)", "WEBP image (*.webp)",
                        "JPEG image (*.jpg)", "TIFF image (*.tiff)",
                        "All files (*.*)")
        )
        return result if result else None

    def select_folder_dialog(self, initial: str = ""):
        """Open a native folder-picker dialog, return path or None."""
        try:
            result = window.create_file_dialog(webview.FileDialog.FOLDER)
        except AttributeError:
            result = window.create_file_dialog(webview.FOLDER_DIALOG)
        return result[0] if result else None

    def get_port(self):
        return PORT


from webview.dom import DOMEventHandler

def on_native_drop(e):
    """Called by pywebview on drop — provides pywebviewFullPath for each file."""
    files = e.get('dataTransfer', {}).get('files', [])
    paths = [f.get('pywebviewFullPath','') for f in files if f.get('pywebviewFullPath')]
    if not paths:
        return
    import json
    window.evaluate_js(f'window._nativeDropPaths({json.dumps(paths)})')

def bind_events(w):
    """Called by webview.start(bind, window) — runs after window is ready."""
    w.dom.document.events.dragover += DOMEventHandler(lambda e: None, True, True, debounce=50)
    w.dom.document.events.drop     += DOMEventHandler(on_native_drop, True, True)

def on_closing():
    """Called when the window X button is pressed — clean up before exit."""
    _cleanup()
    # Do NOT call sys.exit() here — this runs inside a .NET/pythonnet
    # event handler (FormClosing) and sys.exit raises SystemExit which
    # .NET cannot handle, producing a crash dialog.
    # pywebview will destroy the window and exit normally after this returns.


api = NativeAPI()

# Check if critical inference deps are present — open settings tab if missing
def _has_inference_deps():
    """Check if inference deps are installed using metadata (avoids stale import cache)."""
    import importlib.metadata
    try:
        importlib.metadata.version("spandrel")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False

start_url = f"http://127.0.0.1:{PORT}/"
if not _has_inference_deps():
    start_url = f"http://127.0.0.1:{PORT}/?open=settings"
    print("  NOTE: spandrel not found — opening Settings tab for dependency setup")

window = webview.create_window(
    title       = "eScalator",
    url         = start_url,
    js_api      = api,
    width       = 1280,
    height      = 820,
    min_size    = (900, 600),
    text_select = False,
)
window.events.closing += on_closing

# Give server module a reference to the window for direct pixel push
try:
    import server as _srv
    _srv._webview_window = window
except Exception:
    pass

# Pass WebView2 user-data dir via environment so pywebview picks it up.
# This keeps all WebView2 cache/locks out of the app folder.
os.environ["PYWEBVIEW_USER_DATA_DIR"] = _webview2_data_dir

# Start the GUI event loop (blocks until window is closed).
# We redirect stderr to nul just before starting to suppress WebView2's
# harmless "Failed to unregister class Chrome_WidgetWin_0" error on exit.
if sys.platform == "win32":
    try:
        import ctypes, msvcrt
        nul_handle = ctypes.windll.kernel32.CreateFileW(
            "nul", 0x40000000, 3, None, 3, 0x80, None)
        ctypes.windll.kernel32.SetStdHandle(-12, nul_handle)
        # Also redirect Python's stderr fd so C extensions see it too
        _stderr_fd = sys.stderr.fileno()
        _old_stderr_fd = os.dup(_stderr_fd)
        _nul_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_nul_fd, _stderr_fd)
        os.close(_nul_fd)
    except Exception:
        _old_stderr_fd = None
else:
    _old_stderr_fd = None

webview.start(bind_events, window, debug=False)

# Restore stderr after window closes (in case anything needs it post-exit)
if _old_stderr_fd is not None:
    try:
        os.dup2(_old_stderr_fd, sys.stderr.fileno())
        os.close(_old_stderr_fd)
    except Exception:
        pass
