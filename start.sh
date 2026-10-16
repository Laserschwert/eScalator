#!/usr/bin/env bash
# eScalator launcher — macOS / Linux
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
APP_VERSION="$(head -n1 VERSION 2>/dev/null | tr -d '\r\n')"; [ -n "$APP_VERSION" ] || APP_VERSION="?"

VENV=".venv"
PYTHON=""
IS_MAC=false
IS_LINUX=false

[[ "$OSTYPE" == "darwin"* ]] && IS_MAC=true
[[ "$OSTYPE" == "linux"*  ]] && IS_LINUX=true
ARCH="$(uname -m)"

SEP="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── macOS: check for Homebrew Python paths explicitly ────────────────────────
if $IS_MAC; then
  # Add common Homebrew Python paths that may not be in PATH
  for brew_py in \
    /opt/homebrew/bin/python3.12 \
    /opt/homebrew/bin/python3.11 \
    /opt/homebrew/bin/python3.10 \
    /usr/local/bin/python3.12 \
    /usr/local/bin/python3.11 \
    /usr/local/bin/python3.10 \
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
    /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
    /Library/Frameworks/Python.framework/Versions/3.10/bin/python3; do
    if [ -f "$brew_py" ] && "$brew_py" -c "import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)" 2>/dev/null; then
      export PATH="$(dirname "$brew_py"):$PATH"
      break
    fi
  done
fi

# ── Linux: check WebKitGTK (required for pywebview) ──────────────────────────
if $IS_LINUX; then
  WEBKIT_OK=false
  for ver in 4.1 4.0; do
    if python3 -c "import gi; gi.require_version('WebKit2','$ver'); from gi.repository import WebKit2" 2>/dev/null; then
      WEBKIT_OK=true; break
    fi
  done

  if ! $WEBKIT_OK; then
    echo "$SEP"
    echo "  [ERROR] WebKitGTK not found."
    echo "  pywebview requires WebKitGTK to display the UI."
    echo ""
    echo "  Install for your distro:"
    echo "  Ubuntu 22.04:  sudo apt install python3-gi python3-gi-cairo gir1.2-webkit2-4.0 gir1.2-gtk-3.0"
    echo "  Ubuntu 24.04+: sudo apt install python3-gi python3-gi-cairo gir1.2-webkit2-4.1 gir1.2-gtk-3.0"
    echo "  Fedora:        sudo dnf install python3-gobject python3-gobject-devel webkit2gtk4.1"
    echo "  Arch:          sudo pacman -S python-gobject webkit2gtk-4.1"
    echo "  openSUSE:      sudo zypper install python3-gobject typelib-1_0-WebKit2-4_1"
    echo ""
    echo "  Note: these are system packages — install with sudo, then re-run start.sh"
    echo "$SEP"
    exit 1
  fi

  # Force GTK backend for pywebview on Linux (avoids Qt conflicts)
  export PYWEBVIEW_GUI=gtk
  # Wayland: force X11 backend (pywebview/GTK Wayland has known issues)
  if [ "${GDK_BACKEND:-}" = "" ]; then
    if [ "${WAYLAND_DISPLAY:-}" != "" ] || [ "${XDG_SESSION_TYPE:-}" = "wayland" ]; then
      export GDK_BACKEND=x11
    fi
  fi
fi

# ── Find Python 3.10+ ─────────────────────────────────────────────────────────
for candidate in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$candidate" &>/dev/null; then
    if "$candidate" -c "import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)" 2>/dev/null; then
      PYTHON="$candidate"; break
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  echo "$SEP"
  echo "  [ERROR] Python 3.10 or newer not found."
  if $IS_MAC; then
    echo "  Install via Homebrew:  brew install python@3.12"
    echo "  Or from:               https://python.org"
    echo "  Apple's built-in Python (2.7/3.x stub) is not supported."
  elif $IS_LINUX; then
    echo "  Ubuntu/Debian: sudo apt install python3.12"
    echo "  Fedora:        sudo dnf install python3.12"
    echo "  Arch:          sudo pacman -S python"
  else
    echo "  Download from: https://python.org"
  fi
  echo "$SEP"
  exit 1
fi

# ── Create venv ────────────────────────────────────────────────────────────────
if [ ! -f "$VENV/bin/python" ]; then
  echo "$SEP"
  echo "  eScalator v$APP_VERSION"
  echo "  Creating virtual environment with $PYTHON..."
  "$PYTHON" -m venv "$VENV"

  # macOS: install pyobjc for pywebview (needed for WKWebView bridge)
  if $IS_MAC; then
    echo "  Installing macOS UI bridge (pyobjc)..."
    "$VENV/bin/python" -m pip install -q --upgrade pip
    "$VENV/bin/python" -m pip install -q pyobjc-core pyobjc-framework-Cocoa pyobjc-framework-WebKit
  fi
  echo "  Done."
fi

VENV_PYTHON="$VENV/bin/python"

# ── Verify / install base dependencies ────────────────────────────────────────
"$VENV_PYTHON" -c "import fastapi,uvicorn,multipart,webview,PIL,numpy,packaging,scipy" 2>/dev/null || {
  echo "$SEP"
  echo "  eScalator v$APP_VERSION"
  echo "  Installing missing base dependencies..."
  "$VENV_PYTHON" -m pip install -q --upgrade pip
  "$VENV_PYTHON" -m pip install --no-cache-dir --progress-bar on -r requirements-base.txt
  # macOS: ensure pyobjc is present even on existing venvs
  if $IS_MAC; then
    "$VENV_PYTHON" -m pip install -q pyobjc-core pyobjc-framework-Cocoa pyobjc-framework-WebKit 2>/dev/null || true
  fi
  echo "  Done."
}

# ── Launch ─────────────────────────────────────────────────────────────────────
echo "$SEP"
echo "  eScalator v$APP_VERSION"
if $IS_MAC; then
  if [[ "$ARCH" == "arm64" ]]; then
    echo "  Apple Silicon (M-series) — MPS available via PyTorch"
  else
    echo "  macOS Intel — CPU mode (no MPS)"
  fi
fi
echo "  Launching...  (Settings tab for GPU/deps setup)"
echo "$SEP"
exec "$VENV_PYTHON" app.py
