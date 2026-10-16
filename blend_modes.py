"""
eScalator - Photoshop-style blend modes for the "Overlay" pipeline action.

base  = result of the previous pipeline step ("a", bottom layer)
blend = output of the overlay model           ("b", top layer)
Both layers are fully opaque; transparency is handled separately (alpha track).
All maths in float32 0..1. Formulas follow Photoshop / the W3C compositing spec;
Soft Light uses Photoshop's variant, the non-separable modes (Hue, Saturation,
Color, Luminosity) use the W3C SetLum/SetSat/ClipColor definitions.
"""
import numpy as np
from PIL import Image

# (key, label) in Photoshop menu order; the UI builds its dropdown from the same keys
BLEND_MODES = [
    ("normal", "Normal"), ("dissolve", "Dissolve"),
    ("darken", "Darken"), ("multiply", "Multiply"), ("color_burn", "Color Burn"),
    ("linear_burn", "Linear Burn"), ("darker_color", "Darker Color"),
    ("lighten", "Lighten"), ("screen", "Screen"), ("color_dodge", "Color Dodge"),
    ("linear_dodge", "Linear Dodge (Add)"), ("lighter_color", "Lighter Color"),
    ("overlay", "Overlay"), ("soft_light", "Soft Light"), ("hard_light", "Hard Light"),
    ("vivid_light", "Vivid Light"), ("linear_light", "Linear Light"),
    ("pin_light", "Pin Light"), ("hard_mix", "Hard Mix"),
    ("difference", "Difference"), ("exclusion", "Exclusion"),
    ("subtract", "Subtract"), ("divide", "Divide"),
    ("hue", "Hue"), ("saturation", "Saturation"), ("color", "Color"), ("luminosity", "Luminosity"),
]
MODE_KEYS = {k for k, _ in BLEND_MODES}
_EPS = 1e-6


# ── Helpers for the non-separable modes (W3C compositing spec) ────────────────
def _lum(c):
    return c[..., 0:1] * 0.3 + c[..., 1:2] * 0.59 + c[..., 2:3] * 0.11

def _clip_color(c):
    l = _lum(c)
    n = c.min(axis=-1, keepdims=True)
    x = c.max(axis=-1, keepdims=True)
    c = np.where(n < 0, l + (c - l) * l / np.maximum(l - n, _EPS), c)
    c = np.where(x > 1, l + (c - l) * (1 - l) / np.maximum(x - l, _EPS), c)
    return c

def _set_lum(c, l):
    return _clip_color(c + (l - _lum(c)))

def _sat(c):
    return c.max(axis=-1, keepdims=True) - c.min(axis=-1, keepdims=True)

def _set_sat(c, s):
    mn = c.min(axis=-1, keepdims=True)
    mx = c.max(axis=-1, keepdims=True)
    rng = mx - mn
    return np.where(rng > _EPS, (c - mn) * s / np.maximum(rng, _EPS), 0.0)


# ── Separable helpers ─────────────────────────────────────────────────────────
def _color_burn(a, b):
    r = 1 - np.minimum(1, (1 - a) / np.maximum(b, _EPS))
    r = np.where(b <= 0, 0.0, r)
    return np.where(a >= 1, 1.0, r)

def _color_dodge(a, b):
    r = np.minimum(1, a / np.maximum(1 - b, _EPS))
    r = np.where(b >= 1, 1.0, r)
    return np.where(a <= 0, 0.0, r)

def _hard_light(a, b):
    return np.where(b <= 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))


def _blend(a, b, mode):
    """Raw blend result (before opacity). a, b: float32 HxWx3."""
    if mode == "normal":        return b
    if mode == "darken":        return np.minimum(a, b)
    if mode == "multiply":      return a * b
    if mode == "color_burn":    return _color_burn(a, b)
    if mode == "linear_burn":   return a + b - 1
    if mode == "lighten":       return np.maximum(a, b)
    if mode == "screen":        return 1 - (1 - a) * (1 - b)
    if mode == "color_dodge":   return _color_dodge(a, b)
    if mode == "linear_dodge":  return a + b
    if mode == "overlay":       return _hard_light(b, a)
    if mode == "soft_light":    # Photoshop formula
        return np.where(b <= 0.5, 2 * a * b + a * a * (1 - 2 * b),
                        2 * a * (1 - b) + np.sqrt(np.maximum(a, 0)) * (2 * b - 1))
    if mode == "hard_light":    return _hard_light(a, b)
    if mode == "vivid_light":
        return np.where(b <= 0.5, _color_burn(a, 2 * b), _color_dodge(a, 2 * b - 1))
    if mode == "linear_light":  return a + 2 * b - 1
    if mode == "pin_light":
        return np.where(b <= 0.5, np.minimum(a, 2 * b), np.maximum(a, 2 * b - 1))
    if mode == "hard_mix":      return (a + b >= 1).astype(np.float32)
    if mode == "difference":    return np.abs(a - b)
    if mode == "exclusion":     return a + b - 2 * a * b
    if mode == "subtract":      return a - b
    if mode == "divide":
        return np.where(b <= 0, np.where(a <= 0, 0.0, 1.0), a / np.maximum(b, _EPS))
    if mode in ("darker_color", "lighter_color"):
        la, lb = a.sum(-1, keepdims=True), b.sum(-1, keepdims=True)
        pick_b = (lb < la) if mode == "darker_color" else (lb > la)
        return np.where(pick_b, b, a)
    # Non-separable modes (W3C)
    if mode in ("hue", "saturation", "color"):
        if mode == "hue":        return _set_lum(_set_sat(b, _sat(a)), _lum(a))
        if mode == "saturation": return _set_lum(_set_sat(a, _sat(b)), _lum(a))
        return _set_lum(b, _lum(a))                       # color
    if mode == "luminosity":
        return _set_lum(a, _lum(b))
    raise ValueError(f"Unknown blend mode: {mode}")


def blend_images(base: Image.Image, top: Image.Image, mode: str = "normal",
                 opacity: float = 1.0, chunk_rows: int = 512) -> Image.Image:
    """Blend `top` over `base` (same size, RGB) with a Photoshop blend mode and opacity 0..1.
    Works in row chunks to keep memory bounded for very large images."""
    if mode not in MODE_KEYS:
        raise ValueError(f"Unknown blend mode: {mode}")
    if base.size != top.size:
        raise ValueError(f"blend_images: size mismatch {base.size} vs {top.size}")
    opacity = float(min(1.0, max(0.0, opacity)))
    a_all = np.asarray(base.convert("RGB"))
    b_all = np.asarray(top.convert("RGB"))
    if opacity <= 0:
        return Image.fromarray(a_all.copy(), "RGB")
    out = np.empty_like(a_all)
    h = a_all.shape[0]
    rng = np.random.default_rng(0)             # deterministic Dissolve pattern
    for y0 in range(0, h, chunk_rows):
        y1 = min(h, y0 + chunk_rows)
        a = a_all[y0:y1].astype(np.float32) / 255.0
        b = b_all[y0:y1].astype(np.float32) / 255.0
        if mode == "dissolve":
            # Opacity is the probability that a pixel shows the top layer
            pick = rng.random((y1 - y0, a.shape[1], 1)) < opacity
            r = np.where(pick, b, a)
        else:
            r = np.clip(_blend(a, b, mode), 0.0, 1.0)
            if opacity < 1.0:
                r = a + (r - a) * opacity
        out[y0:y1] = (np.clip(r, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(out, "RGB")
