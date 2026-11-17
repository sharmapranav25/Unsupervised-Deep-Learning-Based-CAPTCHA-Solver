"""
Background Generators for Universal CAPTCHA Generator.

8 background types, each with randomizable parameters.
All generators take an Image and BackgroundConfig, modify the image in-place or return a new one.
"""

import math
import random
from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter
import numpy as np

from .config import BackgroundType, BackgroundConfig, ColorPalette


# ═══════════════════════════════════════════════════════════
# Default palettes
# ═══════════════════════════════════════════════════════════

DEFAULT_BG_COLORS = ColorPalette(colors=[
    (255, 255, 255), (245, 245, 245), (240, 240, 240),
    (235, 235, 230), (230, 235, 240), (240, 235, 230),
    (250, 248, 240), (240, 245, 250), (245, 240, 245),
])

PASTEL_PALETTE = ColorPalette(colors=[
    (240, 195, 195), (195, 240, 195), (195, 195, 240),
    (240, 240, 195), (240, 195, 240), (195, 240, 240),
    (215, 195, 240), (240, 210, 195), (200, 225, 200),
    (225, 205, 225), (240, 210, 210), (210, 220, 240),
])

FAINT_LINE_PALETTE = ColorPalette(colors=[
    (220, 195, 195), (195, 220, 195), (195, 195, 220),
    (220, 220, 195), (220, 195, 220), (195, 220, 220),
    (230, 210, 210), (210, 230, 210), (210, 210, 230),
    (180, 160, 160), (160, 180, 160), (160, 160, 180),
    (200, 180, 180), (180, 200, 180), (180, 180, 200),
])

DARK_BG_COLORS = ColorPalette(colors=[
    (30, 30, 40), (20, 25, 35), (40, 35, 30),
    (25, 30, 25), (35, 25, 35), (20, 20, 30),
    (45, 40, 35), (30, 35, 45), (35, 30, 25),
])


# ═══════════════════════════════════════════════════════════
# Background generator functions
# ═══════════════════════════════════════════════════════════

def bg_solid(img: Image.Image, config: BackgroundConfig) -> Image.Image:
    """Fill with a single solid color."""
    palette = config.bg_colors or DEFAULT_BG_COLORS
    color = palette.sample()
    return Image.new('RGB', img.size, color)


def bg_gradient(img: Image.Image, config: BackgroundConfig) -> Image.Image:
    """Linear or radial gradient background (vectorized)."""
    w, h = img.size
    palette = config.bg_colors or DEFAULT_BG_COLORS
    c1 = np.array(palette.sample(), dtype=np.float32)
    c2 = np.array(palette.sample(), dtype=np.float32)

    direction = random.choice(config.gradient_directions)

    if direction == "horizontal":
        t = np.linspace(0, 1, w, dtype=np.float32)[np.newaxis, :]
        t = np.broadcast_to(t, (h, w))
    elif direction == "vertical":
        t = np.linspace(0, 1, h, dtype=np.float32)[:, np.newaxis]
        t = np.broadcast_to(t, (h, w))
    elif direction == "diagonal":
        tx = np.linspace(0, 1, w, dtype=np.float32)
        ty = np.linspace(0, 1, h, dtype=np.float32)
        t = (tx[np.newaxis, :] + ty[:, np.newaxis]) / 2
    else:
        # Radial
        cx, cy = w / 2, h / 2
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        t = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        t = np.clip(t / (t.max() + 1e-6), 0, 1)

    # Blend colors
    arr = (c1[np.newaxis, np.newaxis, :] * (1 - t[:, :, np.newaxis]) +
           c2[np.newaxis, np.newaxis, :] * t[:, :, np.newaxis])
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def bg_noise_speckle(img: Image.Image, config: BackgroundConfig) -> Image.Image:
    """Salt-and-pepper or Gaussian noise background."""
    w, h = img.size
    palette = config.bg_colors or DEFAULT_BG_COLORS
    base_color = palette.sample()

    arr = np.full((h, w, 3), base_color, dtype=np.uint8)

    density = random.uniform(*config.speckle_density)
    n_pixels = int(density * w * h)

    # Random speckle
    if random.random() < 0.5:
        # Gaussian noise
        noise = np.random.normal(0, random.uniform(15, 40), (h, w, 3))
        arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    else:
        # Salt-and-pepper (vectorized)
        n_pixels = min(n_pixels, w * h)
        ys = np.random.randint(0, h, n_pixels)
        xs = np.random.randint(0, w, n_pixels)
        vals = np.where(
            np.random.random(n_pixels)[:, np.newaxis] < 0.5,
            np.random.randint(200, 256, (n_pixels, 3)),
            np.random.randint(0, 81, (n_pixels, 3)),
        ).astype(np.uint8)
        arr[ys, xs] = vals

    return Image.fromarray(arr)


def bg_grid(img: Image.Image, config: BackgroundConfig) -> Image.Image:
    """Axis-aligned grid lines (inspired by CryptBB grid background)."""
    w, h = img.size
    palette = config.bg_colors or DEFAULT_BG_COLORS
    base_color = palette.sample()
    result = Image.new('RGB', (w, h), base_color)
    draw = ImageDraw.Draw(result)

    grid_palette = config.grid_color_palette or FAINT_LINE_PALETTE
    spacing_min, spacing_max = config.grid_spacing_range
    lw = config.grid_line_width

    # Vertical lines
    num_v = w // random.randint(spacing_min, spacing_max)
    spacing = w / max(num_v, 1)
    for i in range(num_v + 1):
        x = int(i * spacing + random.gauss(0, spacing * 0.15))
        x = max(0, min(w - 1, x))
        draw.line([(x, 0), (x, h)], fill=grid_palette.sample(), width=lw)

    # Horizontal lines
    num_h = h // random.randint(spacing_min, max(spacing_min + 1, spacing_max // 2))
    spacing_h = h / max(num_h, 1)
    for i in range(num_h + 1):
        y = int(i * spacing_h + random.gauss(0, spacing_h * 0.15))
        y = max(0, min(h - 1, y))
        draw.line([(0, y), (w, y)], fill=grid_palette.sample(), width=lw)

    # Occasional diagonal accent lines
    if random.random() < 0.3:
        for _ in range(random.randint(1, 3)):
            x1, y1 = random.randint(0, w), random.randint(0, h)
            x2, y2 = random.randint(0, w), random.randint(0, h)
            draw.line([(x1, y1), (x2, y2)], fill=grid_palette.sample(), width=1)

    return result


def bg_tiled_patches(img: Image.Image, config: BackgroundConfig) -> Image.Image:
    """Overlapping colored rectangles with stipple dots (CryptBB patch style)."""
    w, h = img.size
    palette = config.bg_colors or DEFAULT_BG_COLORS
    base_color = palette.sample()

    # Work entirely with numpy array for speed
    result_arr = np.full((h, w, 4), list(base_color) + [255], dtype=np.uint8)

    patch_palette = PASTEL_PALETTE
    n_patches = random.randint(*config.patch_count_range)
    sz_min, sz_max = config.patch_size_range
    op_min, op_max = config.patch_opacity_range

    for _ in range(n_patches):
        pw = random.randint(sz_min, sz_max)
        ph = random.randint(sz_min, max(sz_min + 1, int(sz_max * 0.7)))
        px = random.randint(-15, w - 15)
        py_ = random.randint(-10, h - 10)

        color = patch_palette.sample()
        alpha = random.randint(op_min, op_max)

        # Clamp to image bounds
        x1 = max(0, px)
        y1 = max(0, py_)
        x2 = min(w, px + pw)
        y2 = min(h, py_ + ph)
        if x2 <= x1 or y2 <= y1:
            continue

        # Alpha-blend patch rectangle onto result
        patch_color = np.array(list(color) + [alpha], dtype=np.float32)
        region = result_arr[y1:y2, x1:x2].astype(np.float32)
        a = alpha / 255.0
        region[:, :, :3] = region[:, :, :3] * (1 - a) + np.array(color, dtype=np.float32) * a
        result_arr[y1:y2, x1:x2] = np.clip(region, 0, 255).astype(np.uint8)

        # Stipple/halftone pattern — apply as grid mask
        if random.random() < 0.8:
            spacing = random.choice([3, 4, 5])
            ox = random.randint(0, spacing)
            oy = random.randint(0, spacing)
            # Create stipple mask for this patch region
            rh, rw = y2 - y1, x2 - x1
            ys_local = np.arange(rh)
            xs_local = np.arange(rw)
            y_match = ((ys_local + y1 + oy) % spacing == 0)
            x_match = ((xs_local + x1 + ox) % spacing == 0)
            # Outer product gives 2D mask
            mask = y_match[:, np.newaxis] & x_match[np.newaxis, :]
            # Apply white stipple dots
            stipple_alpha = random.uniform(0.5, 0.9)
            region = result_arr[y1:y2, x1:x2].astype(np.float32)
            region[mask, :3] = region[mask, :3] * (1 - stipple_alpha) + 255 * stipple_alpha
            result_arr[y1:y2, x1:x2] = np.clip(region, 0, 255).astype(np.uint8)

        # Subtle border
        if random.random() < 0.3:
            edge_c = [max(0, c - 40) for c in color] + [alpha]
            result_arr[y1, x1:x2] = edge_c
            result_arr[y2-1, x1:x2] = edge_c
            result_arr[y1:y2, x1] = edge_c
            result_arr[y1:y2, x2-1] = edge_c

    return Image.fromarray(result_arr[:, :, :3])


def bg_dense_lines(img: Image.Image, config: BackgroundConfig) -> Image.Image:
    """Dense random lines and arcs (CryptBB line style)."""
    w, h = img.size
    palette = config.bg_colors or DEFAULT_BG_COLORS
    base_color = palette.sample()
    result = Image.new('RGB', (w, h), base_color)
    draw = ImageDraw.Draw(result)

    line_palette = config.line_color_palette or FAINT_LINE_PALETTE
    n_lines = random.randint(*config.line_count_range)

    # Straight lines
    for _ in range(n_lines):
        x1 = random.randint(-40, w + 40)
        y1 = random.randint(-40, h + 40)
        length = random.randint(15, 120)
        angle = random.uniform(0, 2 * math.pi)
        x2 = int(x1 + length * math.cos(angle))
        y2 = int(y1 + length * math.sin(angle))
        draw.line([(x1, y1), (x2, y2)], fill=line_palette.sample(), width=1)

    # Arcs
    n_arcs = random.randint(n_lines // 3, n_lines // 2)
    for _ in range(n_arcs):
        cx, cy = random.randint(-20, w + 20), random.randint(-20, h + 20)
        r = random.randint(15, 100)
        start = random.randint(0, 360)
        end = start + random.randint(40, 200)
        draw.arc([(cx - r, cy - r), (cx + r, cy + r)],
                 start, end, fill=line_palette.sample(), width=1)

    return result


def bg_texture_fill(img: Image.Image, config: BackgroundConfig) -> Image.Image:
    """Procedural noise texture using Perlin-like patterns (vectorized)."""
    w, h = img.size
    palette = config.bg_colors or DEFAULT_BG_COLORS
    c1 = np.array(palette.sample(), dtype=np.float32)
    c2 = np.array(palette.sample(), dtype=np.float32)

    scale = random.uniform(*config.texture_scale)
    xs = np.arange(w, dtype=np.float32)[np.newaxis, :]
    ys = np.arange(h, dtype=np.float32)[:, np.newaxis]

    arr = np.zeros((h, w), dtype=np.float32)
    for octave in range(3):
        freq = scale * (2 ** octave)
        amp = 1.0 / (1.5 ** octave)
        px = random.uniform(0, 100)
        py = random.uniform(0, 100)
        arr += amp * (
            np.sin(xs * freq + px) * np.cos(ys * freq + py) +
            np.sin((xs + ys) * freq * 0.7 + px * 1.3)
        ) / 2

    # Normalize to [0, 1]
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr = np.full_like(arr, 0.5)

    # Map to color gradient
    result = (c1[np.newaxis, np.newaxis, :] * (1 - arr[:, :, np.newaxis]) +
              c2[np.newaxis, np.newaxis, :] * arr[:, :, np.newaxis])
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def bg_image_bg(img: Image.Image, config: BackgroundConfig) -> Image.Image:
    """Generate a pseudo-photographic background using filtered noise."""
    w, h = img.size
    palette = config.bg_colors or DEFAULT_BG_COLORS

    # Start with colored noise
    arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    result = Image.fromarray(arr)

    # Heavy blur to make it look like an out-of-focus photo
    result = result.filter(ImageFilter.GaussianBlur(radius=random.uniform(8, 20)))

    # Tint toward a base color
    base = palette.sample()
    arr = np.array(result).astype(np.float32)
    base_arr = np.array(base, dtype=np.float32)
    blend = random.uniform(0.4, 0.7)
    arr = arr * (1 - blend) + base_arr * blend
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


# ═══════════════════════════════════════════════════════════
# Dispatcher
# ═══════════════════════════════════════════════════════════

_BG_GENERATORS = {
    BackgroundType.SOLID: bg_solid,
    BackgroundType.GRADIENT: bg_gradient,
    BackgroundType.NOISE_SPECKLE: bg_noise_speckle,
    BackgroundType.GRID: bg_grid,
    BackgroundType.TILED_PATCHES: bg_tiled_patches,
    BackgroundType.DENSE_LINES: bg_dense_lines,
    BackgroundType.TEXTURE_FILL: bg_texture_fill,
    BackgroundType.IMAGE_BG: bg_image_bg,
}


def generate_background(
    width: int,
    height: int,
    config: BackgroundConfig,
) -> Image.Image:
    """Generate a background image using the config's specified types."""
    bg_type = random.choice(config.types)
    dummy = Image.new('RGB', (width, height), (255, 255, 255))
    generator = _BG_GENERATORS.get(bg_type, bg_solid)
    return generator(dummy, config)
