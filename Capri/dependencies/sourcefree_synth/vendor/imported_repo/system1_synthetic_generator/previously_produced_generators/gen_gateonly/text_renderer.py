"""
Text Renderer for Universal CAPTCHA Generator.

Renders individual characters with per-character styling (font, size, color,
rotation, shadow, outline, halo) and composites them onto the background image.
"""

import random
import math
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from .config import TextConfig, DistortionConfig, TextDistortion, FontCategory, CaptchaSchema, ColorPalette
from .font_manager import FontManager


# ═══════════════════════════════════════════════════════════
# Default text color palettes
# ═══════════════════════════════════════════════════════════

DEFAULT_TEXT_COLORS = ColorPalette(colors=[
    (180, 30, 30), (200, 50, 50), (160, 20, 20),      # reds
    (30, 140, 30), (40, 160, 60), (0, 120, 0),         # greens
    (30, 30, 180), (60, 60, 200), (0, 80, 170),        # blues
    (140, 30, 140), (100, 0, 140), (130, 50, 130),     # purples
    (0, 140, 140), (0, 150, 120),                       # teals
    (180, 100, 0), (140, 80, 20), (160, 110, 30),      # browns/oranges
    (200, 0, 100), (180, 40, 100),                      # pinks
    (80, 80, 80), (50, 50, 50), (10, 10, 10),          # darks
    (220, 120, 0), (0, 100, 60), (120, 0, 80),         # accents
])

BRIGHT_TEXT_COLORS = ColorPalette(colors=[
    (255, 50, 50), (50, 255, 50), (50, 50, 255),
    (255, 255, 50), (255, 50, 255), (50, 255, 255),
    (255, 128, 0), (128, 255, 0), (255, 0, 128),
])


# ═══════════════════════════════════════════════════════════
# Character renderer
# ═══════════════════════════════════════════════════════════

def render_single_char(
    char: str,
    font: ImageFont.FreeTypeFont,
    color: Tuple[int, int, int],
    text_config: TextConfig,
    is_hollow: bool = False,
    max_size: int = 200,
) -> Image.Image:
    """
    Render a single character as an RGBA image with optional halo, shadow, outline.

    Returns: RGBA Image of the rendered character, cropped to content.
    """
    # Measure character
    test_img = Image.new('RGBA', (max_size, max_size), (0, 0, 0, 0))
    test_draw = ImageDraw.Draw(test_img)
    bbox = test_draw.textbbox((0, 0), char, font=font)
    cw = max(bbox[2] - bbox[0] + 20, 10)
    ch = max(bbox[3] - bbox[1] + 20, 10)

    canvas = Image.new('RGBA', (cw + 30, ch + 30), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    ox = 15 - bbox[0]
    oy = 15 - bbox[1]

    # 1. White halo/knockout (background occlusion)
    if text_config.halo_enabled:
        halo_color = text_config.halo_color + (255,)
        r = text_config.halo_radius
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    draw.text((ox + dx, oy + dy), char, fill=halo_color, font=font)

    # 2. Shadow
    if text_config.shadow_enabled:
        sx, sy = text_config.shadow_offset
        shadow_color = tuple(max(0, c - 60) for c in color) + (text_config.shadow_opacity,)
        draw.text((ox + sx, oy + sy), char, fill=shadow_color, font=font)

    # 3. Outline/stroke
    if text_config.outline_enabled:
        oc = text_config.outline_color
        if oc is None:
            oc = tuple(max(0, c - 40) for c in color)
        outline_fill = oc + (180,)
        w = text_config.outline_width
        for dx in range(-w, w + 1):
            for dy in range(-w, w + 1):
                if abs(dx) + abs(dy) > 0:
                    draw.text((ox + dx, oy + dy), char, fill=outline_fill, font=font)

    # 4. Main character (skip if hollow)
    if not is_hollow:
        draw.text((ox, oy), char, fill=color + (255,), font=font)

    # Crop to content
    arr = np.array(canvas)
    alpha = arr[:, :, 3]
    rows = np.any(alpha > 5, axis=1)
    cols = np.any(alpha > 5, axis=0)
    if not rows.any() or not cols.any():
        return None  # empty render = tofu

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    pad = 2
    rmin = max(0, rmin - pad)
    rmax = min(arr.shape[0] - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(arr.shape[1] - 1, cmax + pad)

    cropped = canvas.crop((cmin, rmin, cmax + 1, rmax + 1))
    
    # Runtime tofu detection: check if font actually has this glyph
    # GlyphID 0 = .notdef / tofu glyph in TrueType/OpenType fonts
    if not is_hollow:
        try:
            glyph_id = font.font.getGlyphID(char)
            if glyph_id == 0:
                return None  # font doesn't have this character
        except (AttributeError, TypeError):
            pass  # older PIL versions may not support this
    
    return cropped


def _apply_wave_warp(img: Image.Image, amplitude: float, frequency: float) -> Image.Image:
    """Apply a sine wave warp to an image (vectorized)."""
    arr = np.array(img)
    h, w = arr.shape[:2]

    # Compute per-row shifts
    ys = np.arange(h)
    shifts = (amplitude * np.sin(2 * np.pi * frequency * ys)).astype(np.int32)

    # Build source x-coordinate map
    xs = np.arange(w)
    src_x = xs[np.newaxis, :] - shifts[:, np.newaxis]  # (h, w)

    # Mask valid coordinates
    valid = (src_x >= 0) & (src_x < w)
    src_x_clamped = np.clip(src_x, 0, w - 1)

    # Gather pixels
    row_idx = np.arange(h)[:, np.newaxis].repeat(w, axis=1)
    result = arr[row_idx, src_x_clamped]
    result[~valid] = 0

    return Image.fromarray(result)


def _apply_elastic_deform(img: Image.Image, alpha: float, sigma: float) -> Image.Image:
    """Apply elastic deformation to an image (vectorized)."""
    arr = np.array(img)
    h, w = arr.shape[:2]

    # Random displacement fields
    dx = np.random.uniform(-1, 1, (h, w)).astype(np.float32)
    dy = np.random.uniform(-1, 1, (h, w)).astype(np.float32)

    # Smooth via PIL Gaussian blur
    for i, field in enumerate([dx, dy]):
        f_img = Image.fromarray(((field + 1) * 127.5).astype(np.uint8))
        f_img = f_img.filter(ImageFilter.GaussianBlur(radius=sigma))
        smoothed = (np.array(f_img).astype(np.float32) / 127.5 - 1.0) * alpha
        if i == 0:
            dx = smoothed
        else:
            dy = smoothed

    # Build coordinate grids
    ys, xs = np.mgrid[0:h, 0:w]
    src_x = (xs + dx).astype(np.int32)
    src_y = (ys + dy).astype(np.int32)

    # Mask valid coordinates
    valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)
    src_x_clamped = np.clip(src_x, 0, w - 1)
    src_y_clamped = np.clip(src_y, 0, h - 1)

    # Gather pixels
    result = arr[src_y_clamped, src_x_clamped]
    result[~valid] = 0

    return Image.fromarray(result)


def _apply_perspective(img: Image.Image, strength: float) -> Image.Image:
    """Apply a random perspective transform."""
    w, h = img.size
    # Random perspective coefficients
    coeffs = [
        1 + random.uniform(-strength, strength),
        random.uniform(-strength, strength),
        0,
        random.uniform(-strength, strength),
        1 + random.uniform(-strength, strength),
        0,
        random.uniform(-strength * 0.01, strength * 0.01),
        random.uniform(-strength * 0.01, strength * 0.01),
    ]
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC,
                          fillcolor=(0, 0, 0, 0) if img.mode == 'RGBA' else (255, 255, 255))


def _apply_shear(img: Image.Image, shear: float) -> Image.Image:
    """Apply shear transform."""
    w, h = img.size
    coeffs = [1, shear, -shear * h / 2, 0, 1, 0]
    return img.transform((w, h), Image.AFFINE, coeffs, Image.BICUBIC,
                          fillcolor=(0, 0, 0, 0) if img.mode == 'RGBA' else (255, 255, 255))


# ═══════════════════════════════════════════════════════════
# Full text rendering pipeline
# ═══════════════════════════════════════════════════════════

def render_text_on_image(
    image: Image.Image,
    text: str,
    schema: CaptchaSchema,
    font_manager: FontManager,
    return_meta: bool = False,
) -> Image.Image:
    """
    Render text onto the background image using the schema's configuration.

    This is the main text rendering function that handles:
    - Per-character font/size/color selection
    - Character-level distortions (rotation, scale jitter, vertical offset)
    - Layout (spacing, overlap)
    - Compositing onto the background
    - Whole-image distortions (wave warp, elastic, perspective, shear)

    Returns: The image with text rendered on it.
    """
    w, h = image.size
    text_config = schema.text
    dist_config = schema.distortion

    # Get color palette
    palette = text_config.text_colors or DEFAULT_TEXT_COLORS

    # Prepare per-character parameters
    char_images = []
    for i, char in enumerate(text):
        # Font selection
        cat = random.choice(text_config.font_categories)
        size = random.randint(*text_config.font_size_range)

        # Scale jitter per character
        if TextDistortion.SCALE_JITTER in dist_config.distortions and random.random() < schema.distortion_prob:
            scale = random.uniform(*dist_config.scale_jitter)
            size = int(size * scale)
            size = max(28, min(size, 72))  # min 28px so chars stay readable, max 72 to limit variation

        font = font_manager.get_font(category=cat, size=size)

        # Color
        if text_config.per_char_color:
            color = palette.sample()
        else:
            # Use same color for all (sample once)
            if i == 0:
                shared_color = palette.sample()
            color = shared_color  # noqa

        # Hollow font check
        is_hollow = TextDistortion.HOLLOW_FONT in dist_config.distortions and random.random() < schema.distortion_prob
        # If hollow, force a thick outline so the character shape is always visible
        if is_hollow:
            text_config.outline_enabled = True
            text_config.outline_width = max(2, size // 12)  # thicker outline for visibility
            text_config.outline_color = color

        # Render character — with tofu fallback
        char_img = render_single_char(char, font, color, text_config, is_hollow=is_hollow)
        
        # If tofu detected, retry with fallback fonts
        if char_img is None:
            for _fallback_attempt in range(3):
                fb_font = font_manager.get_font(category=cat, size=size)
                char_img = render_single_char(char, fb_font, color, text_config, is_hollow=is_hollow)
                if char_img is not None:
                    break
            # Last resort: DejaVuSans is always safe
            if char_img is None:
                try:
                    safe_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)
                except:
                    safe_font = ImageFont.load_default()
                char_img = render_single_char(char, safe_font, color, text_config, is_hollow=False)
                if char_img is None:
                    # Absolute last resort: render a simple colored rectangle
                    char_img = Image.new('RGBA', (size, int(size*1.2)), color + (255,))

        # Per-character rotation
        if TextDistortion.ROTATION in dist_config.distortions and random.random() < schema.distortion_prob:
            angle = random.uniform(*dist_config.rotation_range)
            char_img = char_img.rotate(angle, expand=True, resample=Image.BICUBIC)

        char_images.append(char_img)

    # ── Layout ──

    # Calculate spacing
    total_char_width = sum(ci.width for ci in char_images)
    spacing = random.randint(*text_config.char_spacing_range)

    # Handle overlap and extreme ligature kerning
    if TextDistortion.LIGATURE_KERNING in dist_config.distortions and random.random() < schema.distortion_prob:
        # Extreme negative kerning to blend character strokes
        # Based on average char width
        avg_w = total_char_width / max(1, len(char_images))
        spacing = -int(random.uniform(0.3, 0.6) * avg_w)
    elif TextDistortion.OVERLAP in dist_config.distortions and random.random() < schema.distortion_prob:
        overlap = random.randint(*dist_config.overlap_pixels)
        spacing = max(-overlap, spacing - overlap)

    total_width = total_char_width + spacing * (len(char_images) - 1)

    # Start position — center text horizontally
    start_x = max(2, (w - total_width) // 2)
    if start_x + total_width > w:
        # Shrink spacing if text is too wide
        if len(char_images) > 1:
            spacing = max(-3, (w - total_char_width - 10) // (len(char_images) - 1))
            start_x = 5
        total_width = total_char_width + spacing * (len(char_images) - 1)

    # If STILL too wide after spacing reduction, scale down all chars
    if total_width > w - 4:
        scale_factor = (w - 10) / max(total_width, 1)
        scale_factor = max(0.55, scale_factor)  # don't go below 55% — chars become unreadable blobs
        new_char_images = []
        for ci in char_images:
            new_w = max(4, int(ci.width * scale_factor))
            new_h = max(8, int(ci.height * scale_factor))
            new_char_images.append(ci.resize((new_w, new_h), Image.LANCZOS))
        char_images = new_char_images
        total_char_width = sum(ci.width for ci in char_images)
        spacing = max(1, (w - total_char_width - 10) // max(1, len(char_images) - 1))
        start_x = max(2, (w - total_char_width - spacing * (len(char_images) - 1)) // 2)

    # Composite each character
    result = image.convert('RGBA')
    current_x = start_x
    char_meta = []

    # Prepare curved path if active
    is_curved = TextDistortion.CURVED_PATH in dist_config.distortions and random.random() < schema.distortion_prob
    if is_curved:
        path_amplitude = random.uniform(5, h * 0.15)  # cap amplitude to prevent chars going offscreen
        path_frequency = random.uniform(0.5, 1.2) / max(1, len(char_images))
        path_phase = random.uniform(0, math.pi * 2)

    # Prepare adversarial blend if active
    adv_blend = TextDistortion.ADVERSARIAL_BLEND in dist_config.distortions and random.random() < schema.distortion_prob * 0.5
    blend_mode = random.choice(['multiply', 'multiply', 'difference']) if adv_blend else 'normal'  # bias toward multiply (safer)
    if adv_blend:
        from PIL import ImageChops

    for i, char_img in enumerate(char_images):
        # Vertical position
        base_y = (h - char_img.height) // 2

        # Curved path adjustment
        if is_curved:
            progress = i / max(1, len(char_images) - 1)
            base_y += int(path_amplitude * math.sin(progress * math.pi * 2 * path_frequency + path_phase))

        # Per-character vertical offset
        if TextDistortion.VERTICAL_OFFSET in dist_config.distortions and random.random() < (schema.distortion_prob if not is_curved else 0.2):
            v_offset = random.randint(*dist_config.vertical_offset_range)
            base_y += v_offset

        paste_x = int(current_x)
        paste_y = int(base_y)

        # Clamp to bounds
        paste_x = max(0, min(paste_x, w - char_img.width))
        paste_y = max(0, min(paste_y, h - char_img.height))

        # --- degeneracy meta: per-char box + glyph/background luminance contrast ---
        if return_meta:
            ga = np.array(char_img)
            amask = ga[:, :, 3] > 128 if (ga.ndim == 3 and ga.shape[2] == 4) else np.ones(ga.shape[:2], bool)
            bgc = np.array(result.crop((paste_x, paste_y, paste_x + char_img.width,
                                        paste_y + char_img.height)).convert('RGB'), dtype=np.float32)
            _lum = lambda a: 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
            alpha = ga[:, :, 3].astype(np.float32) / 255.0 if (ga.ndim == 3 and ga.shape[2] == 4) \
                else np.ones(ga.shape[:2], np.float32)
            asum = float(alpha.sum())
            if asum > 1.0 and bgc.size:
                # alpha-WEIGHTED glyph luma (robust to thin anti-aliased strokes that an alpha>128
                # mask would miss) vs the PAPER (median bg, robust to grid lines / noise minority).
                gl = float((_lum(ga[:, :, :3].astype(np.float32)) * alpha).sum() / asum)
                bl = float(np.median(_lum(bgc)))
                contrast = abs(gl - bl)
            else:
                contrast = 255.0  # no measurable ink here — not a low-contrast case
            char_meta.append({"char": text[i] if i < len(text) else "",
                              "box": (int(paste_x), int(paste_y),
                                      int(paste_x + char_img.width), int(paste_y + char_img.height)),
                              "w": int(char_img.width), "h": int(char_img.height),
                              "contrast": contrast, "ink": float(amask.mean())})

        # Composite character
        if adv_blend:
            try:
                char_rgb = char_img.convert('RGB')
                char_mask = char_img.getchannel('A')
                crop = result.crop((paste_x, paste_y, paste_x + char_img.width, paste_y + char_img.height)).convert('RGB')
                if blend_mode == 'difference':
                    blended = ImageChops.difference(crop, char_rgb)
                else:
                    blended = ImageChops.multiply(crop, char_rgb)
                result.paste(blended, (paste_x, paste_y), char_mask)
            except Exception:
                result.paste(char_img, (paste_x, paste_y), char_img)
        else:
            result.paste(char_img, (paste_x, paste_y), char_img)
            
        current_x += char_img.width + spacing

    # ── Whole-image distortions ──

    # Wave warp — GUARDRAIL: cap amplitude to prevent zigzag destruction
    if TextDistortion.WAVE_WARP in dist_config.distortions and random.random() < schema.distortion_prob:
        # Cap amplitude at 3px regardless of config to keep text readable
        amp = min(3.0, random.uniform(*dist_config.wave_amplitude))
        freq = random.uniform(*dist_config.wave_frequency)
        result = _apply_wave_warp(result, amp, freq)

    # Elastic deformation — GUARDRAIL: cap alpha to prevent text destruction
    if TextDistortion.ELASTIC_DEFORM in dist_config.distortions and random.random() < schema.distortion_prob:
        alpha = min(25.0, random.uniform(*dist_config.elastic_alpha))
        sigma = random.uniform(*dist_config.elastic_sigma)
        result = _apply_elastic_deform(result, alpha, sigma)

    # Perspective
    if TextDistortion.PERSPECTIVE in dist_config.distortions and random.random() < schema.distortion_prob:
        strength = random.uniform(*dist_config.perspective_strength)
        result = _apply_perspective(result, strength)

    # Shear
    if TextDistortion.SHEAR in dist_config.distortions and random.random() < schema.distortion_prob:
        shear = random.uniform(*dist_config.shear_range)
        result = _apply_shear(result, shear)

    out = result.convert('RGB')
    if return_meta:
        return out, char_meta
    return out
