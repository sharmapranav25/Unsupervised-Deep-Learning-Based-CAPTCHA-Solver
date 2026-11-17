"""
Noise Overlay Layers for Universal CAPTCHA Generator.

Applied after text rendering to add adversarial visual noise:
crossing lines, arcs, sine waves, dots, geometric shapes, stripe bands.
"""

import math
import random
from typing import Tuple

from PIL import Image, ImageDraw

from .config import NoiseConfig, NoiseOverlay, ColorPalette


# ═══════════════════════════════════════════════════════════
# Default noise palettes
# ═══════════════════════════════════════════════════════════

DEFAULT_NOISE_LINE_COLORS = ColorPalette(colors=[
    (100, 100, 100), (150, 150, 150), (80, 80, 80),
    (120, 80, 80), (80, 120, 80), (80, 80, 120),
    (160, 100, 100), (100, 160, 100), (100, 100, 160),
    (140, 140, 140), (60, 60, 60), (180, 180, 180),
])

COLORED_NOISE_PALETTE = ColorPalette(colors=[
    (200, 50, 50), (50, 200, 50), (50, 50, 200),
    (200, 200, 50), (200, 50, 200), (50, 200, 200),
    (150, 100, 50), (100, 150, 50), (50, 100, 150),
])


# ═══════════════════════════════════════════════════════════
# Noise generators
# ═══════════════════════════════════════════════════════════

def noise_crossing_lines(img: Image.Image, config: NoiseConfig) -> Image.Image:
    """Random straight lines crossing through the image."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    palette = config.line_color_palette or DEFAULT_NOISE_LINE_COLORS
    # Cap at 5 lines to avoid obscuring text
    n_lines = min(random.randint(*config.line_count_range), 5)

    for _ in range(n_lines):
        x1 = random.randint(-10, w + 10)
        y1 = random.randint(-10, h + 10)
        x2 = random.randint(-10, w + 10)
        y2 = random.randint(-10, h + 10)
        # Keep lines thin (max width 1) so text shows through
        lw = 1
        draw.line([(x1, y1), (x2, y2)], fill=palette.sample(), width=lw)

    return img


def noise_random_arcs(img: Image.Image, config: NoiseConfig) -> Image.Image:
    """Curved arcs at various positions and radii."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    palette = config.line_color_palette or DEFAULT_NOISE_LINE_COLORS
    # Cap at 4 arcs
    n_arcs = min(random.randint(*config.arc_count_range), 4)

    for _ in range(n_arcs):
        cx = random.randint(-20, w + 20)
        cy = random.randint(-20, h + 20)
        r = random.randint(10, max(w, h) // 2)
        start = random.randint(0, 360)
        end = start + random.randint(30, 270)
        lw = 1  # thin arcs so text shows through
        draw.arc([(cx - r, cy - r), (cx + r, cy + r)],
                 start, end, fill=palette.sample(), width=lw)

    return img


def noise_sine_waves(img: Image.Image, config: NoiseConfig) -> Image.Image:
    """Sinusoidal curves drawn across the image."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    palette = config.line_color_palette or DEFAULT_NOISE_LINE_COLORS
    # Cap at 2 waves
    n_waves = min(random.randint(*config.wave_count_range), 2)

    for _ in range(n_waves):
        # Reduced amplitude so waves don't cover too much vertical space
        amplitude = random.uniform(3, h * 0.15)
        frequency = random.uniform(0.01, 0.04)
        phase = random.uniform(0, 2 * math.pi)
        y_base = random.randint(int(h * 0.2), int(h * 0.8))
        color = palette.sample()
        lw = 1  # thin lines

        points = []
        for x in range(0, w, 2):
            y = int(y_base + amplitude * math.sin(frequency * x + phase))
            points.append((x, y))

        if len(points) >= 2:
            draw.line(points, fill=color, width=lw)

    return img


def noise_dot_scatter(img: Image.Image, config: NoiseConfig) -> Image.Image:
    """Random dots/circles of varying sizes scattered across the image."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    palette = config.line_color_palette or COLORED_NOISE_PALETTE
    n_dots = random.randint(*config.dot_count_range)

    for _ in range(n_dots):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        r = random.randint(*config.dot_radius_range)
        color = palette.sample()

        if r <= 1:
            draw.point((x, y), fill=color)
        else:
            draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=color)

    return img


def noise_geometric_shapes(img: Image.Image, config: NoiseConfig) -> Image.Image:
    """Rectangles, triangles, and ellipses overlaid on the image."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    palette = config.line_color_palette or COLORED_NOISE_PALETTE
    n_shapes = random.randint(*config.shape_count_range)

    for _ in range(n_shapes):
        shape = random.choice(['rect', 'ellipse', 'triangle'])
        color = palette.sample()
        lw = random.randint(1, 2)

        if shape == 'rect':
            x1 = random.randint(0, w - 10)
            y1 = random.randint(0, h - 10)
            x2 = x1 + random.randint(5, 40)
            y2 = y1 + random.randint(5, 30)
            if random.random() < 0.5:
                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=lw)
            else:
                draw.rectangle([(x1, y1), (x2, y2)], fill=color + (60,) if img.mode == 'RGBA' else color)

        elif shape == 'ellipse':
            cx = random.randint(0, w)
            cy = random.randint(0, h)
            rx = random.randint(3, 25)
            ry = random.randint(3, 25)
            draw.ellipse([(cx - rx, cy - ry), (cx + rx, cy + ry)], outline=color, width=lw)

        elif shape == 'triangle':
            points = [
                (random.randint(0, w), random.randint(0, h)),
                (random.randint(0, w), random.randint(0, h)),
                (random.randint(0, w), random.randint(0, h)),
            ]
            draw.polygon(points, outline=color)

    return img


def noise_stripe_bands(img: Image.Image, config: NoiseConfig) -> Image.Image:
    """Semi-transparent horizontal or vertical bands."""
    w, h = img.size
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    palette = config.line_color_palette or COLORED_NOISE_PALETTE
    n_bands = random.randint(*config.band_count_range)

    for _ in range(n_bands):
        color = palette.sample()
        opacity = random.randint(*config.band_opacity_range)
        fill = color + (opacity,)

        if random.random() < 0.5:
            # Horizontal band
            y = random.randint(0, h - 5)
            bh = random.randint(3, max(4, h // 6))
            draw.rectangle([(0, y), (w, y + bh)], fill=fill)
        else:
            # Vertical band
            x = random.randint(0, w - 5)
            bw = random.randint(3, max(4, w // 8))
            draw.rectangle([(x, 0), (x + bw, h)], fill=fill)

    # Composite
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    result = Image.alpha_composite(img, overlay)
    return result.convert('RGB')


def noise_stroke_matched_grid(img: Image.Image, config: NoiseConfig) -> Image.Image:
    """Background grid where line thickness approximates font stroke width.
    
    Rendered BEHIND text to camouflage without occluding.
    Semi-transparent so text remains readable.
    """
    w, h = img.size
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    palette = config.line_color_palette or DEFAULT_NOISE_LINE_COLORS
    color = palette.sample()
    
    # Estimate stroke width: 1-2px — thin enough to not dominate
    stroke_width = random.randint(1, 2)
    
    # Semi-transparent so text on top is always readable
    opacity = random.randint(60, 120)
    fill = color + (opacity,)
    
    # Grid spacing
    spacing = random.randint(max(12, h//5), max(25, h//3))
    
    # Draw vertical lines
    for x in range(0, w, spacing):
        draw.line([(x, 0), (x, h)], fill=fill, width=stroke_width)
        
    # Draw horizontal lines
    for y in range(0, h, spacing):
        draw.line([(0, y), (w, y)], fill=fill, width=stroke_width)
    
    # Composite
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    result = Image.alpha_composite(img, overlay)
    return result.convert('RGB')


# ═══════════════════════════════════════════════════════════
# Dispatcher
# ═══════════════════════════════════════════════════════════

_NOISE_GENERATORS = {
    NoiseOverlay.CROSSING_LINES: noise_crossing_lines,
    NoiseOverlay.RANDOM_ARCS: noise_random_arcs,
    NoiseOverlay.SINE_WAVES: noise_sine_waves,
    NoiseOverlay.DOT_SCATTER: noise_dot_scatter,
    NoiseOverlay.GEOMETRIC_SHAPES: noise_geometric_shapes,
    NoiseOverlay.STRIPE_BANDS: noise_stripe_bands,
    NoiseOverlay.STROKE_MATCHED_GRID: noise_stroke_matched_grid,
}


def apply_noise_layers(
    image: Image.Image,
    config: NoiseConfig,
    activation_prob: float = 0.5,
) -> Image.Image:
    """
    Apply noise overlay layers to the image.
    Each configured overlay is independently activated with activation_prob.
    """
    for overlay in config.overlays:
        if overlay == NoiseOverlay.NONE:
            continue
        if random.random() > activation_prob:
            continue

        generator = _NOISE_GENERATORS.get(overlay)
        if generator:
            image = generator(image, config)

    return image


# Background noise: rendered BEFORE text (fills/textures that text sits on top of)
_BACKGROUND_NOISE_TYPES = {
    NoiseOverlay.DOT_SCATTER,
    NoiseOverlay.GEOMETRIC_SHAPES,
    NoiseOverlay.STRIPE_BANDS,
    NoiseOverlay.STROKE_MATCHED_GRID,  # grid goes BEHIND text
}

# Foreground noise: rendered AFTER text (lines that cross through characters)
_FOREGROUND_NOISE_TYPES = {
    NoiseOverlay.CROSSING_LINES,
    NoiseOverlay.RANDOM_ARCS,
    NoiseOverlay.SINE_WAVES,
}


def apply_background_noise(
    image: Image.Image,
    config: NoiseConfig,
    activation_prob: float = 0.5,
) -> Image.Image:
    """
    Apply background noise layers BEFORE text rendering.
    These are dots, shapes, and bands that text will be rendered on top of.
    """
    for overlay in config.overlays:
        if overlay == NoiseOverlay.NONE:
            continue
        if overlay not in _BACKGROUND_NOISE_TYPES:
            continue
        if random.random() > activation_prob:
            continue

        generator = _NOISE_GENERATORS.get(overlay)
        if generator:
            image = generator(image, config)

    return image


def apply_foreground_noise(
    image: Image.Image,
    config: NoiseConfig,
    activation_prob: float = 0.5,
) -> Image.Image:
    """
    Apply foreground noise layers AFTER text rendering.
    These are crossing lines, arcs, and waves that pass through characters.
    """
    for overlay in config.overlays:
        if overlay == NoiseOverlay.NONE:
            continue
        if overlay not in _FOREGROUND_NOISE_TYPES:
            continue
        if random.random() > activation_prob:
            continue

        generator = _NOISE_GENERATORS.get(overlay)
        if generator:
            image = generator(image, config)

    return image
