"""
Configuration system for Universal CAPTCHA Generator.

Defines all configurable features as enums and dataclasses,
plus the CaptchaSchema that fully describes a CAPTCHA generation recipe.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import random


# ═══════════════════════════════════════════════════════════
# Feature Enums
# ═══════════════════════════════════════════════════════════

class BackgroundType(Enum):
    SOLID = "solid"
    GRADIENT = "gradient"
    NOISE_SPECKLE = "noise_speckle"
    GRID = "grid"
    TILED_PATCHES = "tiled_patches"
    DENSE_LINES = "dense_lines"
    TEXTURE_FILL = "texture_fill"
    IMAGE_BG = "image_bg"


class TextDistortion(Enum):
    NONE = "none"
    ROTATION = "rotation"
    WAVE_WARP = "wave_warp"
    ELASTIC_DEFORM = "elastic_deform"
    PERSPECTIVE = "perspective"
    SHEAR = "shear"
    SCALE_JITTER = "scale_jitter"
    VERTICAL_OFFSET = "vertical_offset"
    OVERLAP = "overlap"
    KERNING_RANDOM = "kerning_random"
    STROKE_OUTLINE = "stroke_outline"
    SHADOW = "shadow"
    
    # Phase 7 Advanced Features
    HOLLOW_FONT = "hollow_font"
    LIGATURE_KERNING = "ligature_kerning"
    ADVERSARIAL_BLEND = "adversarial_blend"
    CURVED_PATH = "curved_path"
class NoiseOverlay(Enum):
    NONE = "none"
    CROSSING_LINES = "crossing_lines"
    RANDOM_ARCS = "random_arcs"
    SINE_WAVES = "sine_waves"
    DOT_SCATTER = "dot_scatter"
    GEOMETRIC_SHAPES = "geometric_shapes"
    STRIPE_BANDS = "stripe_bands"
    STROKE_MATCHED_GRID = "stroke_matched_grid"


class PostEffect(Enum):
    NONE = "none"
    GAUSSIAN_BLUR = "gaussian_blur"
    JPEG_ARTIFACTS = "jpeg_artifacts"
    COLOR_JITTER = "color_jitter"
    PIXELATION = "pixelation"
    EDGE_ENHANCE = "edge_enhance"
    INVERT = "invert"
    HALFTONE = "halftone"
    COMPRESSION_NOISE = "compression_noise"


class FontCategory(Enum):
    SANS_SERIF = "sans_serif"
    SERIF = "serif"
    MONOSPACE = "monospace"
    HANDWRITTEN = "handwritten"
    DISPLAY = "display"
    PIXEL = "pixel"
    SCRIPT = "script"
    ANY = "any"


class CharSet(Enum):
    LOWERCASE = "lowercase"
    UPPERCASE = "uppercase"
    DIGITS = "digits"
    ALPHA_LOWER = "alpha_lower"          # a-z
    ALPHA_UPPER = "alpha_upper"          # A-Z
    ALPHA_MIXED = "alpha_mixed"          # a-z + A-Z
    ALPHANUMERIC_LOWER = "alphanum_lower"  # a-z + 0-9
    ALPHANUMERIC_UPPER = "alphanum_upper"  # A-Z + 0-9
    ALPHANUMERIC_MIXED = "alphanum_mixed"  # a-z + A-Z + 0-9
    DIGITS_ONLY = "digits_only"            # 0-9


# ═══════════════════════════════════════════════════════════
# Schema Dataclass
# ═══════════════════════════════════════════════════════════

@dataclass
class ColorRange:
    """Defines a range of colors to sample from."""
    r_range: Tuple[int, int] = (0, 255)
    g_range: Tuple[int, int] = (0, 255)
    b_range: Tuple[int, int] = (0, 255)

    def sample(self) -> Tuple[int, int, int]:
        return (
            random.randint(*self.r_range),
            random.randint(*self.g_range),
            random.randint(*self.b_range),
        )


@dataclass
class ColorPalette:
    """A set of specific colors to choose from."""
    colors: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (0, 0, 0), (255, 0, 0), (0, 128, 0), (0, 0, 200),
        (128, 0, 128), (0, 128, 128), (200, 100, 0),
    ])

    def sample(self) -> Tuple[int, int, int]:
        return random.choice(self.colors)


@dataclass
class BackgroundConfig:
    """Configuration for background generation."""
    types: List[BackgroundType] = field(default_factory=lambda: [BackgroundType.SOLID])
    # For SOLID / GRADIENT
    bg_colors: Optional[ColorPalette] = None
    # For GRID
    grid_spacing_range: Tuple[int, int] = (10, 30)
    grid_line_width: int = 1
    grid_color_palette: Optional[ColorPalette] = None
    # For NOISE_SPECKLE
    speckle_density: Tuple[float, float] = (0.01, 0.1)
    # For TILED_PATCHES
    patch_count_range: Tuple[int, int] = (20, 50)
    patch_size_range: Tuple[int, int] = (20, 80)
    patch_opacity_range: Tuple[int, int] = (100, 220)
    # For DENSE_LINES
    line_count_range: Tuple[int, int] = (50, 200)
    line_color_palette: Optional[ColorPalette] = None
    # For TEXTURE_FILL
    texture_scale: Tuple[float, float] = (0.01, 0.1)
    # For gradients
    gradient_directions: List[str] = field(default_factory=lambda: ["horizontal", "vertical", "diagonal"])


@dataclass
class TextConfig:
    """Configuration for text rendering."""
    charset: CharSet = CharSet.ALPHANUMERIC_MIXED
    length_range: Tuple[int, int] = (4, 7)
    font_categories: List[FontCategory] = field(default_factory=lambda: [FontCategory.ANY])
    font_size_range: Tuple[int, int] = (24, 48)
    # Colors
    text_colors: Optional[ColorPalette] = None
    per_char_color: bool = True
    # Spacing
    char_spacing_range: Tuple[int, int] = (2, 20)
    # Halo/knockout
    halo_enabled: bool = False
    halo_radius: int = 2
    halo_color: Tuple[int, int, int] = (255, 255, 255)
    # Shadow
    shadow_enabled: bool = False
    shadow_offset: Tuple[int, int] = (2, 2)
    shadow_opacity: int = 150
    # Outline/stroke
    outline_enabled: bool = False
    outline_width: int = 1
    outline_color: Optional[Tuple[int, int, int]] = None  # None = auto darken


@dataclass
class DistortionConfig:
    """Configuration for text distortions."""
    distortions: List[TextDistortion] = field(default_factory=lambda: [TextDistortion.NONE])
    # Rotation
    rotation_range: Tuple[float, float] = (-15.0, 15.0)
    # Wave warp
    wave_amplitude: Tuple[float, float] = (2.0, 8.0)
    wave_frequency: Tuple[float, float] = (0.02, 0.08)
    # Elastic deformation
    elastic_alpha: Tuple[float, float] = (20.0, 60.0)
    elastic_sigma: Tuple[float, float] = (3.0, 6.0)
    # Perspective
    perspective_strength: Tuple[float, float] = (0.001, 0.01)
    # Shear
    shear_range: Tuple[float, float] = (-0.3, 0.3)
    # Scale jitter per character
    scale_jitter: Tuple[float, float] = (0.8, 1.2)
    # Vertical offset per character
    vertical_offset_range: Tuple[int, int] = (-8, 8)
    # Overlap
    overlap_pixels: Tuple[int, int] = (0, 10)


@dataclass
class NoiseConfig:
    """Configuration for noise overlays."""
    overlays: List[NoiseOverlay] = field(default_factory=lambda: [NoiseOverlay.NONE])
    # Crossing lines
    line_count_range: Tuple[int, int] = (3, 15)
    line_width_range: Tuple[int, int] = (1, 3)
    line_color_palette: Optional[ColorPalette] = None
    # Random arcs
    arc_count_range: Tuple[int, int] = (3, 12)
    # Sine waves
    wave_count_range: Tuple[int, int] = (1, 5)
    # Dot scatter
    dot_count_range: Tuple[int, int] = (20, 200)
    dot_radius_range: Tuple[int, int] = (1, 4)
    # Geometric shapes
    shape_count_range: Tuple[int, int] = (5, 20)
    # Stripe bands
    band_count_range: Tuple[int, int] = (2, 8)
    band_opacity_range: Tuple[int, int] = (30, 100)


@dataclass
class PostConfig:
    """Configuration for post-processing effects."""
    effects: List[PostEffect] = field(default_factory=lambda: [PostEffect.NONE])
    # Blur
    blur_radius: Tuple[float, float] = (0.3, 1.5)
    # JPEG
    jpeg_quality: Tuple[int, int] = (30, 80)
    # Color jitter
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    saturation_range: Tuple[float, float] = (0.7, 1.3)
    # Pixelation
    pixel_factor: Tuple[int, int] = (2, 6)


@dataclass 
class CaptchaSchema:
    """Complete schema defining a CAPTCHA generation recipe."""
    name: str = "default"
    image_width: int = 200
    image_height: int = 60
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    text: TextConfig = field(default_factory=TextConfig)
    distortion: DistortionConfig = field(default_factory=DistortionConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    post: PostConfig = field(default_factory=PostConfig)
    # Feature activation probabilities (0.0 - 1.0)
    # Each feature in the lists above is independently activated with this probability
    distortion_prob: float = 0.5
    noise_prob: float = 0.5
    post_prob: float = 0.3


# ═══════════════════════════════════════════════════════════
# Utility: random schema composition
# ═══════════════════════════════════════════════════════════

def random_schema(difficulty: float = 0.5) -> CaptchaSchema:
    """
    Generate a random CaptchaSchema by sampling from the full feature space.
    
    difficulty: 0.0 (trivial) to 1.0 (maximum adversarial).
    
    This is the PRIMARY schema generation method. Every feature dimension is
    sampled independently with difficulty controlling the density and intensity
    of adversarial features. No specific CAPTCHA families are targeted —
    coverage emerges from the combinatorial space of all features.
    """
    # ── Image dimensions ──
    w = random.choice([150, 180, 200, 220, 250, 280, 300, 350])
    h = random.choice([50, 60, 70, 80, 100])

    # ── Background ──
    # Easy: simple backgrounds. Hard: any background.
    simple_bgs = [BackgroundType.SOLID, BackgroundType.GRADIENT]
    medium_bgs = simple_bgs + [BackgroundType.NOISE_SPECKLE, BackgroundType.GRID]
    complex_bgs = medium_bgs + [BackgroundType.TILED_PATCHES, BackgroundType.DENSE_LINES,
                                 BackgroundType.TEXTURE_FILL, BackgroundType.IMAGE_BG]
    
    if difficulty < 0.25:
        bg_pool = simple_bgs
    elif difficulty < 0.5:
        bg_pool = medium_bgs
    elif difficulty < 0.75:
        bg_pool = complex_bgs[:6]
    else:
        bg_pool = complex_bgs
    bg_types = [random.choice(bg_pool)]

    # ── Color scheme (randomized) ──
    # Decide if dark or light background
    is_dark_bg = random.random() < difficulty * 0.3  # more dark bgs at higher difficulty
    
    if is_dark_bg:
        bg_palette = ColorPalette(colors=[
            (random.randint(10, 50), random.randint(10, 50), random.randint(10, 50))
            for _ in range(random.randint(3, 6))
        ])
        # Bright text on dark bg — GUARANTEED high luminance for contrast
        text_palette = ColorPalette(colors=[
            (random.randint(200, 255), random.randint(180, 255), random.randint(180, 255))
            for _ in range(random.randint(4, 10))
        ])
    else:
        bg_palette = ColorPalette(colors=[
            (random.randint(220, 255), random.randint(220, 255), random.randint(220, 255))
            for _ in range(random.randint(3, 6))
        ])
        # Dark text on light bg — GUARANTEED low luminance for contrast
        text_palette = ColorPalette(colors=[
            (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
            for _ in range(random.randint(4, 10))
        ])

    # ── Font selection ──
    # GUARDRAIL: exclude PIXEL and DISPLAY fonts — they produce blocky,
    # unrecognizable glyphs (m → square, B → D, E unreadable)
    readable_font_cats = [FontCategory.SANS_SERIF, FontCategory.SERIF, FontCategory.MONOSPACE,
                          FontCategory.HANDWRITTEN, FontCategory.SCRIPT]
    n_cats = random.randint(1, min(3, max(1, int(1 + difficulty * 2))))
    font_cats = random.sample(readable_font_cats, min(n_cats, len(readable_font_cats)))
    
    # Font size — GUARDRAIL: never below 18px, text must be readable
    min_size = max(24, int(h * 0.40) - int(difficulty * 3))  # min 24px
    max_size = min(int(h * 0.85), 56 - int(difficulty * 4))
    max_size = max(min_size + 4, max_size)

    # ── Character set ──
    charset_pool = [CharSet.ALPHANUMERIC_MIXED, CharSet.ALPHANUMERIC_UPPER,
                    CharSet.ALPHA_MIXED, CharSet.DIGITS_ONLY]
    charset = random.choice(charset_pool)

    # ── Text length ──
    # Estimate max chars that fit: image_width / avg_char_width (approx font_size * 0.7)
    avg_char_w = max(10, int(max_size * 0.6))
    max_chars_that_fit = max(3, w // avg_char_w)
    min_len = random.randint(4, min(5 + int(difficulty), max_chars_that_fit))
    max_len = min(min_len + random.randint(1, 2), max_chars_that_fit)

    # ── Text effects (stochastic) ──
    per_char_color = random.random() < (0.3 + difficulty * 0.5)
    halo_enabled = random.random() < difficulty * 0.4
    halo_radius = random.randint(1, max(1, int(1 + difficulty * 3)))
    shadow_enabled = random.random() < difficulty * 0.5
    shadow_offset = (random.randint(1, 3), random.randint(1, 4))
    shadow_opacity = random.randint(100, 200)
    outline_enabled = random.random() < difficulty * 0.3
    outline_width = random.randint(1, 2)
    char_spacing = (random.randint(-3 if difficulty > 0.5 else 2, 5),
                    random.randint(8, 22))

    # GUARDRAIL: force outline on thin font categories (handwritten/script)
    # These have narrow strokes that become invisible on busy backgrounds
    thin_cats = {FontCategory.HANDWRITTEN, FontCategory.SCRIPT}
    if any(c in thin_cats for c in font_cats):
        outline_enabled = True
        outline_width = max(1, outline_width)

    # ── Text distortions — more features at higher difficulty ──
    # GUARDRAIL: cap at 3 simultaneous distortions to keep text readable
    all_distortions = [d for d in TextDistortion if d != TextDistortion.NONE]
    n_dist = max(0, int(difficulty * 3.5) + random.randint(-1, 1))
    n_dist = min(n_dist, 3, len(all_distortions))
    distortions = random.sample(all_distortions, n_dist) if n_dist > 0 else [TextDistortion.NONE]

    # Randomize distortion parameters — GUARDRAIL: capped to prevent illegibility
    rotation_range = (-5 - difficulty * 12, 5 + difficulty * 12)  # max ±17°
    wave_amp = (2 + difficulty * 1, 3 + difficulty * 4)  # gentle waves
    wave_freq = (0.02, 0.03 + difficulty * 0.03)
    elastic_alpha = (10 + difficulty * 10, 20 + difficulty * 25)  # max ~45 not 60
    elastic_sigma = (2 + difficulty, 3 + difficulty * 1.5)
    perspective_str = (0.001, 0.001 + difficulty * 0.004)  # max 0.005 not 0.008
    shear_range = (-0.05 - difficulty * 0.12, 0.05 + difficulty * 0.12)  # max ±0.17
    scale_jitter = (max(0.8, 1.0 - difficulty * 0.18), min(1.3, 1.0 + difficulty * 0.25))
    v_offset = (-int(2 + difficulty * 5), int(2 + difficulty * 5))  # max ±7px
    overlap = (0, max(1, int(difficulty * 5)))  # max 5px overlap

    # ── Noise overlays ──
    # With split-noise pipeline, background noise (dots, shapes, bands) renders
    # BEFORE text so it can never occlude. Foreground noise (lines, arcs, waves)
    # renders AFTER text and crosses through characters.
    # All noise types are now safe to include.
    all_noise = [n for n in NoiseOverlay if n != NoiseOverlay.NONE]
    n_noise = max(0, int(difficulty * 2.5) + random.randint(-1, 1))
    n_noise = min(n_noise, 3, len(all_noise))
    noise_overlays = random.sample(all_noise, n_noise) if n_noise > 0 else [NoiseOverlay.NONE]

    # ── Post effects ──
    # GUARDRAIL: exclude destructive effects
    safe_post = [
        PostEffect.GAUSSIAN_BLUR,
        PostEffect.JPEG_ARTIFACTS,
        PostEffect.COLOR_JITTER,
        PostEffect.COMPRESSION_NOISE,
        PostEffect.EDGE_ENHANCE,
        # EXCLUDED: HALFTONE (reduces to 1-bit dither, destroys text)
        # EXCLUDED: INVERT (breaks text-bg contrast assumptions)
        # EXCLUDED: PIXELATION (at higher factors makes text unreadable)
    ]
    # Include pixelation only at low difficulty where fonts are large
    if difficulty < 0.4:
        safe_post.append(PostEffect.PIXELATION)

    n_post = max(0, int(difficulty * 1.5) + random.randint(-1, 0))
    n_post = min(n_post, 2, len(safe_post))
    post_effects = random.sample(safe_post, n_post) if n_post > 0 else [PostEffect.NONE]

    # ── Build schema ──
    return CaptchaSchema(
        name=f"synth_d{difficulty:.2f}",
        image_width=w,
        image_height=h,
        background=BackgroundConfig(
            types=bg_types,
            bg_colors=bg_palette,
        ),
        text=TextConfig(
            charset=charset,
            length_range=(min_len, max_len),
            font_categories=font_cats,
            font_size_range=(min_size, max_size),
            text_colors=text_palette,
            per_char_color=per_char_color,
            char_spacing_range=char_spacing,
            halo_enabled=halo_enabled,
            halo_radius=halo_radius,
            shadow_enabled=shadow_enabled,
            shadow_offset=shadow_offset,
            shadow_opacity=shadow_opacity,
            outline_enabled=outline_enabled,
            outline_width=outline_width,
        ),
        distortion=DistortionConfig(
            distortions=distortions,
            rotation_range=rotation_range,
            wave_amplitude=wave_amp,
            wave_frequency=wave_freq,
            elastic_alpha=elastic_alpha,
            elastic_sigma=elastic_sigma,
            perspective_strength=perspective_str,
            shear_range=shear_range,
            scale_jitter=scale_jitter,
            vertical_offset_range=v_offset,
            overlap_pixels=overlap,
        ),
        noise=NoiseConfig(
            overlays=noise_overlays,
            # GUARDRAIL: constrained noise parameters
            dot_count_range=(10, min(50, int(20 + difficulty * 40))),
            dot_radius_range=(1, max(1, min(2, int(1 + difficulty)))),
            line_count_range=(2, min(8, int(2 + difficulty * 6))),
            line_width_range=(1, max(1, min(2, int(1 + difficulty)))),
            arc_count_range=(2, min(6, int(2 + difficulty * 4))),
            wave_count_range=(1, min(3, int(1 + difficulty * 2))),
        ),
        post=PostConfig(
            effects=post_effects,
            # GUARDRAIL: cap blur based on font size
            blur_radius=(0.3, min(1.0, 0.4 + difficulty * 0.5)),  # max 1.0 not 1.5
            jpeg_quality=(max(50, int(80 - difficulty * 30)), 85),  # min quality 50 not 40
        ),
        # GUARDRAIL: lower activation probs to prevent simultaneous feature stacking
        distortion_prob=0.3 + difficulty * 0.4,    # max 0.7 not 0.8
        noise_prob=0.15 + difficulty * 0.35,        # max 0.5 not 0.7
        post_prob=0.1 + difficulty * 0.3,            # max 0.4 not 0.5
        # GUARDRAIL: limit blur radius when font is small
    )

