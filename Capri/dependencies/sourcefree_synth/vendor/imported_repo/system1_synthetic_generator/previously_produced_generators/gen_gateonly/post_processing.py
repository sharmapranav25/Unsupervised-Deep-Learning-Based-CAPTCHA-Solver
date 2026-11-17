"""
Post-Processing Effects for Universal CAPTCHA Generator.

Final image adjustments applied after text and noise rendering:
blur, JPEG artifacts, color jitter, pixelation, edge enhance, invert, etc.
"""

import io
import random

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

from .config import PostConfig, PostEffect


# ═══════════════════════════════════════════════════════════
# Post-processing functions
# ═══════════════════════════════════════════════════════════

def post_gaussian_blur(img: Image.Image, config: PostConfig) -> Image.Image:
    """Apply Gaussian blur with random radius."""
    radius = random.uniform(*config.blur_radius)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def post_jpeg_artifacts(img: Image.Image, config: PostConfig) -> Image.Image:
    """Simulate JPEG compression artifacts by encoding/decoding at low quality."""
    quality = random.randint(*config.jpeg_quality)
    buffer = io.BytesIO()
    img = img.convert('RGB')
    img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert('RGB')


def post_color_jitter(img: Image.Image, config: PostConfig) -> Image.Image:
    """Random brightness, contrast, and saturation adjustments."""
    # Brightness
    factor = random.uniform(*config.brightness_range)
    img = ImageEnhance.Brightness(img).enhance(factor)

    # Contrast
    factor = random.uniform(*config.contrast_range)
    img = ImageEnhance.Contrast(img).enhance(factor)

    # Saturation
    factor = random.uniform(*config.saturation_range)
    img = ImageEnhance.Color(img).enhance(factor)

    return img


def post_pixelation(img: Image.Image, config: PostConfig) -> Image.Image:
    """Downscale then upscale to create pixelation effect."""
    w, h = img.size
    factor = random.randint(*config.pixel_factor)
    small = img.resize((max(1, w // factor), max(1, h // factor)), Image.NEAREST)
    return small.resize((w, h), Image.NEAREST)


def post_edge_enhance(img: Image.Image, config: PostConfig) -> Image.Image:
    """Apply edge enhancement / sharpening."""
    if random.random() < 0.5:
        return img.filter(ImageFilter.EDGE_ENHANCE)
    else:
        return img.filter(ImageFilter.SHARPEN)


def post_invert(img: Image.Image, config: PostConfig) -> Image.Image:
    """Invert image colors."""
    return ImageOps.invert(img.convert('RGB'))


def post_halftone(img: Image.Image, config: PostConfig) -> Image.Image:
    """Simulate halftone print effect via dithering."""
    # Convert to grayscale, dither, then re-color
    gray = img.convert('L')
    # Floyd-Steinberg dithering to 1-bit
    bw = gray.convert('1', dither=Image.Dither.FLOYDSTEINBERG)
    # Convert back to RGB
    result = bw.convert('RGB')

    # Tint with a random subtle color
    tint = np.array(result).astype(np.float32)
    tint_color = np.array([
        random.uniform(0.9, 1.1),
        random.uniform(0.9, 1.1),
        random.uniform(0.9, 1.1),
    ])
    tint = np.clip(tint * tint_color, 0, 255).astype(np.uint8)
    return Image.fromarray(tint)


def post_compression_noise(img: Image.Image, config: PostConfig) -> Image.Image:
    """Add uniform Gaussian noise simulating sensor/compression noise."""
    arr = np.array(img).astype(np.float32)
    noise_level = random.uniform(3, 12)
    noise = np.random.normal(0, noise_level, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ═══════════════════════════════════════════════════════════
# Dispatcher
# ═══════════════════════════════════════════════════════════

_POST_GENERATORS = {
    PostEffect.GAUSSIAN_BLUR: post_gaussian_blur,
    PostEffect.JPEG_ARTIFACTS: post_jpeg_artifacts,
    PostEffect.COLOR_JITTER: post_color_jitter,
    PostEffect.PIXELATION: post_pixelation,
    PostEffect.EDGE_ENHANCE: post_edge_enhance,
    PostEffect.INVERT: post_invert,
    PostEffect.HALFTONE: post_halftone,
    PostEffect.COMPRESSION_NOISE: post_compression_noise,
}


def apply_post_processing(
    image: Image.Image,
    config: PostConfig,
    activation_prob: float = 0.3,
) -> Image.Image:
    """
    Apply post-processing effects to the image.
    Each configured effect is independently activated with activation_prob.
    """
    for effect in config.effects:
        if effect == PostEffect.NONE:
            continue
        if random.random() > activation_prob:
            continue

        generator = _POST_GENERATORS.get(effect)
        if generator:
            image = generator(image, config)

    # Ensure RGB output
    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image
