"""
Main Generator Orchestrator for Universal CAPTCHA Generator.

Ties together all components: schema sampling, text generation,
background rendering, text rendering, noise application, and post-processing.

All schemas are composed by random sampling from the feature space —
no hand-crafted family presets are used. A quality validation gate
rejects degenerate samples and retries with a new schema.
"""

import random
import numpy as np
from typing import Optional, Tuple, List

from PIL import Image

from .config import CaptchaSchema, random_schema
from .font_manager import FontManager
from .text_generator import generate_text
from .backgrounds import generate_background
from .text_renderer import render_text_on_image
from .noise_layers import apply_background_noise, apply_foreground_noise
from .post_processing import apply_post_processing
from .quality_validator import validate_captcha
from .schema_presets import sample_schema
from .curriculum import CurriculumScheduler


# Maximum retries before accepting whatever we get
MAX_RETRIES = 15


def _is_degenerate(meta, ov=0.45, hmin=14, blob=0.82, cmin=45):
    """Per-character degeneracy gate from the instrumented renderer's meta. Flags:
    char-box overlap (clumping), crushed/too-small glyph, solid-ink blob, low text/bg contrast."""
    if not meta:
        return False
    overlaps = []
    for a, b in zip(meta, meta[1:]):
        gap = b["box"][0] - a["box"][2]
        minw = max(1, min(a["w"], b["w"]))
        overlaps.append(max(0, -gap) / minw)
    max_overlap = max(overlaps) if overlaps else 0.0
    return (max_overlap > ov
            or min(m["h"] for m in meta) < hmin
            or max(m["ink"] for m in meta) > blob
            or min(m["contrast"] for m in meta) < cmin)


class UniversalCaptchaGenerator:
    """
    Main generator class that orchestrates all CAPTCHA generation components.
    
    All schemas are randomly sampled from the feature space, stratified by
    difficulty level. No specific CAPTCHA families are targeted.
    A quality gate rejects degenerate samples (invisible text, no contrast,
    excessive noise) and retries with a fresh random schema.

    Usage:
        gen = UniversalCaptchaGenerator()
        img, label = gen.generate()              # Random difficulty
        img, label = gen.generate(difficulty=0.7) # Specific difficulty
        img, label = gen.generate(tier=3)         # Curriculum tier
    """

    def __init__(
        self,
        font_dirs: Optional[List[str]] = None,
        seed: Optional[int] = None,
        validate: bool = True,
    ):
        """
        Initialize the generator.

        Args:
            font_dirs: Custom font directories. If None, uses defaults.
            seed: Random seed for reproducibility.
            validate: If True, reject degenerate samples and retry.
        """
        if seed is not None:
            random.seed(seed)

        self._font_manager = FontManager(font_dirs=font_dirs)
        self._curriculum = CurriculumScheduler()
        self._validate = validate

    @property
    def font_manager(self) -> FontManager:
        return self._font_manager

    @property
    def curriculum(self) -> CurriculumScheduler:
        return self._curriculum

    def _resolve_schema(
        self,
        schema_obj: Optional[CaptchaSchema],
        difficulty: Optional[float],
        tier: Optional[int],
    ) -> CaptchaSchema:
        """Resolve a schema from the given parameters."""
        if schema_obj is not None:
            return schema_obj
        elif tier is not None:
            return self._curriculum.get_schema(tier=tier)
        elif difficulty is not None:
            return random_schema(difficulty=difficulty)
        else:
            return sample_schema()

    def _render_pipeline(
        self,
        schema: CaptchaSchema,
        text: str,
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Run the full rendering pipeline with split-noise architecture.
        
        Pipeline order:
          1. Background
          2. Background noise (dots, shapes, bands) — text sits ON TOP
          3. Text rendering (with per-char distortions)
          4. Foreground noise (crossing lines, arcs, waves) — crosses THROUGH text
          5. Post-processing (blur, JPEG, color jitter)
        
        Returns:
            (final_image, pre_noise_image) — pre_noise used by quality validator.
        """
        # 1. Generate background
        bg = generate_background(schema.image_width, schema.image_height, schema.background)

        # 2. Apply BACKGROUND noise (dots, geometric shapes, bands)
        #    These go UNDER the text so they can never fully occlude it
        bg_with_noise = apply_background_noise(bg, schema.noise, activation_prob=schema.noise_prob)

        # 3. Render text on top of noised background (with per-char meta for the degeneracy gate)
        with_text, meta = render_text_on_image(bg_with_noise, text, schema, self._font_manager,
                                               return_meta=True)

        # Capture the pre-foreground-noise state for validation
        pre_fg_noise = with_text.copy()

        # 4. Apply FOREGROUND noise (crossing lines, arcs, sine waves)
        #    These cross THROUGH the text, which is the actual adversarial feature
        with_fg_noise = apply_foreground_noise(with_text, schema.noise, activation_prob=schema.noise_prob)

        # 5. Apply post-processing (blur, JPEG artifacts, etc.)
        final = apply_post_processing(with_fg_noise, schema.post, activation_prob=schema.post_prob)

        # 5. Ensure final size matches schema
        if final.size != (schema.image_width, schema.image_height):
            final = final.resize((schema.image_width, schema.image_height), Image.LANCZOS)

        # Re-measure per-char contrast on the FINAL image: foreground noise + post-processing
        # (blur/jpeg/color-jitter) run AFTER the render-time measurement and can wash text back out.
        # The gate must judge what's actually visible in the output, not the pre-noise render.
        if meta:
            farr = np.asarray(final.convert('RGB'), dtype=np.float32)
            flum = 0.299 * farr[..., 0] + 0.587 * farr[..., 1] + 0.114 * farr[..., 2]
            H, W = flum.shape
            for m in meta:
                x0, y0, x1, y1 = m["box"]
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(W, x1), min(H, y1)
                if x1 <= x0 or y1 <= y0:
                    m["contrast"] = 0.0
                    continue
                patch = flum[y0:y1, x0:x1]
                dev = np.abs(patch - np.median(patch))   # deviation from local paper
                k = max(1, int(0.15 * dev.size))          # strongest 15% = the ink, if any survived
                m["contrast"] = float(np.sort(dev.ravel())[-k:].mean())

        return final, pre_fg_noise, meta

    def generate(
        self,
        schema_obj: Optional[CaptchaSchema] = None,
        text: Optional[str] = None,
        difficulty: Optional[float] = None,
        tier: Optional[int] = None,
    ) -> Tuple[Image.Image, str]:
        """
        Generate a single CAPTCHA image with label.

        Args:
            schema_obj: A CaptchaSchema object directly (highest priority).
            text: Specific text to render. If None, generates random text.
            difficulty: Difficulty level 0.0-1.0 for random schema generation.
            tier: Curriculum tier 1-5 for tier-based generation.

        Returns:
            Tuple of (PIL Image, label string).

        Priority: schema_obj > tier > difficulty > random
        """
        for attempt in range(MAX_RETRIES):
            # Resolve schema (re-sample on retry for randomized schemas)
            if attempt == 0:
                s = self._resolve_schema(schema_obj, difficulty, tier)
            else:
                # On retry, re-sample schema at same difficulty level
                if difficulty is not None:
                    s = random_schema(difficulty=difficulty)
                elif tier is not None:
                    s = self._curriculum.get_schema(tier=tier)
                else:
                    s = sample_schema()

            # Generate text
            label = text if text is not None else generate_text(
                charset=s.text.charset,
                length_range=s.text.length_range,
            )

            # Run rendering pipeline
            final_img, pre_noise_img, meta = self._render_pipeline(s, label)

            # Quality validation — scale thresholds by difficulty
            if not self._validate:
                return final_img, label

            # Per-character degeneracy gate: reject clumped/low-contrast/crushed glyphs and retry
            degenerate = _is_degenerate(meta)

            # Strict at low difficulty, lenient at high difficulty
            d = difficulty if difficulty is not None else 0.5
            contrast_threshold = max(35, 55 - d * 20)   # d=0: 55, d=1: 35
            edge_threshold = max(0.02, 0.04 - d * 0.02) # d=0: 0.04, d=1: 0.02
            vis_threshold = max(0.20, 0.35 - d * 0.15)  # d=0: 0.35, d=1: 0.20

            if (not degenerate) and validate_captcha(final_img, pre_noise_img,
                                min_contrast=contrast_threshold,
                                min_edge_density=edge_threshold,
                                min_text_visibility=vis_threshold):
                return final_img, label

            # Track best attempt by contrast for fallback (never keep a degenerate candidate as best)
            if not degenerate:
                gray = np.array(final_img.convert('L'), dtype=np.float32)
                contrast_score = gray.std()
                if not hasattr(self, '_best') or contrast_score > self._best_score:
                    self._best = (final_img, label)
                    self._best_score = contrast_score
            # else: retry with new schema

        # All retries failed — return the best attempt we found
        if hasattr(self, '_best'):
            result = self._best
            del self._best
            del self._best_score
            return result
        return final_img, label

    def generate_batch(
        self,
        count: int,
        difficulty: Optional[float] = None,
        tier: Optional[int] = None,
        mode: str = "random",
    ) -> List[Tuple[Image.Image, str]]:
        """
        Generate a batch of CAPTCHA images.

        Args:
            count: Number of images to generate.
            difficulty: Difficulty level (applies to all if set).
            tier: Curriculum tier (applies to all if set).
            mode: "random" (random difficulty), "curriculum" (tier-based),
                  "progressive" (easy→hard over batch).

        Returns:
            List of (PIL Image, label string) tuples.
        """
        results = []

        for i in range(count):
            if mode == "progressive":
                progress = i / max(count - 1, 1)
                s = self._curriculum.get_progressive_schema(progress)
                img, label = self.generate(schema_obj=s)
            elif mode == "curriculum":
                img, label = self.generate(tier=tier)
            elif difficulty is not None:
                img, label = self.generate(difficulty=difficulty)
            else:
                img, label = self.generate()

            results.append((img, label))

        return results

    def info(self) -> str:
        """Return a summary of the generator's capabilities."""
        lines = [
            "Universal CAPTCHA Generator",
            f"  Mode: Random feature-space sampling (no hand-crafted presets)",
            f"  Quality gate: {'enabled' if self._validate else 'disabled'}",
            f"  {self._font_manager.summary()}",
            f"  Curriculum tiers: 5 (difficulty 0.0 → 1.0)",
        ]
        return "\n".join(lines)
