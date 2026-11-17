"""
Font Manager for Universal CAPTCHA Generator.

Discovers, categorizes, validates, and serves fonts from configurable paths.
Fully standalone — no dependencies on external project code.
"""

import os
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from PIL import ImageFont, Image, ImageDraw

from .config import FontCategory


# ═══════════════════════════════════════════════════════════
# Default font search paths
# ═══════════════════════════════════════════════════════════

DEFAULT_FONT_DIRS = [
    # Project-local font directories
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "prebuilt_generator", "fonts"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "prebuilt_generator", "fonts_expanded"),
    # System font directories (Linux)
    "/usr/share/fonts/truetype",
    "/usr/share/fonts/opentype",
    "/usr/local/share/fonts",
]

# Alphanumeric charset for validation
REQUIRED_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Heuristic keyword → category mapping
CATEGORY_KEYWORDS = {
    FontCategory.MONOSPACE: [
        "mono", "code", "cousine", "courier", "hack", "fira code",
        "source code", "jetbrains", "liberation mono", "ubuntu mono",
        "cascadia", "dmmono", "azeretmono", "spacemono", "b612mono",
    ],
    FontCategory.SERIF: [
        "serif", "times", "garamond", "georgia", "palatino", "baskervill",
        "bodoni", "caslon", "didot", "mincho", "noto serif", "libre baskerville",
        "playfair", "lora", "merriweather", "vollkorn", "crimson", "cardo",
        "cormorant", "eb garamond", "old standard", "spectral", "alegreya",
        "gentium", "pt serif", "liberation serif", "free serif",
    ],
    FontCategory.HANDWRITTEN: [
        "handwrit", "hand", "script", "cursive", "brush", "caveat", "kalam",
        "dancing", "pacifico", "lobster", "satisfy", "cookie", "allura",
        "sacramento", "great vibes", "alex brush", "architects daughter",
        "indie flower", "schoolbell", "coming soon", "covered by",
        "homemade", "permanent marker", "rock salt", "shadows into",
        "just another", "walter", "bad script", "cedarville", "calligraff",
        "damion", "dawning", "crafty", "sunshiney",
    ],
    FontCategory.DISPLAY: [
        "display", "bangers", "bungee", "impact", "fredoka", "righteous",
        "modak", "monoton", "creepster", "anton", "bevan", "bowlby",
        "black ops", "alfa slab", "big shoulders", "calistoga",
        "concert one", "luckiest", "passion one", "press start",
        "special elite", "ultra",
    ],
    FontCategory.PIXEL: [
        "pixel", "press start", "silkscreen", "dot gothic", "doto",
        "bitcount", "bytesized", "digital",
    ],
    FontCategory.SCRIPT: [
        "script", "allura", "arizonia", "bilbo", "bonheur", "birthstone",
        "carattere", "comforter", "cherish", "corinthia", "devonshire",
        "dr sugiyama",
    ],
}


class FontManager:
    """
    Discovers fonts from configurable directories, categorizes them,
    validates glyph coverage, and provides random font selection.
    """

    def __init__(
        self,
        font_dirs: Optional[List[str]] = None,
        validate_glyphs: bool = True,
        cache_size: int = 200,
    ):
        self.font_dirs = font_dirs or DEFAULT_FONT_DIRS
        self.validate_glyphs = validate_glyphs
        self._cache_size = cache_size

        # font_path → FontCategory
        self._font_catalog: Dict[str, FontCategory] = {}
        # FontCategory → list of font paths
        self._by_category: Dict[FontCategory, List[str]] = {c: [] for c in FontCategory}
        # Loaded font cache: (path, size) → ImageFont
        self._cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}
        # Paths that failed validation
        self._invalid: Set[str] = set()

        self._discover_fonts()

    def _discover_fonts(self):
        """Walk font directories, validate against whitelist or per-char checks, and build catalog."""
        seen = set()
        candidates = []

        for font_dir in self.font_dirs:
            font_dir = os.path.expanduser(font_dir)
            if not os.path.isdir(font_dir):
                continue
            for root, _, files in os.walk(font_dir):
                for fname in files:
                    if not fname.lower().endswith(('.ttf', '.otf')):
                        continue
                    base = fname.lower()
                    if base in seen:
                        continue
                    seen.add(base)
                    if self._is_non_english_filename(base):
                        continue
                    fpath = os.path.join(root, fname)
                    candidates.append((fpath, fname))

        # Load deep audit whitelist if available
        whitelist_path = os.path.join(os.path.dirname(__file__), '.font_whitelist.json')
        whitelist = None
        if os.path.exists(whitelist_path):
            try:
                import json
                with open(whitelist_path) as f:
                    data = json.load(f)
                whitelist = set(data.get('whitelist', []))
            except Exception:
                whitelist = None

        for fpath, fname in candidates:
            # Use whitelist if available, else fallback to per-char validation
            if whitelist is not None:
                is_valid = fpath in whitelist
            else:
                is_valid = self._validate_font(fpath)
            
            if is_valid:
                category = self._categorize(fname)
                self._font_catalog[fpath] = category
                self._by_category[category].append(fpath)
                if category != FontCategory.ANY:
                    self._by_category[FontCategory.ANY].append(fpath)
            else:
                self._invalid.add(fpath)

    @staticmethod
    def _is_non_english_filename(name_lower: str) -> bool:
        """Pre-filter fonts that are obviously non-English by filename keywords."""
        # CJK, Arabic, Indic, Thai, Hebrew, and other non-Latin script indicators
        non_english_keywords = [
            'noto sans arabic', 'noto sans hebrew', 'noto sans thai',
            'noto sans devanagari', 'noto sans bengali', 'noto sans tamil',
            'noto sans telugu', 'noto sans kannada', 'noto sans malayalam',
            'noto sans gujarati', 'noto sans gurmukhi', 'noto sans oriya',
            'noto sans sinhala', 'noto sans myanmar', 'noto sans khmer',
            'noto sans lao', 'noto sans tibetan', 'noto sans ethiopic',
            'noto sans georgian', 'noto sans armenian', 'noto sans cherokee',
            'noto sans coptic', 'noto sans adlam', 'noto sans balinese',
            'noto sans javanese', 'noto sans sundanese', 'noto sans tagalog',
            'noto sans thaana', 'noto sans nko', 'noto sans vai',
            'noto sans yi', 'noto sans mongolian', 'noto sans tifinagh',
            'noto sans cjk', 'noto sans jp', 'noto sans kr', 'noto sans sc',
            'noto sans tc', 'noto sans hk',
            'noto serif cjk', 'noto serif jp', 'noto serif kr',
            'noto serif sc', 'noto serif tc',
            'noto serif devanagari', 'noto serif bengali', 'noto serif tamil',
            'noto serif hebrew', 'noto serif arabic', 'noto serif thai',
            'noto serif ethiopic', 'noto serif georgian', 'noto serif armenian',
            'noto serif khmer', 'noto serif lao', 'noto serif myanmar',
            'noto serif sinhala', 'noto serif tibetan', 'noto serif telugu',
            'noto serif kannada', 'noto serif malayalam', 'noto serif gujarati',
            'noto serif gurmukhi',
            # Generic non-Latin keywords
            'arabic', 'hebrew', 'thai', 'devanagari', 'bengali', 'tamil',
            'telugu', 'kannada', 'malayalam', 'gujarati', 'gurmukhi',
            'sinhala', 'myanmar', 'khmer', 'tibetan', 'ethiopic',
            'georgian', 'armenian', 'cherokee', 'chinese', 'japanese',
            'korean', 'hangul', 'hiragana', 'katakana', 'kanji',
            'urdu', 'farsi', 'persian', 'hindi',
            # Symbol/icon fonts
            'symbol', 'icons', 'emoji', 'dingbat', 'wingding',
            'materialicon', 'fontawesome', 'ionicon', 'weathericon',
            'webding', 'zapfdingbats',
        ]
        name_clean = name_lower.replace('-', ' ').replace('_', ' ')
        for kw in non_english_keywords:
            if kw in name_clean:
                return True
        return False

    def _categorize(self, filename: str) -> FontCategory:
        """Categorize a font by filename heuristics."""
        name_lower = filename.lower().replace('-', ' ').replace('_', ' ')

        # Check each category's keywords
        for category, keywords in CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in name_lower:
                    return category

        # Default: sans-serif (most common)
        return FontCategory.SANS_SERIF

    def _validate_font(self, font_path: str, size: int = 30) -> bool:
        """
        Validate a font renders ALL 62 alphanumeric chars without tofu/fallback glyphs.
        
        Uses two detection methods:
        1. Zero-width check: character renders with no width (missing glyph)
        2. Duplicate bitmap check: if many different chars produce identical bitmaps, 
           the font is using a fallback/tofu glyph for unsupported characters
        """
        try:
            font = ImageFont.truetype(font_path, size)
            test_img = Image.new('L', (size * 2, size * 2), 0)
            draw = ImageDraw.Draw(test_img)

            bitmaps = {}
            for ch in REQUIRED_CHARS:
                # Clear canvas
                draw.rectangle([(0, 0), (size * 2 - 1, size * 2 - 1)], fill=0)
                draw.text((5, 5), ch, fill=255, font=font)
                
                bbox = draw.textbbox((5, 5), ch, font=font)
                char_width = bbox[2] - bbox[0]
                char_height = bbox[3] - bbox[1]
                
                # Check 1: zero width/height = missing glyph
                if char_width < 1 or char_height < 1:
                    return False
                
                # Capture bitmap for duplicate detection
                import numpy as np
                arr = np.array(test_img)
                # Crop to character region
                region = arr[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                if region.size == 0:
                    return False
                bitmaps[ch] = region.tobytes()

            # Check 2: if too many chars produce identical bitmaps, it's a tofu font
            # Group by identical bitmaps
            from collections import Counter
            bitmap_counts = Counter(bitmaps.values())
            # If any single bitmap appears for more than 4 different characters,
            # the font is likely using a fallback/tofu glyph
            max_identical = max(bitmap_counts.values()) if bitmap_counts else 0
            if max_identical > 4:
                return False
            
            # Check 3: if all chars have suspiciously uniform dimensions (box-like tofu)
            # Real fonts have variable widths (W is wider than i)
            widths = []
            for ch in REQUIRED_CHARS:
                bbox = draw.textbbox((5, 5), ch, font=font)
                widths.append(bbox[2] - bbox[0])
            if len(set(widths)) < 5:  # fewer than 5 unique widths across 62 chars = tofu
                return False

            # Check 4: glyph fill ratio — catch blocky fonts where chars are solid rectangles
            # Real letters have internal whitespace (holes in O, gaps in E, thin strokes in l)
            # Block/blob fonts fill their bounding box uniformly
            high_fill_count = 0
            for ch in REQUIRED_CHARS:
                draw.rectangle([(0, 0), (size * 2 - 1, size * 2 - 1)], fill=0)
                draw.text((5, 5), ch, fill=255, font=font)
                bbox = draw.textbbox((5, 5), ch, font=font)
                cw = bbox[2] - bbox[0]
                ch_h = bbox[3] - bbox[1]
                if cw < 2 or ch_h < 2:
                    continue
                region = np.array(test_img)[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                fill_ratio = (region > 128).sum() / max(1, region.size)
                if fill_ratio > 0.85:
                    high_fill_count += 1
            # If more than 20 chars (out of 62) are solid blocks, reject the font
            if high_fill_count > 20:
                return False

            return True
        except Exception:
            return False

    def get_font(
        self,
        category: FontCategory = FontCategory.ANY,
        size: int = 32,
    ) -> ImageFont.FreeTypeFont:
        """
        Get a random font of the requested category at the given size.
        Falls back to ANY if the category is empty, then to default.
        """
        candidates = self._by_category.get(category, [])
        if not candidates:
            candidates = self._by_category.get(FontCategory.ANY, [])
        if not candidates:
            # Ultimate fallback
            return ImageFont.load_default()

        # Try up to 5 random fonts
        for _ in range(5):
            fpath = random.choice(candidates)
            if fpath in self._invalid:
                continue
            cache_key = (fpath, size)
            if cache_key in self._cache:
                return self._cache[cache_key]
            try:
                font = ImageFont.truetype(fpath, size)
                # Cache management
                if len(self._cache) >= self._cache_size:
                    # Evict oldest entry
                    oldest = next(iter(self._cache))
                    del self._cache[oldest]
                self._cache[cache_key] = font
                return font
            except Exception:
                self._invalid.add(fpath)
                continue

        return ImageFont.load_default()

    def get_font_path(
        self,
        category: FontCategory = FontCategory.ANY,
    ) -> Optional[str]:
        """Get a random font path of the requested category."""
        candidates = self._by_category.get(category, [])
        if not candidates:
            candidates = self._by_category.get(FontCategory.ANY, [])
        if not candidates:
            return None
        return random.choice(candidates)

    @property
    def total_fonts(self) -> int:
        return len(self._font_catalog)

    @property
    def category_counts(self) -> Dict[str, int]:
        return {
            cat.value: len(paths)
            for cat, paths in self._by_category.items()
            if cat != FontCategory.ANY
        }

    def summary(self) -> str:
        lines = [f"FontManager: {self.total_fonts} fonts discovered"]
        for cat, count in self.category_counts.items():
            lines.append(f"  {cat}: {count}")
        lines.append(f"  invalid: {len(self._invalid)}")
        return "\n".join(lines)
