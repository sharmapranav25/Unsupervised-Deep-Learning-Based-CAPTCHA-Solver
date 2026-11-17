"""
Text Generator for Universal CAPTCHA Generator.

Generates random text strings with configurable character sets,
length ranges, and confusable character oversampling.
"""

import random
import string
from typing import Optional, List

from .config import CharSet


# ═══════════════════════════════════════════════════════════
# Character sets
# ═══════════════════════════════════════════════════════════

CHARSET_MAP = {
    CharSet.LOWERCASE: string.ascii_lowercase,
    CharSet.UPPERCASE: string.ascii_uppercase,
    CharSet.DIGITS: string.digits,
    CharSet.ALPHA_LOWER: string.ascii_lowercase,
    CharSet.ALPHA_UPPER: string.ascii_uppercase,
    CharSet.ALPHA_MIXED: string.ascii_letters,
    CharSet.ALPHANUMERIC_LOWER: string.ascii_lowercase + string.digits,
    CharSet.ALPHANUMERIC_UPPER: string.ascii_uppercase + string.digits,
    CharSet.ALPHANUMERIC_MIXED: string.ascii_letters + string.digits,
    CharSet.DIGITS_ONLY: string.digits,
}

# Confusable character groups — characters that look similar and are common OCR errors
CONFUSABLE_GROUPS = [
    '0Oo',
    '1lI',
    '5Ss',
    '8B',
    '2Zz',
    '9qg',
    'Cc',
    'Kk',
    'Pp',
    'Uu',
    'Vv',
    'Ww',
    'Xx',
    'Yy',
    'bh',     # similar in some fonts
    'rn',     # looks like 'm' together
    'cl',     # looks like 'd' together
    'DO0',    # D vs O vs 0
]


def generate_text(
    charset: CharSet = CharSet.ALPHANUMERIC_MIXED,
    length: Optional[int] = None,
    length_range: tuple = (4, 7),
    confusable_prob: float = 0.4,
    custom_charset: Optional[str] = None,
) -> str:
    """
    Generate a random CAPTCHA text string.

    Args:
        charset: Which character set to use
        length: Exact length (overrides length_range)
        length_range: (min, max) inclusive range for random length
        confusable_prob: Probability of forcing at least one confusable char
        custom_charset: If provided, overrides charset enum

    Returns:
        Random text string
    """
    if length is None:
        length = random.randint(*length_range)

    chars = custom_charset if custom_charset else CHARSET_MAP.get(charset, string.ascii_letters + string.digits)

    # Generate base text
    text = list(random.choices(chars, k=length))

    # Optionally inject confusable characters
    if random.random() < confusable_prob:
        # Filter groups to only include chars that exist in our charset
        valid_groups = []
        for group in CONFUSABLE_GROUPS:
            valid_chars = [c for c in group if c in chars]
            if len(valid_chars) >= 2:
                valid_groups.append(valid_chars)

        if valid_groups:
            group = random.choice(valid_groups)
            # Replace 1-2 positions with confusable chars
            n_inject = min(random.randint(1, 2), length)
            positions = random.sample(range(length), n_inject)
            for pos in positions:
                text[pos] = random.choice(group)

    return ''.join(text)


def generate_text_batch(
    count: int,
    charset: CharSet = CharSet.ALPHANUMERIC_MIXED,
    length_range: tuple = (4, 7),
    confusable_prob: float = 0.4,
    unique: bool = True,
) -> List[str]:
    """
    Generate a batch of random text strings.

    Args:
        count: Number of strings to generate
        charset: Character set to use
        length_range: (min, max) length range
        confusable_prob: Probability of confusable injection
        unique: If True, ensures no duplicate strings

    Returns:
        List of text strings
    """
    results = []
    seen = set()

    max_attempts = count * 10

    for _ in range(max_attempts):
        if len(results) >= count:
            break

        text = generate_text(
            charset=charset,
            length_range=length_range,
            confusable_prob=confusable_prob,
        )

        if unique and text in seen:
            continue

        results.append(text)
        seen.add(text)

    return results
