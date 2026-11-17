"""
Schema Sampling for Universal CAPTCHA Generator.

Provides difficulty-stratified random schema generation from the full feature space.
No hand-crafted family presets — all schemas are composed by random sampling from
the taxonomy of visual features defined in config.py.

Coverage of real-world CAPTCHA styles emerges naturally from the combinatorial
space of backgrounds × distortions × noise × post-processing × fonts.
"""

from .config import CaptchaSchema, random_schema

import random as _random


# ═══════════════════════════════════════════════════════════
# Difficulty tiers — probability distributions, not recipes
# ═══════════════════════════════════════════════════════════

# Difficulty ranges for each tier (min_difficulty, max_difficulty)
DIFFICULTY_TIERS = {
    1: (0.0, 0.15),    # Trivial — clean text, simple backgrounds
    2: (0.15, 0.35),   # Easy — mild distortion, some noise
    3: (0.35, 0.55),   # Medium — moderate features across all layers
    4: (0.55, 0.80),   # Hard — heavy distortion, complex backgrounds
    5: (0.80, 1.0),    # Adversarial — maximum feature density
}


def sample_schema(difficulty: float = None, tier: int = None) -> CaptchaSchema:
    """
    Generate a random schema by sampling from the feature space.
    
    Args:
        difficulty: Explicit difficulty in [0.0, 1.0]. If None, sampled randomly.
        tier: Difficulty tier (1-5). Overrides difficulty if provided.
    
    Returns:
        A randomly composed CaptchaSchema.
    """
    if tier is not None:
        tier = max(1, min(5, tier))
        d_min, d_max = DIFFICULTY_TIERS[tier]
        difficulty = _random.uniform(d_min, d_max)
    elif difficulty is None:
        difficulty = _random.random()
    
    difficulty = max(0.0, min(1.0, difficulty))
    return random_schema(difficulty)


def sample_schema_batch(
    count: int,
    difficulty: float = None,
    tier: int = None,
    progressive: bool = False,
) -> list:
    """
    Generate multiple random schemas.
    
    Args:
        count: Number of schemas to generate.
        difficulty: Fixed difficulty for all (if set).
        tier: Fixed tier for all (if set).
        progressive: If True, linearly increase difficulty from 0 to 1.
    
    Returns:
        List of CaptchaSchema objects.
    """
    schemas = []
    for i in range(count):
        if progressive:
            d = i / max(1, count - 1)
            schemas.append(random_schema(d))
        elif difficulty is not None or tier is not None:
            schemas.append(sample_schema(difficulty=difficulty, tier=tier))
        else:
            schemas.append(sample_schema())
    return schemas
