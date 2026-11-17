"""
Universal CAPTCHA Generator

A standalone, modular system for generating diverse synthetic CAPTCHA images
spanning multiple schemas, families, adversarial features and difficulty levels.
Designed to produce pre-training data for OCR models (SVTRv2, etc.).
"""

__version__ = "1.0.0"

from .generator import UniversalCaptchaGenerator
