"""Stub: docTR v0.8.1 imports weasyprint for HTML->PDF reading, which the teacher fine-tune never uses."""


class HTML:  # pragma: no cover
    def __init__(self, *a, **k):
        raise RuntimeError('weasyprint is stubbed out in model4 (HTML input unsupported)')
