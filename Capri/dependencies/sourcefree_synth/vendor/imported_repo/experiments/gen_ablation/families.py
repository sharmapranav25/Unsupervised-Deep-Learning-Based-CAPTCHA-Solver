"""Single source of truth for family config in the Q1–Q3 ablation.

Family charsets are read from the canonical charset files in system5_clean_baseline/charsets/
(line-separated chars) so they never drift from the eval-protocol definitions.
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache

REPO_ROOT = Path(__file__).resolve().parents[2]
CHARSET_DIR = REPO_ROOT / "system5_clean_baseline" / "charsets"
DATA_DIR = REPO_ROOT / "experiments" / "gen_ablation" / "data"

# family name → charset filename
FAMILIES = {
    "cryptbb":  "charset_cryptbb.txt",
    "mgm":      "charset_mgm.txt",
    "darkbay":  "charset_darkbay.txt",
    "city":     "charset_city.txt",
}

# Canonical captcha length per family (from DATA_ABLATION_PLAN.md §1).
# Held constant across all conditions so the only variable is difficulty sampling.
FAMILY_LENGTH = {
    "cryptbb":  5,
    "mgm":      5,
    "darkbay":  6,
    "city":     5,
}

FAMILY_DIMENSIONS = {
    "cryptbb": ((200, 60),),
    "mgm": ((120, 40),),
    "darkbay": ((200, 50),),
    "city": ((318, 79), (324, 81), (788, 198)),
}

FAMILY_ALPHABET_DESCRIPTIONS = {
    "cryptbb": "case-sensitive digits 0-9, uppercase A-Z, and lowercase a-z",
    "mgm": "case-sensitive digits 0-9, uppercase A-Z, and lowercase a-z",
    "darkbay": "digits 0-9 and uppercase A-Z excluding I",
    "city": "digits 0-9 only",
}


@dataclass(frozen=True)
class FamilySpec:
    """Immutable text and image contract shared by every benchmark stage."""

    family: str
    charset: str
    length: int
    observed_dimensions: tuple[tuple[int, int], ...]
    alphabet_description: str
    parser_rule: str = "exact_length_and_ordered_charset_membership:v1"

    def _contract_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "charset": self.charset,
            "length": self.length,
            "observed_dimensions": [list(item) for item in self.observed_dimensions],
            "alphabet_description": self.alphabet_description,
            "parser_rule": self.parser_rule,
        }

    @property
    def stable_hash(self) -> str:
        payload = json.dumps(
            self._contract_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @property
    def prompt(self) -> str:
        return (
            f"Family contract {self.stable_hash}. This {self.family} CAPTCHA contains "
            f"exactly {self.length} characters from {self.alphabet_description}. "
            f"Transcribe exactly and reply with only the {self.length} characters."
        )

    @property
    def generator_instruction(self) -> str:
        return (
            f"Render exactly {self.length} characters. The exact ordered character set is "
            f"{self.charset}. FamilySpec SHA-256: {self.stable_hash}."
        )

    def valid(self, text: str) -> bool:
        return len(text) == self.length and all(char in self.charset for char in text)

    def to_dict(self) -> dict[str, object]:
        return self._contract_dict() | {
            "stable_hash": self.stable_hash,
            "prompt": self.prompt,
            "generator_instruction": self.generator_instruction,
        }


@lru_cache
def family_charset(family: str) -> str:
    """Return the family's charset as a single concatenated string."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; valid: {sorted(FAMILIES)}")
    path = CHARSET_DIR / FAMILIES[family]
    chars = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    return "".join(chars)


@lru_cache
def family_spec(family: str) -> FamilySpec:
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; valid: {sorted(FAMILIES)}")
    return FamilySpec(
        family=family,
        charset=family_charset(family),
        length=FAMILY_LENGTH[family],
        observed_dimensions=FAMILY_DIMENSIONS[family],
        alphabet_description=FAMILY_ALPHABET_DESCRIPTIONS[family],
    )


FAMILY_SPECS = {family: family_spec(family) for family in FAMILIES}


def family_lmdb_path(condition: str, family: str) -> Path:
    """Canonical LMDB path for a (condition, family) cell.

    condition ∈ {"c1_random", "c2_curriculum", "c3_adaptive"}
    """
    return DATA_DIR / condition / family / "100k" / "lmdb"
