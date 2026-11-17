# experiments/gen_ablation/generate_lmdb.py
"""Generate a single (condition, family) 100k LMDB for the Q1–Q3 ablation.

Three conditions, one variable (difficulty-sampling strategy):
  c1_random      — difficulty ~ U(0,1), no tier structure
  c2_curriculum  — fixed tier distribution {1:0.1, 2:0.2, 3:0.3, 4:0.25, 5:0.15}
  c3_adaptive    — Optuna-learned tier weights (passed in as JSON)

All three reuse system1_synthetic_generator/.../gen_gateonly/ unchanged.
Per-family charset is plumbed via generate_text(custom_charset=...).
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
# Repo root first: makes `experiments.gen_ablation.families` importable when this
# file is run as a standalone CLI script (pytest already resolves it via rootdir).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# gen_gateonly lives here and is already importable as a package
GEN_PARENT = REPO_ROOT / "system1_synthetic_generator" / "previously_produced_generators"
if str(GEN_PARENT) not in sys.path:
    sys.path.insert(0, str(GEN_PARENT))

from gen_gateonly.generator import UniversalCaptchaGenerator  # noqa: E402
from gen_gateonly.config import random_schema  # noqa: E402
from gen_gateonly.curriculum import CurriculumScheduler  # noqa: E402
from gen_gateonly.text_generator import generate_text  # noqa: E402
from gen_gateonly.batch_generate import create_lmdb_dataset  # noqa: E402

from experiments.gen_ablation.families import family_charset, family_lmdb_path, FAMILY_LENGTH  # noqa: E402

# gen_gateonly default curriculum weights (from curriculum.py line 60)
C2_DEFAULT_WEIGHTS = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.25, 5: 0.15}


def sample_difficulty_c1() -> float:
    """C1: uniform difficulty in [0,1]."""
    return random.uniform(0.0, 1.0)


def sample_difficulty_c2() -> CurriculumScheduler:
    """C2: fixed curriculum scheduler with gen_gateonly's default distribution."""
    return CurriculumScheduler(tier_distribution=dict(C2_DEFAULT_WEIGHTS))


def sample_difficulty_c3(weights: Dict[int, float]) -> CurriculumScheduler:
    """C3: curriculum scheduler with Optuna-learned tier weights."""
    return CurriculumScheduler(tier_distribution=weights)


def generate_one(
    condition: str,
    charset: str,
    seed: Optional[int] = None,
    c3_weights: Optional[Dict[int, float]] = None,
    gen: Optional[UniversalCaptchaGenerator] = None,
    length: Optional[int] = None,
) -> Tuple:
    """Generate a single (image, label) for the given condition.

    The family charset is enforced via generate_text(custom_charset=...), overriding
    random_schema's internal charset selection. The schema's feature space, guardrails,
    and render pipeline are untouched.

    If ``length`` is given, it pins the captcha length (overriding random_schema's
    variable length_range) so the generated data matches the family's canonical length.
    This is required for the ablation's integrity: family lengths (cryptbb=5, darkbay=6,
    etc.) must be held constant per DATA_ABLATION_PLAN.md §1.

    If ``gen`` is None, a fresh ``UniversalCaptchaGenerator`` is constructed for this
    call (preserving standalone behavior). Pass a shared ``gen`` from a tight loop to
    avoid re-constructing it on every sample.
    """
    if gen is None:
        gen = UniversalCaptchaGenerator(seed=seed)

    if condition == "c1_random":
        d = sample_difficulty_c1()
        schema = random_schema(difficulty=d)
    elif condition == "c2_curriculum":
        sched = sample_difficulty_c2()
        schema = sched.get_schema()
    elif condition == "c3_adaptive":
        if c3_weights is None:
            raise ValueError("c3_adaptive requires c3_weights")
        sched = sample_difficulty_c3(c3_weights)
        schema = sched.get_schema()
    else:
        raise ValueError(f"unknown condition: {condition}")

    # Enforce family charset and length at text-generation time
    length_range = (length, length) if length is not None else schema.text.length_range
    label = generate_text(custom_charset=charset, length_range=length_range)
    img, _ = gen.generate(schema_obj=schema, text=label)
    return img, label


def generate_lmdb(
    condition: str,
    family: str,
    n: int = 100_000,
    seed: int = 1337,
    c3_weights: Optional[Dict[int, float]] = None,
    out_path: Optional[Path] = None,
) -> Path:
    """Generate n images and write them to an LMDB at the canonical path.

    Pairs are streamed into ``create_lmdb_dataset`` via a generator (no materialized
    list), so memory stays bounded regardless of ``n``. ``map_size`` is computed up
    front from ``n`` (matching ``create_lmdb_dataset``'s own formula) so the writer
    does not need to call ``len()`` on the generator.
    """
    charset = family_charset(family)
    length = FAMILY_LENGTH[family]
    out_path = out_path or family_lmdb_path(condition, family)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # One generator reused across all n samples (perf: avoids reconstructing it per
    # sample). gen_gateonly's generator is stateful only via the RNG it holds.
    gen = UniversalCaptchaGenerator(seed=seed)

    def _pair_gen():
        for i in range(n):
            yield generate_one(
                condition, charset, seed=seed + i,
                c3_weights=c3_weights, gen=gen, length=length,
            )
            if (i + 1) % 5000 == 0:
                print(f"  [{condition}/{family}] {i+1}/{n}")

    # Match create_lmdb_dataset's map_size formula but compute from n directly,
    # since _pair_gen is a generator (has no len()).
    map_size = max(2 * 1024 * 1024 * 1024, int(n * 30 * 1024))
    create_lmdb_dataset(_pair_gen(), str(out_path), map_size=map_size)

    # record provenance
    meta_path = out_path.parent / "meta.json"
    meta_path.write_text(json.dumps({
        "condition": condition, "family": family, "n": n, "seed": seed,
        "c3_weights": c3_weights, "charset_len": len(charset), "length": length,
    }, indent=2))
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Generate one ablation cell LMDB")
    ap.add_argument("--condition", required=True,
                    choices=["c1_random", "c2_curriculum", "c3_adaptive"])
    ap.add_argument("--family", required=True, choices=["cryptbb", "mgm", "darkbay", "city"])
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--c3-weights-json", type=str, default=None,
                    help="JSON file of tier weights (required for c3_adaptive)")
    args = ap.parse_args()

    c3_weights = None
    if args.condition == "c3_adaptive":
        if not args.c3_weights_json:
            ap.error("c3_adaptive requires --c3-weights-json")
        c3_weights = {int(k): v for k, v in json.loads(Path(args.c3_weights_json).read_text()).items()}

    path = generate_lmdb(args.condition, args.family, args.n, args.seed, c3_weights)
    print(f"✓ wrote {path}")


if __name__ == "__main__":
    main()
