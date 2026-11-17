"""
Batch Generation CLI and LMDB Writer for Universal CAPTCHA Generator.

Provides command-line interface for mass generation of labeled CAPTCHA images
with optional LMDB output for direct SVTRv2 training integration.

All schemas are randomly sampled from the feature space — no hand-crafted
family presets are used.
"""

import os
import io
import sys
import random
import argparse
import time
from pathlib import Path
from typing import Optional, List

from PIL import Image

from .generator import UniversalCaptchaGenerator
from .curriculum import (
    CurriculumScheduler,
    pretraining_distribution, balanced_distribution,
    adversarial_distribution, hardened_distribution,
    very_easy_distribution,
)


# ═══════════════════════════════════════════════════════════
# LMDB writer
# ═══════════════════════════════════════════════════════════

def create_lmdb_dataset(
    image_label_pairs: List,
    lmdb_path: str,
    map_size: int = None,  # computed from sample count (was a hard 1 GB cap -> MDB_MAP_FULL at scale)
):
    """
    Write image-label pairs to an LMDB dataset compatible with SVTRv2 training.

    Format matches PaddleOCR's expected LMDB structure:
    - num-samples: total count
    - image-{idx}: PNG bytes
    - label-{idx}: text label
    """
    try:
        import lmdb
    except ImportError:
        print("ERROR: lmdb package not installed. Install with: pip install lmdb")
        print("LMDB creation skipped.")
        return False

    if map_size is None:
        map_size = max(2 * 1024 * 1024 * 1024, int(len(image_label_pairs) * 30 * 1024))
    os.makedirs(lmdb_path, exist_ok=True)
    env = lmdb.open(lmdb_path, map_size=map_size)
    cache = {}
    count = 0

    for i, (img, label) in enumerate(image_label_pairs):
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        img_bytes = buf.getvalue()

        image_key = f'image-{i+1:09d}'.encode()
        label_key = f'label-{i+1:09d}'.encode()

        cache[image_key] = img_bytes
        cache[label_key] = label.encode()
        count += 1

        if count % 1000 == 0:
            with env.begin(write=True) as txn:
                for k, v in cache.items():
                    txn.put(k, v)
            cache = {}
            print(f"  LMDB: written {count} entries...")

    if cache:
        with env.begin(write=True) as txn:
            for k, v in cache.items():
                txn.put(k, v)

    with env.begin(write=True) as txn:
        txn.put('num-samples'.encode(), str(count).encode())

    env.close()
    print(f"  LMDB: {count} entries written to {lmdb_path}")
    return True


# ═══════════════════════════════════════════════════════════
# Preview grid generator
# ═══════════════════════════════════════════════════════════

def generate_preview_grid(
    generator: UniversalCaptchaGenerator,
    output_path: str,
    num_rows: int = 7,
    samples_per_row: int = 3,
):
    """
    Generate a visual preview grid showing random samples at various difficulties.

    Args:
        generator: Generator instance.
        output_path: Path to save the preview image.
        num_rows: Number of difficulty levels to show.
        samples_per_row: Number of samples per difficulty level.
    """
    import numpy as np
    from PIL import ImageDraw, ImageFont

    rows = []
    target_w = 350
    target_h = 100
    difficulties = [i / max(num_rows - 1, 1) for i in range(num_rows)]

    for d in difficulties:
        row_images = []
        for _ in range(samples_per_row):
            try:
                img, label = generator.generate(difficulty=d)
                img = img.resize((target_w, target_h), Image.LANCZOS)

                labeled = Image.new('RGB', (target_w, target_h + 18), (255, 255, 255))
                labeled.paste(img, (0, 0))
                draw = ImageDraw.Draw(labeled)
                try:
                    font = ImageFont.truetype(
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
                except Exception:
                    font = ImageFont.load_default()
                draw.text((3, target_h + 2), f"d={d:.2f}: {label}",
                          fill=(0, 0, 0), font=font)
                row_images.append(np.array(labeled))
            except Exception as e:
                print(f"  Warning: Failed at d={d:.2f}: {e}")
                continue

        if row_images:
            row = np.hstack(row_images)
            rows.append(row)

    if rows:
        max_w = max(r.shape[1] for r in rows)
        padded = []
        for r in rows:
            if r.shape[1] < max_w:
                pad = np.full((r.shape[0], max_w - r.shape[1], 3), 255, dtype=np.uint8)
                r = np.hstack([r, pad])
            padded.append(r)

        grid = np.vstack(padded)
        Image.fromarray(grid).save(output_path)
        print(f"  Preview saved to {output_path}")
    else:
        print("  Warning: No images generated for preview")


# ═══════════════════════════════════════════════════════════
# Main CLI
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Universal CAPTCHA Generator - Batch Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 1000 random CAPTCHAs
  python -m universal_captcha_generator.batch_generate --num 1000 --output-dir ./synth_captchas

  # Generate at a specific difficulty
  python -m universal_captcha_generator.batch_generate --num 500 --difficulty 0.7 --output-dir ./hard_data

  # Generate with curriculum scheduling (easy→hard progression)
  python -m universal_captcha_generator.batch_generate --num 5000 --mode progressive --output-dir ./curriculum

  # Generate with LMDB output for SVTRv2
  python -m universal_captcha_generator.batch_generate --num 10000 --lmdb --output-dir ./training_data

  # Generate preview grid at various difficulties
  python -m universal_captcha_generator.batch_generate --preview --output-dir ./preview
        """,
    )

    parser.add_argument('--num', type=int, default=1000,
                        help='Number of images to generate (default: 1000)')
    parser.add_argument('--output-dir', type=str, default='./synth_captchas',
                        help='Output directory for images')
    parser.add_argument('--difficulty', type=float, default=None,
                        help='Difficulty level 0.0-1.0 for all images')
    parser.add_argument('--tier', type=int, default=None, choices=[1, 2, 3, 4, 5],
                        help='Curriculum tier (1=easy, 5=adversarial)')
    parser.add_argument('--mode', type=str, default='random',
                        choices=['random', 'curriculum', 'progressive'],
                        help='Generation mode')
    parser.add_argument('--distribution', type=str, default='balanced',
                        choices=['balanced', 'pretraining', 'very_easy', 'adversarial', 'hardened'],
                        help='Curriculum distribution')
    parser.add_argument('--lmdb', action='store_true',
                        help='Also write LMDB dataset for SVTRv2 training')
    parser.add_argument('--preview', action='store_true',
                        help='Generate a visual preview grid at various difficulties')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set seed
    if args.seed is not None:
        random.seed(args.seed)

    # Create generator
    print("Initializing Universal CAPTCHA Generator...")
    gen = UniversalCaptchaGenerator(seed=args.seed)
    print(gen.info())

    # Configure curriculum distribution
    dist_map = {
        'balanced': balanced_distribution,
        'pretraining': pretraining_distribution,
        'very_easy': very_easy_distribution,
        'adversarial': adversarial_distribution,
        'hardened': hardened_distribution,
    }
    if args.mode == 'curriculum':
        gen._curriculum = CurriculumScheduler(
            tier_distribution=dist_map[args.distribution]()
        )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preview mode
    if args.preview:
        preview_path = output_dir / 'preview_grid.png'
        generate_preview_grid(gen, str(preview_path))
        print(f"\nPreview grid saved to {preview_path}")
        return

    # Batch generation
    print(f"\nGenerating {args.num} CAPTCHAs...")
    print(f"  Output: {output_dir}")
    print(f"  Mode: {args.mode}")
    if args.difficulty is not None:
        print(f"  Difficulty: {args.difficulty}")
    if args.tier is not None:
        print(f"  Tier: {args.tier}")
    print()

    start_time = time.time()
    results = []
    seen_labels = set()

    for i in range(args.num):
        try:
            if args.mode == 'progressive':
                progress = i / max(args.num - 1, 1)
                s = gen.curriculum.get_progressive_schema(progress)
                img, label = gen.generate(schema_obj=s)
            elif args.difficulty is not None:
                img, label = gen.generate(difficulty=args.difficulty)
            elif args.tier is not None:
                img, label = gen.generate(tier=args.tier)
            else:
                img, label = gen.generate()

            # Ensure unique filenames
            save_label = label
            counter = 1
            while save_label in seen_labels:
                save_label = f"{label}_{counter}"
                counter += 1
            seen_labels.add(save_label)

            # Save image
            img_path = output_dir / f"{save_label}.png"
            img.save(img_path)
            results.append((img, label))

            if (i + 1) % 100 == 0 or i == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (args.num - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{args.num}] {label} | "
                      f"{rate:.1f} img/s | ETA: {eta:.0f}s")

        except Exception as e:
            print(f"  ERROR at {i+1}: {e}")
            continue

    elapsed = time.time() - start_time
    print(f"\n✓ Generated {len(results)}/{args.num} images in {elapsed:.1f}s "
          f"({len(results)/elapsed:.1f} img/s)")

    # LMDB output
    if args.lmdb:
        lmdb_path = str(output_dir / 'lmdb')
        print(f"\nWriting LMDB dataset to {lmdb_path}...")
        create_lmdb_dataset(results, lmdb_path)

    # Summary
    print(f"\n{'='*60}")
    print(f"Generation Complete")
    print(f"  Images: {len(results)}")
    print(f"  Directory: {output_dir}")
    if args.lmdb:
        print(f"  LMDB: {output_dir / 'lmdb'}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
