# Training setup

Completed joint training:100000 synthetic scenes,20epochs, original lightweight core plus fine-tuned DINO font branch plus separate count transformer. Final epoch20 EMA selected. Logical batch128 with32-image microbatches and pixel-budget accumulation; AdamW base LR0.0005, DINO LR0.000005, weight decay0.05, warmup300steps, cosine-style original schedule and minimum LR0.00001, EMA0.999, BF16. Consult the saved protocol for exact behavior rather than reconstructing from this summary.

Training scenes and font-reference metadata: training_data/scenes_and_font_references. Actual prepared training cache: training_data/prepared_cache. Frozen source snapshots, metrics and checkpoint history: runs/generator_training. Final checkpoint lives once in models/capri. Original paths in cache metadata are provenance; future training must use a relocated adapter and fresh output directory.
