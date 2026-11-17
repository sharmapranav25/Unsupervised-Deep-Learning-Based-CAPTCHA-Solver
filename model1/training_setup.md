# Model 1: standard PARSeq

Completed one reader on A and one on B, each100000 synthetic images and20epochs. SVTRv2LNConvTwo33 encoder with PARSeq decoder, scratch initialization, batch128, AdamW LR0.00035, weight decay0.05, OneCycleLR with2warmup epochs, BF16 training. Fixed final epoch20 selection, no validation or real-score selection. Saved seed48 is a reproducibility value, not a plan for repeated trials.

Exact historical configs: configs/darkbay/A,B/original.yml. Relocated configs point at shared datasets and fresh reproduction destinations; they are not instructions to retrain. Final checkpoints: models/parseq/darkbay/A,B. Other saved epochs and training logs: runs/darkbay/A,B. Results include requested15/17/20epoch diagnostics and older experiments explicitly under historical/.

Real evaluation:900 RGB-hash-verified historical DarkBay images excluding100 controller inputs; RGB bicubic32x128, normalization0.5, saved autoregressive decoding with one refinement, no forced length or narrowed charset. Freeze predictions before labels. A615/900; B573/900. Use src/verify_migrated_readers.py to validate migrated inference against frozen predictions without scoring labels.
