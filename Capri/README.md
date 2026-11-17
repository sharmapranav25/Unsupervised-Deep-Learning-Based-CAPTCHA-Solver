# CAPRI

The selected thesis generator is models/capri/checkpoint.pt (epoch20 EMA). It is the lightweight3.66M core plus fine-tuned DINO font branch and independent count transformer. Generation uses internal count and trained font outputs; the separate counter is not used in the renderer flow.

models/controller_dependencies contains custom encoder, text mask, original DINO controller weights, and a legacy hybrid needed for original calibration. That hybrid is a calibration dependency, not a second thesis baseline. The historical generic pipeline resolves that hybrid name; therefore do not run the copied capri.py generation/training commands without adapting them to selected_generator and the new paths. This migration preserves source behavior rather than silently routing to the wrong model.

See models/registry.json, architecture.md, training_setup.md and baseline_selection.md. Original source/config snapshots are in handoff/original_configuration and runs/generator_training. Shared reader curriculum implementations remain under src/capri_controller/training. Preserve their historical protocol provenance; create new A/B adapters before training.
