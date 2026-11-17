> UPDATE: User subsequently requested stopping the run. The fit, observer and automatic pipeline have been stopped; partial outputs are preserved. See cryptbb_v2_20261114/stop_receipt.json. The running-process descriptions below are historical. Do not restart without new authorization.

# Claude handoff: investigate CryptBB controller-v2 fitting runtime

User request: "hand off for claude to figure out why its taking so long".

## Task and scope

Diagnose the runtime of the currently running CryptBB v2 controller fit. Explain where time goes and recommend evidence-backed speedups. The user already authorized new datasets A/B and then concurrent Model1-A/B training; the automatic pipeline is running. This handoff does not cancel it or authorize discarding progress. Start with read-only inspection; isolate profiling from the production run. Do not silently reduce search coverage, reference count, seeds, validation, image quality checks, or training budget to make it faster.

Workspace: E:\pranav\Thesis-Pranav-Final. Read AGENTS.md, README.md, handoff/CURRENT_STATE.md (newest entry first), and Capri/docs/CODE_AND_MODELS.md before workflow changes. The original E:\pranav\CAPRI is not the active study workspace. Source environment.ps1 for the configured Python and E-drive caches.

## Snapshot taken November 13 evening PST / November 14 UTC

- Full fit: Capri/runs/cryptbb_controller_v2_free_full_20261114.
- Latest observed status: 24/100 references completed in 3927.25 seconds (65.45 minutes), averaging 163.64 seconds/reference. Early 12-reference average was 157.82 seconds/reference. This is ongoing progress, not evidence of a hang.
- Extrapolation only: about 4.55 hours total fitting, about 3.45 hours remaining at this snapshot. This excludes corpus generation, packing, probes, and training.
- entries.json contains 96 selected entries: 24 each custom, dino, mixed, synthtiger. None of these 96 is flagged unvalidated.
- Average candidates_tried: CAPRI engine 210.25/reference; SynthTIGER engine 132.125/reference. The CAPRI count is repeated on the three CAPRI-arm entries: do NOT sum it three times. Approx. 342.375 distinct engine/configuration candidates and 1027.125 fitting renders/reference before fresh validation and comparison/sample renders.
- One instantaneous nvidia-smi snapshot: RTX A4000, 0% GPU utilization, 5466 MiB / 16376 MiB used. This is a momentary observation, NOT proof the GPU is always idle or the bottleneck.
- Workflow status: controller_fit / waiting_for_completion. New A/B production generation and readers have not started.

Timing log (cumulative seconds): reference 1=173, 4=674, 8=1271, 12=1894, 16=2561, 20=3257, 24=3927. Full log is in the study directory below.

## Running processes and safety

Study directory: handoff/cryptbb_v2_20261114.

Launch receipts record these identities; verify PID AND creation time before any action because PIDs can be reused:
- Fit launcher process: PID 137140, creation time 1794632645.2347612. Windows venv Python may have an Anaconda child executing the work; inspect the process tree.
- Fit observer: PID 135988, creation time 1794633128.399861.
- Workflow coordinator: PID 59988, creation time 1794633218.2868235.

Relevant receipts/logs:
- fit_status.json, fit_launch.json, workflow_launch.json, workflow_status.json
- controller_fit.log, fit_observer.log, workflow.log
- authorization.json, run_protocol.json, fit_sources.json, workflow_sources.json
- monitor_repair.json, observe_fit.py

The original start_fit.py monitor had an incorrect completion-schema assumption and was replaced WITHOUT restarting the fit. Do not relaunch start_fit.py. Current observe_fit.py verifies final status state=complete, 100 input IDs, 400 entries, and 100 history records. fit_completion.json is the gate consumed by the workflow.

The live code and downstream inputs are hashed/frozen. Editing source while jobs run can produce mixed provenance or block the workflow. If an optimization requires a new run, first assess retained progress and coordinate a versioned replacement with updated receipts/protocols; do not bypass hash gates. The fit has no CLI resume path: it refuses an existing output directory. entries.json and status.json are written per reference, but controller_history.json and sample PNGs are written only at the end; histories/samples accumulate in memory. Killing it can lose evidence needed by the completion gate even though selected entries survive.

## Code paths and likely profiling targets

Maintained source: Capri/src/capri_controller/controller_v2.py.

Observed architecture (facts from source, not measured timing attribution):
- run(), approximately lines 154-335: one parent Mixed feature extractor and ProcessPoolExecutor(max_workers=14, initializer=_install).
- References are sequential in `for index in chosen_inputs` (around line 208). Fourteen workers do not mean fourteen references run concurrently.
- evaluate(), lines 193-207: deduplicate candidate controls, blocking pool.map(render_job, ..., chunksize=4), collect all rendered arrays, parent model.extract(imgs), then global and local objective scoring per candidate.
- controllers.py: Mixed.extract calls custom.extract_pair(images)[1] followed by dino.extract(images). Inspect features/custom.py and features/dino.py for batching, device placement, preprocessing, and thread settings.
- Initial candidate groups are evaluated separately for both capri and synthtiger engines. Search includes font, geometry, colors, backgrounds, surface effects, merged controls, and up to three refinement rounds.
- Three fitting seeds per candidate; fresh validation uses four seeds. Top candidates are validated through separate small pool.map calls (around lines 283-296).
- Original v1 choices are re-rendered/re-scored for comparison (lines 314-317); fresh display samples are also rendered (318-322).
- SynthTIGER glyph rendering is in synthtiger_recipe_adapter.py, called by render_job. Check repeated construction/font loading/caching and per-render costs.

Please measure or estimate separately:
1. CAPRI versus SynthTIGER rendering wall time and per-image distribution, including slow effects/font cases.
2. Worker occupancy, Windows process/pickling/image-array overhead, chunksize load balance, native-library thread oversubscription, memory pressure.
3. Custom feature extraction, DINO preprocessing/forward passes, and objective scoring versus rendering wait time.
4. Repeated small validation batches and synchronization overhead; feature reuse opportunities that preserve exact semantics.
5. Candidate counts and variation by reference; whether expected search volume alone explains the runtime.
6. GPU utilization over time, not a single snapshot. Avoid competing heavy benchmarks on the production GPU.

Prefer short representative isolated profiles with the identical controls, seeds, engines, and objectives. Report any profiling contention. Distinguish semantic-preserving engineering improvements from a smaller/different experiment. Do not present speculative bottlenecks as measured findings.

## What happens after fitting

cryptbb_v2_workflow.py -> cryptbb_v2_corpus.py -> training/cryptbb_v2_model1.py (all under Capri/src/capri_controller).

- Require all 400 validated controller entries. Any unvalidated fallback blocks production export.
- Generate and pack A, then B: 100,000 images each, 250 per entry, 25,000 per arm. B excludes A pixels; audit also excludes original corpora and real-source pixels. No real labels enter generation or fitting.
- Output datasets/cryptbb_v2/{A,B}; original corpora unchanged.
- Actual-data paired readiness probes, then both readers start concurrently at model1/runs/cryptbb_v2/{A,B}.
- Same scratch seed48, batch128, corrected32x128 shape, 20 epochs. Two resident reader processes alternate complete GPU batches under a Windows lock to fit the 16GB GPU; this lock is for later training, not the current controller fit.
- Prior paired smoke on OLD corpora passed exact checkpoint-next-update replay and full sampler replay for both readers. Evidence: existing_data_parallel_smoke/summary.json. Actual new-data probes remain pending.
- Old CryptBB Model1-B remains canceled; new v2 B is authorized. No Model2/3/4 launch.

## Requested return

Give the user a clear explanation of the slowdown, measured time breakdown where possible, whether this run should simply finish, and ranked optimizations with expected benefit and impact on scientific comparability. Record any changes and evidence in handoff/CHANGELOG.md and Capri/docs/MAINTENANCE_LOG.md. Preserve the running study and its provenance.
