# Shared thesis workspace rules

Read README.md and handoff/CURRENT_STATE.md first. All work, outputs, caches and temporary project files stay on E:. Do not use private assistant workspaces for thesis artifacts. Record changes in handoff/CHANGELOG.md and maintain source/evidence mappings.

Maintain shared Python implementation in Capri/src/capri_controller, selected generator models in Capri/models/registry.json and machine paths in runtime/local.json and Capri/configs/local.json. Model-specific adapters belong in model1/2/3/src. Original protocol copies and historical evidence remain unchanged. New model runs go under the appropriate modelN/runs/<descriptive-name>; generator runs under Capri/runs. Historical scripts need path/protocol adaptation before launching.

Preserve the pixel-only target workflow and custom/DINO/mixed controller mixture. No VLMs, human target strings or manually chosen family lengths enter generation. Reuse A/B; do not regenerate or duplicate datasets inside reader folders. Model1-B ranks A; Model1-A ranks B. Model2 keeps that ranking fixed; Model3 starts there and adapts using synthetic training labels. One run per model/dataset combination, not repeated seed trials.

Both Model1 runs completed20epochs and the coordinator stopped as requested. Migration is not authorization to start Model2/3. Freeze the new curriculum, sampling budget and selection protocol before launching. Do not silently use the old95k/5k split or historical three-run launchers. Match the100k/20epoch Model1 budget; adaptive rescoring overhead must be reported separately.

Real DarkBay is a repeatedly inspected historical test, not untouched validation. Freeze predictions before scoring; never use real labels to steer curriculum or checkpoint selection. For shoot-rate policies, learn confusions and attempt/skip rules on synthetic validation, freeze policy and real decisions, then score real labels. Clearly distinguish historical real confidence sweeps. Requested diagnostics do not change training.

Do not delete original source experiments or model files without explicit authorization. Report what was verified and remaining dependencies honestly. Source-map links can contain old C: strings as immutable provenance; current execution paths and write destinations must use E:.
