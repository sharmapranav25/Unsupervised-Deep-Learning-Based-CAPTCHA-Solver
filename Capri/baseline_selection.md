# Selected thesis baseline

The author specified one CAPRI baseline and preferred the newer generator if it beat the previous one. The completed matched Model1-A reader comparison supports selecting the new joint20 checkpoint for this thesis workspace:615/900 (68.33%) versus the previous generator reader492/900 (54.67%), same single-run initialization setting and final20epochs on100k generated images. B is separately573/900 (63.67%); A was the predesignated comparison, not selected as the better A/B result.

Checkpoint SHA256:28f95fd8daf3831fb860ab59e15b8b65bd4a68bacc434eb7eb7a296e6190b764. The original workspace registry is preserved unchanged; the final workspace selected_generator record identifies this baseline. The copied generic old pipeline is not automatically adapted to it.

This is evidence on one repeatedly inspected DarkBay test set. It does not establish a font-only causal improvement or superiority on every scheme. Current reader evaluation used CPU FP32; the historical comparison used GPU FP32. Original comparison and protocols are retained in model1/results and handoff/study_records.
