# Shared implementation

The current A/B adapter is train_darkbay.py, backed by Capri/src/capri_controller/training/ab_curriculum.py. Historical ds2.py is preserved and must not launch the new study. The frozen protocol, readiness receipt and coordinator status live in handoff/darkbay_curriculum_20261113. Run the study with Capri/capri.py darkbay-ab-curriculum coordinate after setup; the coordinator gates production on checks. Direct adapters also require passed readiness. Model1-B ranks A; Model1-A ranks B. Outputs: model2/rankings/darkbay/{A,B} and model2/runs/darkbay/{A,B}.
