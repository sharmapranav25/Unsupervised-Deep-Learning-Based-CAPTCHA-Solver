# CryptBB Model One A/B

Authorized13November2026. Run A then B sequentially from scratch,20epochs each. Preserve all training data and historical sources. No curriculum experiment or later reader starts here.

Source environment.ps1, then use the configured thesisPython:

- Capri/capri.py train-cryptbb-model1 prepare
- Capri/capri.py train-cryptbb-model1 smoke
- Capri/capri.py train-cryptbb-model1 coordinate
- To recover after verifying no live coordinator: same coordinate command with --resume.

Do not duplicate a live coordinator. Read coordinator.lock.json, active_child.json and status.json. Logs: train_A.log/train_B.log. Actual interpreter PID and optimizer-step counts are in model1/runs/cryptbb/{A,B}/progress.json. A STOP file here is checked at committed epoch boundaries and before B.

A successful readiness/receipt.json must match protocol.json. Each reader saves epoch_N.pth and epoch_N_receipt.json plus latest.pth/latest_receipt.json. Resume only a matching committed latest checkpoint. An interruption between archive creation and latest receipt may need manual reconciliation; no silent fallback. Outputs from incomplete epochs are discarded on resume. Selection remains pending after both20epoch runs; no real labels enter training. Prospective reporting uses the existing551-image raw A/B agreement policy, candidates2-20, earliest tie. Model1-B/Model1-A epoch20 remain available as fixed-curriculum teachers.

The existing SynthTIGER validation set belongs to DarkBay; no CryptBB validation set is invented. Training source hashes are frozen in protocol.json and copied under sources. Read completion.json to confirm both readers finished; a launched process is not completion.
