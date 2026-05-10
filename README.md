# my-ai-quality-check-demo
# my-ai-quality-check-demo

This repository was reorganized to group core functionality into a small
package under `app/core/` to improve readability and maintainability.

NOTE: All dataset files and model weight files (including YOLO weights) have
been preserved in-place and were not deleted.

## Current layout

Key implementation now lives under `app/core/`:

- `app/core/config.py`         - core configuration (camera list, thresholds, aggregation params)
- `app/core/inference.py`      - EfficientAD-focused inference wrapper (placeholder until you export a model)
- `app/core/aggregator.py`     - per-shaft aggregation logic
- `app/core/workers.py`        - camera & inference worker threads (use `ModelInfer`)
- `app/core/main_window.py`    - GUI MainWindow implementation (PySide6)
- `app/core/pipeline_demo.py`  - offline simulation/demo runner

Top-level helper/entry files:

- `main.py`                    - small launcher that calls `app.entry.main()` (GUI entry)
- `train_model.py`             - training guidance / placeholder (EfficientAD direction)
- `requirements.txt`           - dependencies
- `README.md`                  - this file

Notes:
- The old top-level modules (e.g. `inference.py`, `aggregator.py`, `workers.py`,
  `main_window.py`, `pipeline_demo.py`, `config.py`) were removed to enforce
  the new package layout. Please update imports to use `app.core.*` modules.

## Quick start

1. Create and activate a virtual environment, install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the offline pipeline demo (uses images under `datasets/metal_defects/train/images`):

```bash
python3 -c "from app.core.pipeline_demo import simulate_run; simulate_run(num_shafts=10, frames_per_shaft=1)"
```

3. Start the GUI (requires a graphical environment):

```bash
python3 main.py
```

## Important notes

- `app/core/inference.py` is currently a placeholder that returns safe
  dummy scores (0.0) if no EfficientAD runtime model is configured. To run
  actual anomaly detection, export your EfficientAD model (TorchScript or
  ONNX), place it at the path configured in `app/core/config.py` or update
  that setting, and I will help integrate a loader using `onnxruntime` or
  TorchScript + MPS.

- `app/core/aggregator.py` is independent from `workers.py` per your
  instruction; I did not automatically integrate them. You can wire
  `Aggregator` into the worker threads when you are ready.

## Next steps (optional)

- Implement EfficientAD runtime loader in `app/core/inference.py` and
  integrate ONNX/TorchScript inference.
- Integrate `Aggregator` into real-time `workers` to produce shaft-level
  decisions and optionally persist NG snapshots.
- Add a light-weight vision-based shaft-id mapping module if your cameras
  cannot attach shaft identifiers to frames.

If you want me to implement any of the next steps, tell me which one and I
will proceed and run local tests.

