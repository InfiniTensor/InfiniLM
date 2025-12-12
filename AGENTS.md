# Repository Guidelines

## Project Structure & Module Organization
- `src/` C++17 inference core: `models/` (DeepSeek, Jiuge, LLaVA variants), `tensor/`, `allocator/`, `dataloader/`, `cache_manager/`, plus shared headers (`tensor.hpp`, `utils.hpp`).
- `include/` Public API headers exported by the shared library (`infinicore_infer.h` and subheaders).
- `scripts/` Python entrypoints for running models, launching servers, and validating performance (e.g., `jiuge.py`, `launch_server.py`, `test_perf.py`, `test_ppl.py`, `test_ceval.py`).
- `build/` xmake build outputs; keep clean in commits. `requirements.txt` lists Python deps used by the scripts.

## Build, Test, and Development Commands
- Configure InfiniCore: install upstream InfiniCore and set `INFINI_ROOT` (defaults to `$HOME/.infini`). Ensure headers/libs available under `$INFINI_ROOT/include` and `$INFINI_ROOT/lib`.
- Build & install library: `xmake && xmake install` (produces `libinfinicore_infer.so` under `$INFINI_ROOT/lib`).
- Run single-shot inference: `python scripts/jiuge.py --cpu path/to/model_dir [n_device]` (switch `--nvidia`, `--ascend`, etc. per target).
- Serve models: `python scripts/launch_server.py --model-path MODEL_PATH --dev {cpu,nvidia,...} --ndev N --max-batch N --max-tokens N`.
- Perf & PPL checks: `python scripts/test_perf.py` and `python scripts/test_ppl.py --model-path MODEL_PATH [--ndev N]`.

## Coding Style & Naming Conventions
- Language: C++17 with `set_warnings("all", "error")`; keep builds warning-free.
- Files and headers use snake_case; types are PascalCase; functions/methods lean camelCase; constants/macros are upper snake (`RUN_INFINI`).
- Prefer 4-space indentation, early returns, RAII for handles/streams, and keep ownership explicit via `std::shared_ptr` where used.
- Match existing include order: local headers first, then STL. Keep headers self-contained with include guards.

## Testing Guidelines
- Run relevant scripts before opening a PR: smoke (`python scripts/jiuge.py --cpu ...`), perf (`python scripts/test_perf.py`), and PPL (`python scripts/test_ppl.py`) on the device types you touched.
- For new kernels/backends, extend existing tests in `scripts/`; prefer deterministic seeds when possible.
- Capture command outputs/logs for reviewers, noting device count and model snapshot used.

## Commit & Pull Request Guidelines
- Commits in history are concise, action-oriented (often Chinese summaries). Follow suit: short present-tense subject, include scope (e.g., backend/feature touched).
- For PRs, include: summary of changes, target hardware/backend, required `INFINI_ROOT` setup, and test commands/results. Attach logs or screenshots for user-visible behavior.
- Avoid committing build artifacts, large checkpoints, or generated debug data; keep diffs minimal and scoped.

## Security & Configuration Tips
- Keep `INFINI_ROOT` writable and isolated; avoid hardcoding absolute paths in code or scripts.
- Do not check in proprietary model weights or third-party binaries. If sharing sample data, scrub sensitive contents and document the source.
