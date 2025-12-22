# Repository Guidelines

## Project Structure & Module Organization
- `src/`: C++ inference runtime and model implementations (e.g., `src/models/jiuge/`, `src/models/llava/`).
- `include/`: Public C/C++ headers exported to `INFINI_ROOT` on install (e.g., `include/infinicore_infer/models/*.h`).
- `scripts/`: Python entry points, ctypes bindings (`scripts/libinfinicore_infer/`), and smoke/perf scripts.
- `docs/`: Engineering notes and model adaptation plans (see `docs/Minicpmv.md`).
- `Fastcache/` (optional): separate Python benchmarking utilities for KV-cache compression.
- `build/` and `.xmake/`: generated build artifacts (do not edit by hand).

## Build, Test, and Development Commands

Prereq: build and install `InfiniCore`, then set `INFINI_ROOT` (default: `$HOME/.infini`).

- Build + install the shared library and headers:
  - `xmake && xmake install`
- Clean rebuild:
  - `xmake clean && xmake`
- Python deps (minimal list is in `requirements.txt`):
  - `pip install -r requirements.txt`
  - Common extras used by scripts: `pip install torch safetensors fastapi uvicorn janus`
- Run a local inference smoke test:
  - `python scripts/jiuge.py --cpu path/to/model_dir`
- Launch the local inference server:
  - `python scripts/launch_server.py --model-path path/to/model_dir --dev cpu --ndev 1`
- Basic perf / ppl scripts:
  - `python scripts/test_perf.py`, `python scripts/test_ppl.py --model-path path/to/model_dir`

## Coding Style & Naming Conventions

- C++: format with `clang-format` using the repo’s `.clang-format` (LLVM-based, 4-space indent). Target is C++17; warnings are treated as errors in `xmake.lua`.
- Naming: files and folders use `snake_case` (e.g., `jiuge_awq_weight.cpp`); new models typically live under `src/models/<model_name>/` with matching headers in `include/infinicore_infer/models/`.
- Keep public API changes compatible: if you add new headers, ensure they are installed by the build (`xmake.lua` install rules).

## Testing Guidelines

- No single test runner is enforced; treat `scripts/test_*.py` and lightweight model runs as validation.
- For changes to model execution, run at least one end-to-end path (e.g., `scripts/jiuge.py`) on your target backend (`--cpu`, `--nvidia`, etc.).

## Commit & Pull Request Guidelines

- Commit messages commonly use short subjects; prefer `feat:` / `fix:` style when applicable, and include issue references when relevant (e.g., `issue/64 - …`).
- PRs: describe what changed, how to run/verify (exact commands), target device/backend tested, and link related issues. Add screenshots/log snippets for user-facing changes (server/API output).
