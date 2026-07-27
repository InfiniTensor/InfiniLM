# ERNIE-4.5-VL verification

This note contains portable verification commands for
`ERNIE-4.5-VL-28B-A3B-Thinking`. It deliberately uses environment variables
and repository-relative output paths so it can be shared without exposing
machine-specific accounts, hosts, credentials, or storage layouts.

## Prerequisites

Build and install InfiniCore and InfiniLM following their repository
instructions. Use a Python environment that provides PyTorch, Transformers,
and the dependencies required by the selected benchmark dataset.

From the InfiniLM repository root, set the model directory and create a local
result directory:

```bash
export MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR to the downloaded model directory}"
mkdir -p artifacts/ernie_vl
```

The commands below assume four visible CUDA devices. Adjust `--tp` and
`--tp-devices` only when the model fits the selected hardware configuration.

## Tensor-parallel correctness

Run deterministic text, image, and video checks against the token IDs embedded
in the verification script:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/ernie_vl_correctness.py \
  --model "$MODEL_DIR" \
  --tp 4 \
  --tp-devices 0,1,2,3 \
  --cases text image video \
  --max-cache-len 512 \
  --reference-mode expected \
  --output-json artifacts/ernie_vl/correctness_tp4.json
```

Acceptance criteria:

- the top-level `ok` field is `true`;
- `text`, `image`, and `video` all report `match_expected=true`;
- the process exits successfully.

To generate a live Transformers reference in a separate process, run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/ernie_vl_correctness.py \
  --model "$MODEL_DIR" \
  --tp 4 \
  --tp-devices 0,1,2,3 \
  --cases text image video \
  --max-cache-len 512 \
  --reference-mode hf \
  --hf-device-map auto \
  --hf-torch-dtype bf16 \
  --output-json artifacts/ernie_vl/correctness_tp4_hf.json
```

This mode requires enough memory for the Transformers reference as well as the
InfiniLM run. Its acceptance criteria are the same as the deterministic check.

## Messages API smoke tests

Static-cache smoke test:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/ernie_vl_llm_smoke.py \
  --model "$MODEL_DIR" \
  --tp 4 \
  --cache-type static \
  --output-json artifacts/ernie_vl/llm_static_tp4.json
```

Paged-cache smoke test:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/ernie_vl_llm_smoke.py \
  --model "$MODEL_DIR" \
  --tp 4 \
  --cache-type paged \
  --num-blocks 64 \
  --block-size 16 \
  --output-json artifacts/ernie_vl/llm_paged_tp4.json
```

Each output JSON should contain `ok=true`.

## C-Eval and MMLU

The standard benchmark entry point can evaluate the C++ backend. The following
commands run the complete validation split and write repository-local CSVs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python test/bench/test_benchmark.py \
  --device nvidia \
  --model "$MODEL_DIR" \
  --bench ceval \
  --subject all \
  --split val \
  --max-new-tokens 5 \
  --backend cpp \
  --tp 4 \
  --dtype bfloat16 \
  --output-csv artifacts/ernie_vl/ceval_val_tp4.csv

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python test/bench/test_benchmark.py \
  --device nvidia \
  --model "$MODEL_DIR" \
  --bench mmlu \
  --subject all \
  --split val \
  --max-new-tokens 5 \
  --backend cpp \
  --tp 4 \
  --dtype bfloat16 \
  --output-csv artifacts/ernie_vl/mmlu_val_tp4.csv
```

Dataset access may require an existing local cache or outbound network access.
Report the sample count, accuracy, hardware, dtype, and tensor-parallel degree
together; results from different configurations are not directly comparable.

## MMMU adaptation smoke test

`ernie_vl_mmmu_smoke.py` is an adaptation check, not an official leaderboard
submission. A small C++ backend run can be launched with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/ernie_vl_mmmu_smoke.py \
  --model "$MODEL_DIR" \
  --backend cpp \
  --subjects Accounting \
  --split validation \
  --num-samples 2 \
  --max-new-tokens 64 \
  --prompt-style official \
  --tp 4 \
  --tp-devices 0,1,2,3 \
  --output-csv artifacts/ernie_vl/mmmu_smoke_tp4.csv \
  --output-json artifacts/ernie_vl/mmmu_smoke_tp4.json
```

For a fair backend comparison, keep the dataset split, subjects, prompt style,
sample count, image size, and generation limit identical.

## Publishing evidence

Before attaching logs or screenshots to a pull request:

- show the command, exit status, and final result summary;
- remove terminal history, login banners, hostnames, addresses, credentials,
  cache locations, and machine-specific absolute paths;
- do not publish model access tokens or dataset credentials;
- label smoke-test and partial-dataset results as such.
