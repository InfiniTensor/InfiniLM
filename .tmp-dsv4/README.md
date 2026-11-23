# DeepSeek V4 Flash W8A8 Ascend reproduction

This branch saves the stable 2026-11-16 DeepSeek V4 Flash W8A8 work for Ascend 910C.

## Pull

```bash
git clone --branch w8a8-dsv4-ascend --recurse-submodules git@github.com:InfiniTensor/InfiniLM.git
cd InfiniLM
```

## Environment

The saved run assumes the node245/pepe environment:

- 8 Ascend 910C devices
- CANN 9.1.0
- Python 3.12.13 and xmake
- Model at `/home/wuwei/models/Eco-Tech_DeepSeek-V4-Flash-w8a8-mtp`
- Custom transformer operators under `/vllm-workspace/vllm-ascend`

Override the model and dependency locations with `DSV4_MODEL_PATH`, `INFINI_ROOT`, and `INFINI_ASCEND_CUSTOM_OPAPI_ROOT`.

## Build and run

```bash
cd .tmp-dsv4
./build_dsv4_stack.sh
./build_infinilm_current.sh
./run_hetero_nondet_msprof.sh
```

For an existing InfiniRT/InfiniCCL installation, set `INFINI_ROOT` and run `./build_dsv4_ops.sh` instead of rebuilding the full stack.

## Benchmark

Run one case:

```bash
python3 .tmp-dsv4/bench_latency.py --url http://127.0.0.1:18081 --engine infinilm --conc 16 --input 4096 --output 4096 --timeout 21600
```

The five saved cases are:

```text
4  1024 1024
4  4096 4096
16 128  128
16 1024 1024
16 4096 4096
```

For vLLM comparison, start `.tmp-dsv4/run_vllm_server.sh` on port 18082 and run `.tmp-dsv4/run_vllm_bench.sh`.
