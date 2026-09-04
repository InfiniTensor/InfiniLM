# GGUF Route B for Qwen3.5

## Overview

This integration runs selected GGUF block-quantized weights directly from an
InfiniLM checkpoint. The converter copies supported GGUF block bytes into
safetensors as `uint8` tensors and records their GGML type in
`quantization_config.ggml_types`. InfiniLM resolves each weight by checkpoint
name and dispatches it to InfiniCore's `linear_gguf` operator.

The design keeps model-specific mapping in Python while making the C++ packed
Linear path reusable by other model integrations.

## Supported scope

- Model profile: Qwen3.5 / Qwen3.8 27B Route B mapping.
- Native packed Linear types: `Q8_0`, `Q4_K`, `Q5_K`, and `Q6_K`.
- Dense BF16 fallback for parameters without a native packed execution path,
  including embeddings, output head, normalization/scalar tensors, and IQ4
  tensors.
- NVIDIA execution through InfiniCore `linear_gguf`.
- Decode/small-batch and prefill execution paths selected inside InfiniCore.

The current implementation intentionally rejects tensor parallelism for packed
GGUF weights. It also does not provide native IQ4, packed embedding, or packed
output-head kernels.

## Dependency

The InfiniLM changes require the corresponding InfiniCore `linear_gguf`
operator and its supported GGML block decoders:

- InfiniCore pull request: https://github.com/InfiniTensor/InfiniCore/pull/1545

Build and install that InfiniCore revision before building InfiniLM.

## Checkpoint format

Packed Linear weights use a `weight_bytes` suffix and shape
`[out_features, row_bytes]`, where:

```text
row_bytes = in_features / block_size * type_size
```

The converter writes a top-level configuration:

```json
{
  "quantization_config": {
    "quant_method": "gguf",
    "key_prefix": "model.language_model.",
    "ggml_types": {
      "model.language_model.layers.0.mlp.gate_proj.weight_bytes": 14,
      "model.language_model.layers.0.input_layernorm.weight": "dense_bf16"
    },
    "activation_vperm": []
  }
}
```

`ggml_types` keys are exact safetensors parameter names. Values are GGML type
ids or `"dense_bf16"`. Runtime lookup requires exactly one packed or dense
candidate and fails on missing or ambiguous metadata.

## Conversion

The converter depends on Python packages used by the project plus
`gguf-py`. Install `gguf-py` or point `LLAMA_CPP_DIR` at a llama.cpp
checkout:

```bash
export LLAMA_CPP_DIR=/path/to/llama.cpp
python3 scripts/gguf_to_infinilm.py \
  --gguf /path/to/model.gguf \
  --out /path/to/infinilm-checkpoint \
  --tokenizer-dir /path/to/tokenizer-config \
  --verify sample
```

`--tokenizer-dir` is optional. Vocabulary and merges are exported from GGUF;
the directory only supplies auxiliary files such as
`tokenizer_config.json` and a chat template.

Useful options:

- `--dry-run`: validate metadata, shapes, orientation, and packed row sizes
  without writing tensors.
- `--layers N`: create a loadable checkpoint containing the first N layers.
- `--verify {off,sample,all}`: control post-write verification.
- `--skip-pack`: keep existing tensor shards while refreshing configuration,
  tokenizer files, and verification.
- `--emit-dense-ref PATH`: create a fully dequantized BF16 reference for
  numerical comparison.

The converter is intentionally fail-closed. A model whose dimensions differ
from the Qwen3.8 27B profile requires a new mapping profile instead of silently
reusing incompatible shapes.

## Mapping and transforms

`scripts/gguf_mapping.py` is the single source of truth for:

- GGUF-to-InfiniLM parameter names and shapes;
- packed versus dense storage;
- fused tensor slices;
- conversion-time value-head permutations;
- runtime activation-permutation metadata;
- generated InfiniLM model configuration.

`scripts/gguf_transforms.py` contains the shared NumPy transformations. GGUF
Qwen conversion stores selected value heads in tiled `[value][key]` order,
while InfiniLM uses grouped `[key][value]` order. Complete packed rows can be
permuted during conversion without modifying block bytes.

Some output-projection transformations affect columns instead of rows. Moving
packed columns across quantization blocks would require requantization, so the
converter emits `activation_vperm` rules and the runtime applies the
equivalent grouped-to-tiled permutation to the input activation.

## Runtime integration

`GGUFBlockQuantization` provides:

- name-aware parameter layout selection;
- exact GGML type resolution;
- independent buffers for fused Linear shards;
- per-shard dispatch when fused projections use different GGML types;
- dense BF16 execution through the regular Linear operator;
- packed execution through `linear_gguf`;
- validation for dtype, contiguity, block divisibility, bias, and unsupported
  tensor-parallel configurations.

Linear constructors pass a checkpoint stem to the quantization layer. Fused
projections retain one stem per shard so Q/K/V or gate/up components can resolve
different source types and concatenate their outputs in the original order.

Qwen3.5 model changes supply these stems for attention, MLP, and gated-delta-net
projections. The Python remap also avoids applying a second normalization
`+1` adjustment because llama.cpp already bakes that offset into GGUF.

## Validation performed

The submitted branch has been validated with:

- repository formatting checks;
- a successful InfiniLM extension build;
- the official single-request test;
- the official offline benchmark;
- a local fixed MMLU-format smoke test;
- the official service test with 64/64 successful requests;
- end-to-end Qwen3.8 27B packed-checkpoint loading and generation.

Observed offline performance on the validation machine was approximately:

```text
decode throughput: 5.33 tokens/s
prefill throughput: 6.1 tokens/s
time to first token: 10.49 s
```

These numbers establish functionality, not a portable performance claim. They
were not collected as a controlled comparison against llama.cpp with identical
prompts, context lengths, sampling, and device settings.

## Known limitations

- Packed GGUF tensor parallelism is not implemented.
- IQ4 tensors, embeddings, and the output head use dense BF16 fallbacks.
- Strict token-for-token agreement with llama.cpp is not guaranteed; numerical
  comparisons are the appropriate correctness criterion for quantized kernels.
- A full external MMLU dataset run was unavailable on the validation machine;
  only the local MMLU-format execution path was exercised.
- Upstream CI and maintainer review remain authoritative for merge readiness.

## Extending Route B to another model

Most C++ work is reusable. A new model integration should:

1. Define a model-specific mapping profile with exact checkpoint names, logical
   shapes, fused slices, and required transforms.
2. Generate exact `ggml_types` entries and any activation-permutation rules.
3. Pass checkpoint stems from each model Linear constructor.
4. Add native block types to InfiniCore only when the model uses unsupported
   GGML formats; otherwise reuse `linear_gguf`.
5. Validate conversion with dry-run, exact key/shape/dtype checks, packed-row
   byte preservation, a loadable small-layer checkpoint, and end-to-end output.
6. Benchmark correctness and performance separately with controlled settings.

This separation keeps GGUF storage and dispatch generic while isolating
architecture-specific tensor naming and permutation rules in the converter.
