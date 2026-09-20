# InfiniLM Model Adaptation Reference

This document describes how to support a new model in the InfiniLM C++ inference framework.


---

## 1. Major Content

### 1.1 `InfinilmModel` Abstract Class

Inference models must inherit from `infinilm::InfinilmModel`(`csrc/models/infinilm_model.hpp`), and implement at least the following:

- **`Output forward(const Input &input) const`(pure virtual)**  
  Forward computation entry point: produces **`Output`** from `Input`.

- **`void reset_cache(const cache::CacheConfig *cache_config)`(overridable; default implementation in base class)**  
  **Allocates or rebuilds per-layer KV tensors** based on `cache_config` .


---

### 1.2 Model Structure

- **Model Directory**: Each model has its own directory, e.g., `csrc/models/qwen3/`, `csrc/models/qwen3_next/`. The directory name corresponds to the `model_type` name in `config.json`.
- **Module Separation**: Files are split by module. Common files include `<name>_for_causal_lm.hpp/.cpp`, `<name>_attention.*`, `<name>_decoderLayer.*`, `<name>_allocate_kv_cache_tensors.cpp` (only required when a custom KV allocation is needed).
- **Reuse**: Consider using components such as `TextDecoderLayer`, `TextModel`, `TextCausalLM`, and `MLP` provided by `csrc/layers/`.

---

### 1.3 KV cache

- **Implementation and Invocation**:  The KV cache is implemented in the `default_allocate_kv_cache_tensors`function or a **custom `<name>_allocate_kv_cache_tensors`** function; it is called within the `reset_cache(cache_config)` function.
- **Custom Implementation**: When the default KV cache implementation is unsuitable, implement your own function by referring to examples such as `qwen3_next_allocate_kv_cache_tensors` and `minicpm_sala_allocate_kv_cache_tensors`.

---

### 1.4 `ModelConfig` Object

- The **`ModelConfig`** object is constructed from **`config.json`**.
- The **`create_<架构>_model_config`** function is used to **validate** `model_type`,  and to **complete** missing information (e.g., `head_dim`, `layer_types`).

---

### 1.5 Model Registration

- Register **model information** using the macro **`INFINILM_REGISTER_CAUSAL_LM_MODEL(...)`**.
- **Location**: The registration call is placed in an anonymous namespace **at the end of `<name>_for_causal_lm.cpp`** .
- **Constraint**: The string used for registration must match the **`model_type` in `config.json`**.


---

## 2. Notes

### 2.1 Naming Conventions
The following conventions are recommended for new models:

- **Directory Name**: `csrc/models/<model_type>/`, matching `model_type` in `config.json` (e.g., `qwen3`).
- **Namespace**: `namespace infinilm::models::<model_type> { ... }`, to reduce naming conflicts between different models.
- **Core Files**: 
  - `<model_type>_for_causal_lm.hpp/.cpp`: Top-level model and registration entry point.
  - `<model_type>_attention.hpp/.cpp`: Custom attention (added when the general implementation is insufficient).
  - `<model_type>_decoderLayer.hpp/.cpp`: Custom decoder layer (added when templates are insufficient).
  - `<model_type>_allocate_kv_cache_tensors.cpp`: Custom KV cache allocation (added when the default implementation does not fit).
- **Configuration Post-processing Functions**: `create_<model_type>_model_config(...)`, keeping the name consistent with the function specified in the registration macro.
- **Registration Macro Usage**: `INFINILM_REGISTER_CAUSAL_LM_MODEL(qwen3, Qwen3ForCausalLM, create_qwen3_model_config)`(example).

### 2.2 Code Reuse
- **Level 1: `csrc/layers/`**
  - Prefer using templates such as `TextDecoderLayer`, `TextModel`, `TextCausalLM`.
  - Prefer using modules such as `infinilm::layers::MLP`, `ReplicatedLinear`, and the general `AttentionLayer`.
  - Example: `using Qwen3MLP = infinilm::layers::MLP;`

- **Level 2: Same-series `csrc/models/` modules**
  - When the architecture is consistent with an existing model, reusing stable modules is recommended.
  - Example: `qwen3_moe` reuses the existing `Qwen3Attention` module via `using Qwen3MoeAttention = qwen3::Qwen3Attention`.

- **Level 3: New Implementation**
  - When required modules are incompatible with existing implementations, custom attention, decoder, cache, or other related code may be written.

### 2.3 Avoid Modifying the Framework
Implementation of a new model should be concentrated within the model's own directory (involving tasks such as: `model structure assembly`, `forward`, `reset_cache`, `create_<name>_model_config`, and model registration), **Avoid modifying** common framework code unless necessary.

- **Scope**: Avoid modifying `csrc/layers/`, `csrc/models/infinilm_model.*`, `models_registry.*`, `model_factory.*`, etc., unless strictly necessary.

- **Change Note**: If modifications to files outside a model directory are required, it indicates that the framework's capabilities or interfaces are insufficient to meet the requirements. Such changes will be reviewed carefully and may involve framework-level changes to be added and rebased on.

### 2.4 Do Not Reference/Modify/Use the llama_legacy Directory

The integration approach within `csrc/models/llama_legacy/` is **not the currently recommended path** and might be removed in the future. For new model implementations, please use the models listed in §3 as primary references.


`python/infinilm/auto_config.py` typically requires no changes.

---

## 3. Reference Model Selection Guide

### 3.1 `fm9g`: Composition of Existing Modules

- **Attention**: General `infinilm::layers::attention::Attention` (see `common_modules.hpp` and the `attention` module for dependencies).
- **MLP**: `infinilm::layers::MLP`.
- **Type Aliases**: `TextDecoderLayer<FM9GAttention, FM9GMLP>` → `TextModel` → `TextCausalLM`.
- **Configuration**: `create_fm9g_model_config` can supplement JSON fields (e.g., deriving `head_dim` from dimensions).


**Files**: `csrc/models/fm9g/fm9g_for_causal_lm.hpp`, `.cpp`.

### 3.2 `qwen3`: Custom Attention + Standard MLP

- **Attention**: `Qwen3Attention`(`qwen3_attention.hpp`, `.cpp`).
- **MLP**: `infinilm::layers::MLP`.
- **Top Level**: `TextCausalLM<Qwen3Model>`.

**Files**: `csrc/models/qwen3/qwen3_for_causal_lm.hpp`, `.cpp`, and `qwen3_attention.*`.



### 3.3 `minicpm_sala`: Custom KV Allocation

- `MiniCPMSALAForCausalLM` inherits from `InfinilmModel`; its `reset_cache` calls **`minicpm_sala_allocate_kv_cache_tensors`**.

**Files**: `csrc/models/minicpm_sala/minicpm_sala_for_causal_lm.hpp`, `.cpp`, `minicpm_sala_allocate_kv_cache_tensors.cpp` etc.



---

## 4. Implementation Steps (C++)

### 4.1 Create a New Directory

Organize header and implementation files under `csrc/models/<your_model>/`. The following is an example directory layout using `qwen3`; add or remove files as needed:

```text
csrc/models/qwen3/
├── qwen3_for_causal_lm.hpp  
├── qwen3_for_causal_lm.cpp   
├── qwen3_attention.hpp
└── qwen3_attention.cpp
```

- `<name>_for_causal_lm.hpp` / `.cpp`: Assembles the top-level `ForCausalLM` or `TextCausalLM` and contains the **registration macro** translation unit.
- If custom sub-modules exist, add files such as `<name>_attention.*`, `<name>_decoderLayer.*`, `<name>_allocate_kv_cache_tensors.cpp`.

### 4.2 Implement Decoder Modules

**(1)Type Alias Composition for `TextModel` / `TextCausalLM`(Dense model example: `qwen3`)**

```7:15:csrc/models/qwen3/qwen3_for_causal_lm.hpp
using Qwen3MLP = infinilm::layers::MLP;

using Qwen3Attention = infinilm::models::qwen3::Qwen3Attention;

using Qwen3DecoderLayer = infinilm::layers::causal_lm_templates::TextDecoderLayer<Qwen3Attention, Qwen3MLP>;

using Qwen3Model = infinilm::layers::causal_lm_templates::TextModel<Qwen3DecoderLayer>;

using Qwen3ForCausalLM = infinilm::layers::causal_lm_templates::TextCausalLM<Qwen3Model>;
```

**(2)Sub-module Constructor Convention: `TextDecoderLayer` requires `Attention` and `MLP`(or MoE block)to be registered with fixed parameters** (see full implementation at `csrc/layers/causal_lm_templates/text_decoder_layer.hpp`)

```cpp
// Interface summary (implementation found in source file)
template <typename Attention, typename MLP>
class TextDecoderLayer : public infinicore::nn::Module {
public:
    TextDecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     size_t layer_idx,
                     const infinicore::Device &device);
    // 内部: register_module<Attention>("self_attn", model_config, layer_idx, device);
    //       register_module<MLP>("mlp", model_config, device);
};
```

A custom `Attention` module must provide a constructor matching the signature `(model_config, layer_idx, device)`; the FFN slot must provide `(model_config, device)`.

**(3)`TextCausalLM`: Registers `model` and `lm_head`, `forward` maps hidden states to logits** (see full implementation at `csrc/layers/causal_lm_templates/text_causal_lm.hpp`)

```cpp
// Interface summary (constructor and forward implementation found in source file)
template <typename Model>
class TextCausalLM : public InfinilmModel {
public:
    TextCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                 const infinicore::Device &device);
    Output forward(const Input &input) const override;
};
```

If `TextCausalLM` cannot be used directly as the top-level class, create a custom subclass and explicitly execute `INFINICORE_NN_MODULE_INIT(model, ...)` and `lm_head` initialization.

### 4.3 Configuration Post-processing： `create_<name>_model_config`

Signature： `std::shared_ptr<infinilm::config::ModelConfig> create_<name>_model_config(std::shared_ptr<infinilm::config::ModelConfig>)`(see `models_registry.hpp`).

**Validating `model_type` only (example: `qwen3`)**(`csrc/models/qwen3/qwen3_for_causal_lm.cpp`); 
**Supplementing JSON fields (example: `fm9g` deriving `head_dim`)**(`csrc/models/fm9g/fm9g_for_causal_lm.cpp`): 
```cpp
std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);
```



### 4.4 Static Registration

`#include` `models_registry.hpp` and place within an anonymous namespace at the end of **`qwen3_for_causal_lm.cpp`**. Structure overview (using `qwen3` as an example; see `csrc/models/qwen3/qwen3_for_causal_lm.cpp` for the complete code):

```cpp
#include "<name>_for_causal_lm.hpp"
#include "../models_registry.hpp"  // relative path adjusted for directory depth

namespace infinilm::models::qwen3 {
std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);
// create_xxx: logic such as model_type validation
} // namespace infinilm::models::qwen3

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    qwen3,
    infinilm::models::qwen3::Qwen3ForCausalLM,
    infinilm::models::qwen3::create_qwen3_model_config);
}
```

### 4.5 KV Cache

- **Default Path**: When `reset_cache` is not overridden, the base class `InfinilmModel::reset_cache` calls `default_allocate_kv_cache_tensors` (`infinilm_model.cpp`).
- **KV Element dtype** is retrieved from **`model_config_->get_kv_cache_dtype()`**.
- **Custom Path**: The subclass overrides `reset_cache`, clears and populates `global_state::get_forward_context().kv_cache_vec`. Declaration example (`minicpm_sala`, see `csrc/models/minicpm_sala/minicpm_sala_for_causal_lm.cpp` for implementation): 

```cpp
void MiniCPMSALAForCausalLM::reset_cache(const cache::CacheConfig *cache_config);
// Process summary: if cache_config is nullptr, delegate to base class reset_cache(nullptr);
// otherwise, perform a unique_copy of cache_config, clear kv_cache_vec, call
// minicpm_sala_allocate_kv_cache_tensors(...) and assign the result back.
```

---

## 5. Speculative Decoding with Qwen3.5 MTP Draft Models

Qwen3.5 checkpoints embed their MTP draft weights under `mtp.*` keys (transformers loads and ignores them). The speculative runner accepts such a checkpoint as `--draft-model`, derives a standalone single-layer draft config from the checkpoint's own `config.json`, and verifies draft tokens against the target model — the weight shards are shared via symlinks, no copies are made.

### Usage

- Python: `LLM(model_path=<qwen3.5 checkpoint>, draft_model_path=<same checkpoint>, num_draft_tokens=K, ...)`.
- CLI: `--draft-model <checkpoint>` with `--num-draft-tokens K` (`examples/bench.py`, `test/models/qwen3_5_mtp/test_speculative_lossless.py`).
- Hybrid Qwen3.5 targets (linear-attention + full-attention layers) require the paged attention backend with prefix caching disabled: `cache_type="paged"`, `attn_backend="paged-attn"`, `enable_prefix_caching=False`.
- Drafting requires the paged KV cache. With `cache_type="static"` the engine still constructs, and requests stay on the non-speculative path (the engine logs a warning): the static scheduler hands out no verification slots, so `--draft-model` has no effect. Whether a request can then be served at all is a separate matter of the cache fitting the model's recurrent state — the hybrid targets above need the paged backend for that reason.
- The derived draft config is a **temporary directory**: the runner resolves the checkpoint once, writes a standalone draft `config.json` into a fresh directory under the system temp directory, and symlinks the checkpoint's shard files into it, so no weights are copied. Nothing removes that directory. It has to outlive the draft engine — the shards are read through those symlinks while the engine runs — and it can be deleted once the engine is closed; a cleaned `/tmp` after the engine started shows up as a missing shard file at load time.
- `attn_backend="paged-attn"` is the backend **paired with** the paged cache rather than a third independent requirement: whether the engine uses paged attention is decided by `cache_type`, and this value is what the supported configurations pass.
- Greedy decoding only (`temperature=1.0`, `top_k=1`): the current MTP verification is exact for greedy decoding at batch size 1 (one request in flight); non-greedy sampling falls back to the non-speculative path.

### Parameters

- `num_draft_tokens` (`--num-draft-tokens`): number of draft tokens verified per target step. **Losslessness holds for `num_draft_tokens=1` at batch size 1 (one request in flight), and only there**: in that setting greedy output matches speculation-off token for token. K>1 is a mechanism performance reference, and so is more than one request in flight; both are described below. For K>1 a partially accepted verification advances the target's recurrent linear-attention state by the accepted prefix only — the accepted tokens are replayed into borrowed state rows, and a full accept swaps those rows in — so the committed state stays aligned with the emitted tokens; both the replay and the full-accept hand-off are covered by the hand-off test for K=2..4. One caveat remains, and it belongs to the kernels rather than to the mechanism: K>1 verifies with the chunked recurrent kernels where K=1 decodes one token at a time, and when a step's top-2 logits fall inside a single bf16 quantum the two can round to different argmaxes. That has been observed at more than one position, and on both an RTX 4090D and a MetaX C500, but not as one uniform phenomenon: with speculation off, the decode leg alone already lands on a different branch on each platform, so the same configuration does not produce the same pass counts. In the cases where the logits margin was recomputed in fp32, the two candidates fell inside a single bf16 quantum — a near-tie in the model's own logits, not a state divergence. This path carries no device-specific code, and it has been exercised on both NVIDIA (sm_89) and MetaX C500 hardware. With more than one request in flight the runs are reported per request and are a performance reference, not a lossless setting. Two limits on this reading are worth stating: the near-tie attribution above is about the engine compared against itself (speculation on versus off), where a systematic numeric difference cancels; this engine's logits can differ from a transformers reference by far more than a bf16 quantum (measured 4.4–19.3 on the fixed prompts), so the two readings must not be carried over to each other. And engine-versus-transformers agreement is input-dependent rather than general: the fixed prompt sets used here agree token for token, while generated text that becomes long and repetitive can diverge, which is a property of this engine's numerics and not of the draft path.

### Verification and benchmarking

```bash
# Losslessness: speculative vs non-speculative greedy output, token by token.
# Without --model this runs on a tiny synthetic checkpoint carrying the target
# and its draft head, so it needs no downloaded weights; --model runs the same
# comparison on a released checkpoint at K=1 (see §6.4 for the K>1 boundary).
python test/models/qwen3_5_mtp/test_speculative_lossless.py --device cuda --num-draft-tokens 1

# State hand-off: the K=2..4 verification, partial-accept replay and full-accept
# swap, against plain decoding over the same tokens (needs a real checkpoint).
python test/models/qwen3_5_mtp/test_verify_handoff.py --device cuda

# Latency: speculation off vs on over the same prompt set, multiple rounds.
python test/models/qwen3_5_mtp/test_speculative_latency.py --device cuda --num-draft-tokens 1

# The same losslessness comparison for the other implemented family. Without
# --model it uses its own synthetic checkpoint; with --model <MiMo-7B> it runs
# the released checkpoint the §6.4 measurement names.
python test/models/mimo_mtp/test_speculative_lossless.py --device cuda --num-draft-tokens 1

# Run the unit-level checks of each family. `unittest discover` needs one
# directory at a time: the two families carry same-named test modules and the
# tree has no package __init__.py, so a single interpreter cannot import both.
python -m unittest discover -s test/models/qwen3_5_mtp -t test/models/qwen3_5_mtp
python -m unittest discover -s test/models/mimo_mtp   -t test/models/mimo_mtp

# Throughput matrix over batch sizes (speculative off):
python examples/bench.py --model ~/models/Qwen3.5-2B --device cuda --dtype bfloat16 \
    --enable-paged-attn --batch-size 1 --input-len 128 --output-len 64

# Same matrix with MTP speculation on:
python examples/bench.py --model ~/models/Qwen3.5-2B --device cuda --dtype bfloat16 \
    --enable-paged-attn --batch-size 1 --input-len 128 --output-len 64 \
    --draft-model ~/models/Qwen3.5-2B --num-draft-tokens 1
```

The two branches do not read `--dtype` the same way: the speculative branch builds its engines
through `LLM(...)`, which accepts a `dtype`, while the non-speculative branch goes through
`InferEngine(...)`, whose signature has no `dtype` at all — it takes the dtype from the checkpoint's
own `torch_dtype`. `--dtype` therefore does not decide the weights either way; on a checkpoint whose
`torch_dtype` is `bfloat16`, `--dtype bfloat16` and `--dtype float32` load the same bf16 engines and
produce the same tokens, so the two lines above are the same configuration and comparing them is
valid.

### Changes outside the model directories

§2.3 asks for the framework changes a contribution needs to be listed, because they are the part a
reviewer cannot read as "one more model". This contribution's draft path is family-neutral, so most
of its framework surface is the reusable half of the mechanism; nothing here changes an existing
model's inference results.

Reusable additions (new optional parameters, defaulted off):

- `csrc/engine/infer_engine.cpp`, `csrc/engine/rank_worker.hpp`,
  `csrc/global_state/forward_context.hpp`, `csrc/pybind11/engine/engine.hpp`,
  `python/infinilm/infer_engine.py` — a batch can now state its shape through
  `mamba_multi_token_batch` instead of leaving the model to infer it from the packed layout, and a
  batch can name the recurrent state rows it reads and writes (`mamba_init_state_indices`,
  `mamba_final_state_indices`). Both default to the previous behaviour.
- `python/infinilm/llm/cache_manager.py` — the state-row allocator (`MambaCacheManager`) and the
  borrow/swap/release cycle a verification uses to keep a request's committed row aligned with the
  tokens it actually emitted.
- `python/infinilm/llm/scheduler.py` — the scheduler hands the state-row owner to the runner
  alongside the KV-cache ops.

Changes to existing models, each one needed for the draft to read the target:

- `csrc/models/qwen3_5/qwen3_5_for_causal_lm.cpp` — the target's forward also returns its last
  hidden states, which is the input the draft head consumes.
- `csrc/models/qwen3_5/qwen3_5_attention.cpp` — the gated-attention output multiplies the sigmoid
  of its gate after reshaping it, which is what the released checkpoints require.
- `csrc/models/qwen3_next/qwen3_next_gated_deltanet.cpp` — the recurrent layer can read and write
  named state rows, which is what a multi-token verification needs.
- `csrc/models/mimo/` — the MiMo target's own directory, extended for the embedded draft head.
- `python/infinilm/modeling_utils.py` — the weight loader drops the embedded draft tensors from a
  target load (see §6), and injects the draft's own embedding and head from the target's tables.
- `python/infinilm/llm/model_runner/speculative_runner.py` — the draft path itself: draft rollout,
  verification, accepted-prefix replay and the full-accept hand-off.

---

## 6. Adding a New MTP Draft Model

InfiniLM treats an MTP head as a **class of draft**: any checkpoint that publishes a serial
"target hidden state → next-token logits" head is a **candidate** for `--draft-model`. Whether this
build can run it depends on the criteria in §6.1 and on whether the family's draft block is
implemented here (§6.4) — a serial head whose block is not implemented still resolves to an
actionable error rather than running. The speculative
runner, the scheduler and the recurrent-state bookkeeping are family-neutral — they decide what a
checkpoint is from the checkpoint's own metadata and from the family description in
`python/infinilm/draft_spec.py`. The **draft side** of adding a family therefore means:

1. **one description** (`DraftModelSpec`) registered in `python/infinilm/draft_spec.py`, covering
   the criteria below, and
2. **one registration entry** (`register_draft_model_spec`), and
3. *only if* the family's draft decoder block is not one this build can already compose, **one draft
   model under `csrc/models/<family>/`** following §4 — the same contribution any new model needs,
   and
4. **one row in the status table of §6.5** — the test suite keeps that table in sync with the
   registry, so a described family that is missing from it fails the tests.

The draft side needs **no change to `speculative_runner.py`**: the runner resolves the family through
the registry, and the loader derives the family's weight mapping from the description.

A family whose head lives **inside the target checkpoint** also has a **target side**: the target
model loads the whole checkpoint, so the embedded draft tensors have to leave the shard dictionaries
before the target module tree sees them — they are unknown keys there and would fail the load. The
same description answers that side: a description whose head is embedded and which carries a
key mapping (`weight_map`) serves the target
`model_type`s it lists in `target_model_types`, and the mapping it derives removes the draft tensors
that input publishes — the keys its `family_keys` recognise, plus the draft layers those keys
locate in the same tensors — while handing every other tensor back unchanged. A
standalone family (one whose head is not embedded) is never answered for this way, and an entry in
the weight-remapper table (`python/infinilm/modeling_utils.py`) always takes precedence when one
exists, so the registered entries keep the behaviour they had. A family that publishes its block as
an extra layer of the target's own layout (the DeepSeek-V3 layout in §6.3 is one)
is covered too: its own keys locate that layer, and the whole layer leaves with it,
including the decoder-block tensors the layer reuses, which the key feature alone does not name. The
loader maps one shard at a time, so that location is per shard: a shard whose tensors belong to a
draft layer but carry none of the family's keys does not locate it, and such a load is refused with
those keys named instead of accepting them.

The two families implemented here, side by side:

| | `qwen3_5_mtp` | `mimo_mtp` |
| --- | --- | --- |
| draft side | description + registration (+ the target's own layer class) | description + registration + `csrc/models/mimo_mtp/` |
| target side | nothing to add: the table entry for `qwen3_5` already dropped `mtp.*`, and the description answers for `qwen3_5` as well | one new table entry dropping `model.mtp_layers.*`; the description answers for `mimo` as well |
| runner | unchanged | unchanged |

A **standalone** draft checkpoint (its own directory, its own `model_type`) has no target side at
all: nothing adds its tensors to a target load, so criteria C1–C6 and the description are the whole
integration.

### 6.1 Does my checkpoint qualify? (criteria)

A checkpoint is usable when **all six** hold. Check them from the checkpoint alone — its
`config.json` and its safetensors index — before writing any code. Which fields this build can read
is stated per criterion; a metadata shape it cannot read is still a usable clue, but it has to be
recorded in the description by hand.

| # | Criterion | How to check it on the checkpoint | Real counter-example |
| --- | --- | --- | --- |
| **C1** | The checkpoint actually publishes the draft head | the tensor index contains the family's own draft keys (its fusion/norm tensors and its draft-layer namespace) — a draft depth field alone is not enough and belongs to C2 | `moonshotai/Kimi-K2-Instruct`: 139,644 tensors, none matching `mtp`/`nextn` |
| **C2** | The draft depth is determinable, and the counting key matches the weights | this build reads **scalar integer keys** from `config.json` and from its `text_config`: `num_nextn_predict_layers`, `mtp_num_hidden_layers`, `num_mtp_layers`, `num_mtp_modules` are among them, and a family's own `depth_keys` may name others. A **nested object** such as `mtp_config` or a **per-layer list** such as `mtp_layers_block_type` is not read: it is a clue that the depth is derivable, but the description has to declare `declared_depth` itself. Otherwise the description declares the depth | `Qwen/Qwen3-Next-80B-A3B-Instruct` publishes 1553 `mtp.*` tensors with **no** MTP field in `config.json`; `inference-optimization/Nemotron-3.5-Lightning-1.4B-A0.1B-MTP` counts `num_nextn_predict_layers: 1` but publishes two `mtp.layers.*` groups — the weights win and the mismatch is reported |
| **C3** | The head is a serial draft forward over the target **hidden state** | its tensors form `norm(embedding) + norm(hidden) → projection → block`; it must not consume the target's KV cache | `google/gemma-4-e2b-it-assistant` is a `gemma4_assistant` checkpoint whose forward takes the target's `shared_kv_states` (KV values of the target's last layer of each layer type) through `pre_projection`/`post_projection` — a different draft class that needs a target-KV channel |
| **C4** | The block's layers can be composed here | all-attention, linear/hybrid, or MoE — otherwise a `csrc/models/<family>/` draft block is needed (§4) | `deepseek-ai/DeepSeek-V3` (MLA + MoE) needs its own block |
| **C5** | Same vocabulary as the target | `vocab_size` matches the target's, and the embedding/head sharing mode is declared | `zai-org/GLM-4.5` stores one embedding per draft depth |
| **C6** | Float weights, and a causal chain of heads | the head is not quantized; heads are chained rather than parallel | Medusa-style parallel heads; note that *quantized draft heads* are a choice this project does not implement, not a property of the model |

Two boundaries are worth stating explicitly, because they are about this infrastructure rather than
about the model: a head that needs the target's KV states (C3) and a head with a different vocabulary
(C5) are out of reach here, and the wording for them is "this build has no channel for X", **not**
"the model does not support MTP". Quantized draft heads (C6a) are simply not implemented.

### 6.2 The description

`DraftModelSpec` fields, grouped by what they decide. Two markings separate what this build does with
a field from what it only records:

- **(recorded)** fields are expressible but not executed: a description that sets one fails at
  construction, on every path that uses it — a checkpoint embedding its head and a standalone draft
  directory alike — naming the gap, so a family can state what it needs without being silently
  mis-built.
- **descriptive only** fields are read by nothing in this build, not even to be rejected. They record
  a family's semantics for the next integrator, and the draft block a family contributes under
  `csrc/models/<family>/` is what decides the behaviour they describe. Setting one changes no
  behaviour here, so a family that needs it needs its own block.

| Group | Fields | Meaning |
| --- | --- | --- |
| Packaging | `family`, `draft_model_type`, `target_model_types`, `embedded` | who the family is, the `model_type` the draft engine is built with, which target `model_type`s embed this head, and whether the head lives inside the target checkpoint. `model_type` is what attributes a checkpoint to a family: a checkpoint whose model type is not listed here is never read through this description |
| Depth | `depth_keys`, `declared_depth`, `layer_config_key` | where the published depth may appear (scalar integer keys, read from `config.json` and its `text_config`); the depth to use when the checkpoint publishes none; the config field the draft engine reads its per-layer types from |
| Depth (recorded) | `runtime_depth`, `runtime_depth_keys`, `shared_layer_depths` | a family that publishes more depths than it rolls out, the config keys that cap the rollout (a family trimming by `num_spec`, for instance), and the published depths that reuse the target's layer instead of their own (a family's `local_layer_ids`) |
| Block | `layer_kinds`, `layer_source`, `fusion`, `concat_order`, `hidden_streams`, `chain_causal` | what the head computes: its layer types, whether it reuses the target's decoder layer class, how the two input streams are fused (and in which order), how many hidden streams it consumes, and whether the heads are chained. Each of these is checked against what this build composes |
| Block (descriptive only) | `embedding_at_position_zero`, `recycle_hidden` | what feeds the embedding stream at the first drafted position, and which tensor is fed back to the next step. The shared draft block fixes both — the position-0 embedding is the previous token and the recycled tensor is the post-norm hidden state (see §6.4) — so the family's own block is the only thing that can change them |
| Sharing | `position_ids` | the position-id layout the draft engine expects; the layout a family needs is chosen when its description is resolved |
| Sharing (descriptive only) | `embedding_sharing` | how the embedding and output head are shared with the target. Only `PER_DEPTH` is acted on, as a rejection: the draft loader here shares a single embedding table with the target, which is what the implemented families do |
| Weights | `weight_map` (`family_keys`, `key_pattern`, `layer_key_pattern`, `canonical_prefix`, `depth_index_key`, `renames`, `zero_centered_keys`, `zero_centered_norms`, `embedding_keys`, `head_keys`) | `family_keys` recognises the family's draft tensors and `layer_key_pattern` counts the published depths; `key_pattern` selects the tensors to rename into the draft module tree (`canonical_prefix` + its `rest` group, with `depth_index_key` anchoring a block that is published after the target's own layers); `renames` rewrites name fragments; `zero_centered_keys` / `zero_centered_norms` list the weights stored around zero; `embedding_keys` / `head_keys` are the target tensors the draft shares |
| Gates | `rejected_config_flags`, `unimplemented` | checkpoint flags the description cannot express (they fail loudly), and what this build still lacks for the family |

Two of the input-fusion modes are **recorded but not executed** here: `ADD_PROJECTION` and
`INDEPENDENT_PROJECTION_SHARED_RESIDUAL` come from third-party MTP implementations (vLLM's
`residual_linear_shared` form and SGLang's broadcast-add NextN form), and this repository has not
reproduced them — only `CONCAT_PROJECTION` runs, so a description asking for the other two fails at
construction. The same reading applies to every `(recorded)` field above; a **descriptive only**
field instead fails nothing and changes nothing.

The `(recorded)` and **descriptive only** sets are pinned in the documentation gate
(`test/models/qwen3_5_mtp/test_documentation_gate.py`): a field that starts being executed, or stops
being executed, has to move between the markings in the same change.

Semantics the description records but this build does not execute are kept rather than dropped: a
family whose `unimplemented` list is non-empty still resolves to an actionable error naming what is
missing, instead of being reported as "not a draft checkpoint".

### 6.3 Worked example: a family with a different layout

DeepSeek-V3 publishes its draft block as **one extra decoder layer** instead of an `mtp.*` subtree
(`deepseek-ai/DeepSeek-V3`, `model.safetensors.index.json`):

```text
model.layers.61.enorm.weight            model.layers.61.hnorm.weight
model.layers.61.eh_proj.weight          model.layers.61.shared_head.norm.weight
model.layers.61.shared_head.head.weight model.layers.61.embed_tokens.weight
```

with `num_nextn_predict_layers: 1` in `config.json` and 61 = `num_hidden_layers`. In the description
this is:

- `family_keys` recognises the block through `enorm`/`hnorm`/`eh_proj`/`shared_head` — not through
  the generic `model.layers.<N>.` namespace, which every transformer checkpoint has;
- `depth_keys=("num_nextn_predict_layers",)`, and the key pattern anchors the depth index on
  `num_hidden_layers`, so the block still maps onto the draft's own layer 0;
- `layer_kinds=(MLA_ATTENTION, MOE)` and `layer_source=REUSE_TARGET`, so the resolver reports the
  missing block instead of building something wrong;
- `recycle_hidden=PRE_FINAL_NORM` — the published head carries its own norm, and the tensor fed back
  to the next step is the block output before it (the Eagle draft in this repository has the same
  shape; `qwen3_5_mtp` instead recycles the post-norm hidden state);
- `embedding_sharing=TARGET_EMBEDDING_AND_HEAD`: the head is published once as `shared_head`, and the
  draft falls back to the target's embedding when the checkpoint publishes no per-depth copy.

A second, structurally different example is the order of the two input streams: some families
concatenate `[hidden; embedding]` instead of `[embedding; hidden]` (`concat_order=HIDDEN_FIRST`, as
published by the Inkling checkpoint's `mtp_hidden_states_first`). Such a family needs a draft block
that implements that order; the description says so instead of silently producing wrong drafts. The
shared draft block concatenates the embedding first, so a family declaring another order also
declares `layer_source=FAMILY_SPECIFIC` — it brings the block that runs it.

### 6.4 What the build runs today

The registry in `python/infinilm/draft_spec.py` is the single place a family is declared. Each
description states what the family's checkpoint *needs*; the draft model registered under
`draft_model_type` is what executes it, so a description and its C++ block must agree.

A checkpoint is attributed to a family by two things together: its `config.json` `model_type` is
listed in that description's `target_model_types`, **and** it publishes the family's characteristic
draft keys. A checkpoint whose model type no description covers is rejected with a message that
opens `model type '<type>' has no draft description` and goes on to list the target model types the
registry does cover (`infinilm/draft_spec.py`, `explain_missing_draft`) — the layout resemblance is
reported, never used as an identification:

- `qwen3_5_mtp` — Qwen3.5 dense checkpoints that embed `mtp.*` weights. Verified end to end on
  real weights at 0.8B, 2B and 9B — a single consumer GPU as well as NVIDIA and MetaX C500 instances.
- `minicpm_eagle` — standalone Eagle draft checkpoints, unchanged.
- `mimo_mtp` — MiMo-7B checkpoints that embed `model.mtp_layers.<depth>.` weights. The block is
  implemented here: one full-attention layer whose q/k/v projections carry biases, the hidden stream
  concatenated first, position 0 masked before the fusion norms, and the final norm feeding both the
  head and the next draft step. The target side of the same checkpoint is covered by the family's own
  loader entry, which drops the embedded draft tensors; the description answers for the target as
  well, so the same checkpoint loads with or without that entry. The depth a checkpoint
  **publishes** decides how many draft
  blocks it carries: `model.mtp_layers.<depth>.` groups win over the `num_nextn_predict_layers`
  count, a disagreement between the two is reported as a warning and read as the published depth,
  and a checkpoint publishing more than one depth is refused, because the draft block here composes
  exactly one. Verified end to end on a MiMo-7B checkpoint: with one drafted token per step and one
  request in flight the speculative output matches non-speculative greedy decoding token for token
  (6 of 6 prompts x 48 tokens) — a self-comparison of the engine, and one whose result is tied to
  those inputs (see §5). With several drafted tokens per step a single position can lose to
  the verification kernels when the model's own top-2 logits fall inside a single bf16 quantum, which
  makes those runs a mechanism performance reference rather than a lossless setting — the same
  boundary the Qwen3.5 family carries for more than one drafted token (see §5 for what was measured
  there). With more than one request in
  flight the runs are reported per request and are a performance reference, not a lossless setting
  either: with two requests in flight one request diverged at a single position, where a one-quantum
  batch-layout difference left the step's top-2 logits inside a single bf16 quantum.
- `qwen_moe_mtp`, `deepseek_v3_mtp` — described and registered, but their draft blocks are not
  implemented in this build, so using one reports exactly which block is missing.

Two further notes for anyone integrating a family:

- The runner only drafts on the **paged** KV cache; with `cache_type="static"` the requests run
  non-speculatively (the engine logs a warning) — see §5.
- A draft must share the target's vocabulary. Equal `vocab_size` is checked at construction; it is
  necessary but not sufficient, and a family with a different tokenizer must be rejected by hand.

A directory whose `config.json` `model_type` is itself a registered draft type (a standalone draft
checkpoint such as the Eagle one, or a draft config materialised from a checkpoint that embeds its
head) is used as given: the draft engine is built from it directly, and no description is applied on
top of it.

### 6.5 Status of the surveyed families

The first column is the **registered description** (`python/infinilm/draft_spec.py`); a row without
one has been surveyed but not described yet. "Block implemented here" is what separates a usable
family from a described-but-not-runnable one, and the row says which piece is missing.

| Registered description | Checkpoint | Draft head | Block | Status here | Tier |
| --- | --- | --- | --- | --- | --- |
| `qwen3_5_mtp` | Qwen3.5 dense (`mtp_num_hidden_layers`) | `mtp.*` | full attention | block implemented here; measured end to end | measured |
| `minicpm_eagle` | MiniCPM4 Eagle draft checkpoints | standalone checkpoint | full attention | block implemented here (unchanged path) | measured previously |
| `qwen_moe_mtp` | Qwen3.5-35B-A3B (`qwen3_5_moe`); Qwen3-Next-80B-A3B (`qwen3_next`) | `mtp.*`, 785 / 1553 tensors; no MTP field in the Qwen3-Next config | full attention + MoE (256 / 512 experts) | block not implemented here: a draft block with a MoE MLP (512 experts plus a shared expert in the published checkpoints) | structurally extensible |
| `deepseek_v3_mtp` | DeepSeek-V3 / V3.2 (`deepseek_v3`, `deepseek_v32`) | `model.layers.<N>.{enorm,hnorm,eh_proj,shared_head.*}` | MLA + MoE | block not implemented here: a DeepSeek MLA + MoE draft block (the target's own decoder layer, reused as the MTP block); also its shared-head norm convention and the relation of its per-depth `embed_tokens` copy to the target embedding | structurally extensible |
| `mimo_mtp` | MiMo-7B (`mimo`) | 16 `model.mtp_layers.0.*` tensors | dense full attention with q/k/v biases | block implemented here; the hidden-first fusion and the position-0 embedding mask follow the family's released rollout code. A MiMo *target* also needs its embedded draft tensors dropped at load time, which the family's own loader entry does and its description answers for as well | measured (K=1); K>1 mechanism reference |
| — | Nemotron-3.5-Lightning-1.4B (`nemotron_h`) | `mtp.*` with DeepSeek-style names, **2 groups** against a count of 1 | full attention + MoE (`mtp_layers_block_type`) | not described yet: needs a MoE draft block | structurally extensible |
| — | Inkling (`inkling_mm_model`) | `mtp_config.num_nextn_predict_layers` | hybrid block, hidden-first fusion | not described yet: needs a hybrid draft block and the hidden-first fusion | not supported yet |
| — | Gemma 4 assistant (`gemma4_assistant`) | separate assistant checkpoint | consumes the target's KV states | not described yet (out of scope: this infrastructure has no target-KV channel) | not supported yet |
| — | Kimi-K2 | none published | — | not described yet: nothing to draft with (C1) | not supported yet |

Every value the Tier column uses, and what it claims:

- **measured** — the family's block ran end to end on a released checkpoint: the lossless
  comparison of §5, or the performance and acceptance numbers of §5, on the platform the row names.
- **measured (K=1); K>1 mechanism reference** — the end-to-end lossless claim holds for one drafted
  token per step on a released checkpoint; more than one is exercised for the mechanism only (see
  §5 for why the verification kernels can move a near-tie).
- **measured previously** — measured by an earlier contribution or run, not re-measured here; this
  pull request neither repeats nor withdraws that result.
- **structurally extensible** — the metadata was read from the published checkpoint and the shape of
  the missing piece is known. It does **not** mean the family has been run.
- **not supported yet** — always names a missing piece of this build, never a claim about the model.

A tier below "measured" that is not used by any row today is **component-verified**: the family's
block was checked against a reference forward and its weight mapping against the published keys, on
synthetic weights rather than a released checkpoint and without an end-to-end measurement. It is
documented here so a family carrying it is not undefined, and it ranks below "measured".

---

*This document may lag behind code changes; for definitive behavior, refer to the source code in `csrc/models`.*
