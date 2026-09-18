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
- Greedy decoding only (`temperature=1.0`, `top_k=1`): the current MTP verification is exact for greedy decoding; non-greedy sampling falls back to the non-speculative path.

### Parameters

- `num_draft_tokens` (`--num-draft-tokens`): number of draft tokens verified per target step. `num_draft_tokens=1` is exact: greedy output matches speculation-off token for token. For K>1 a partially accepted verification advances the target's recurrent linear-attention state by the accepted prefix only — the accepted tokens are replayed into borrowed state rows, and a full accept swaps those rows in — so the committed state stays aligned with the emitted tokens; both the replay and the full-accept hand-off are covered by the hand-off test for K=2..4. One caveat remains, and it belongs to the kernels rather than to the mechanism: K>1 verifies with the chunked recurrent kernels where K=1 decodes one token at a time, and when a step's top-2 logits fall inside a single bf16 quantum the two can round to different argmaxes. That has been observed on one prompt at one position, on both an RTX 4090D and a MetaX C500; it is a near-tie in the model's own logits, not a state divergence. This path carries no device-specific code, and it has been exercised on both NVIDIA (sm_89) and MetaX C500 hardware.

### Verification and benchmarking

```bash
# Losslessness: speculative vs non-speculative greedy output, token by token.
python test/models/qwen3_5_mtp/test_speculative_lossless.py --device cuda --num-draft-tokens 1

# Latency: speculation off vs on over the same prompt set, multiple rounds.
python test/models/qwen3_5_mtp/test_speculative_latency.py --device cuda --num-draft-tokens 1

# Throughput matrix over batch sizes (speculative off):
python examples/bench.py --model ~/models/Qwen3.5-2B --device cuda --dtype bfloat16 \
    --enable-paged-attn --batch-size 1 --input-len 128 --output-len 64

# Same matrix with MTP speculation on:
python examples/bench.py --model ~/models/Qwen3.5-2B --device cuda --dtype bfloat16 \
    --enable-paged-attn --batch-size 1 --input-len 128 --output-len 64 \
    --draft-model ~/models/Qwen3.5-2B --num-draft-tokens 1
```

---

## 6. Adding a New MTP Draft Model

InfiniLM treats an MTP head as a **class of draft**: any checkpoint that publishes a serial
"target hidden state → next-token logits" head can be used as `--draft-model`. The speculative
runner, the scheduler and the recurrent-state bookkeeping are family-neutral — they decide what a
checkpoint is from the checkpoint's own metadata and from the family description in
`python/infinilm/draft_spec.py`. Adding a family therefore means:

1. **one description** (`DraftModelSpec`) registered in `python/infinilm/draft_spec.py`, covering
   the criteria below, and
2. **one registration entry** (`register_draft_model_spec`), and
3. *only if* the family's draft decoder block is not one this build can already compose, **one draft
   model under `csrc/models/<family>/`** following §4 — the same contribution any new model needs.

No change to `speculative_runner.py` and no entry in the weight-remapper table is required: the
runner resolves the family through the registry, and the loader derives the family's weight mapping
from the description.

### 6.1 Does my checkpoint qualify? (criteria)

A checkpoint is usable when **all six** hold. Check them from the checkpoint alone — its
`config.json` and its safetensors index — before writing any code.

| # | Criterion | How to check it on the checkpoint | Real counter-example |
| --- | --- | --- | --- |
| **C1** | The checkpoint actually publishes the draft head | the tensor index contains the family's own draft keys (its fusion/norm tensors and its draft-layer namespace) — a draft depth field alone is not enough and belongs to C2 | `moonshotai/Kimi-K2-Instruct`: 139,644 tensors, none matching `mtp`/`nextn` |
| **C2** | The draft depth is determinable, and the counting key matches the weights | one of `num_nextn_predict_layers` (top level or under `text_config`), `mtp_num_hidden_layers`, `num_mtp_layers`, `num_mtp_modules`, a nested `mtp_config`, or a per-layer list such as `mtp_layers_block_type`; otherwise the description declares the depth | `Qwen/Qwen3-Next-80B-A3B-Instruct` publishes 1553 `mtp.*` tensors with **no** MTP field in `config.json`; `inference-optimization/Nemotron-3.5-Lightning-1.4B-A0.1B-MTP` counts `num_nextn_predict_layers: 1` but publishes two `mtp.layers.*` groups — the weights win and the mismatch is reported |
| **C3** | The head is a serial draft forward over the target **hidden state** | its tensors form `norm(embedding) + norm(hidden) → projection → block`; it must not consume the target's KV cache | `google/gemma-4-e2b-it-assistant` is a `gemma4_assistant` checkpoint whose forward takes the target's `shared_kv_states` (KV values of the target's last layer of each layer type) through `pre_projection`/`post_projection` — a different draft class that needs a target-KV channel |
| **C4** | The block's layers can be composed here | all-attention, linear/hybrid, or MoE — otherwise a `csrc/models/<family>/` draft block is needed (§4) | `deepseek-ai/DeepSeek-V3` (MLA + MoE) needs its own block |
| **C5** | Same vocabulary as the target | `vocab_size` matches the target's, and the embedding/head sharing mode is declared | `zai-org/GLM-4.5` stores one embedding per draft depth |
| **C6** | Float weights, and a causal chain of heads | the head is not quantized; heads are chained rather than parallel | Medusa-style parallel heads; note that *quantized draft heads* are a choice this project does not implement, not a property of the model |

Two boundaries are worth stating explicitly, because they are about this infrastructure rather than
about the model: a head that needs the target's KV states (C3) and a head with a different vocabulary
(C5) are out of reach here, and the wording for them is "this build has no channel for X", **not**
"the model does not support MTP". Quantized draft heads (C6a) are simply not implemented.

### 6.2 The description

`DraftModelSpec` fields, grouped by what they decide. Fields marked **(recorded)** are expressible
but not executed by this build: a description that sets one fails at construction, on every path
that uses it — a checkpoint embedding its head and a standalone draft directory alike — naming the
gap, so a family can state what it needs without being silently mis-built.

| Group | Fields | Meaning |
| --- | --- | --- |
| Packaging | `family`, `draft_model_type`, `target_model_types`, `embedded` | who the family is, the `model_type` the draft engine is built with, which target `model_type`s embed this head, and whether the head lives inside the target checkpoint. `model_type` is what attributes a checkpoint to a family: a checkpoint whose model type is not listed here is never read through this description |
| Depth | `depth_keys`, `declared_depth`, `layer_config_key` | where the published depth may appear; the depth to use when the checkpoint publishes none; the config field the draft engine reads its per-layer types from |
| Depth (recorded) | `runtime_depth`, `runtime_depth_keys`, `shared_layer_depths` | a family that publishes more depths than it rolls out, the config keys that cap the rollout (a family trimming by `num_spec`, for instance), and the published depths that reuse the target's layer instead of their own (a family's `local_layer_ids`) |
| Block | `layer_kinds`, `layer_source`, `fusion`, `concat_order`, `embedding_at_position_zero`, `recycle_hidden`, `hidden_streams`, `chain_causal` | what the head computes: its layer types, whether it reuses the target's decoder layer class, how the two input streams are fused (and in which order), what feeds the embedding stream at the first drafted position, which tensor is fed back to the next step, how many hidden streams it consumes, and whether the heads are chained |
| Sharing | `embedding_sharing`, `position_ids` | how the embedding and output head are shared with the target, and the position-id layout the draft engine expects |
| Weights | `weight_map` (`family_keys`, `key_pattern`, `layer_key_pattern`, `canonical_prefix`, `depth_index_key`, `renames`, `zero_centered_keys`, `zero_centered_norms`, `embedding_keys`, `head_keys`) | `family_keys` recognises the family's draft tensors and `layer_key_pattern` counts the published depths; `key_pattern` selects the tensors to rename into the draft module tree (`canonical_prefix` + its `rest` group, with `depth_index_key` anchoring a block that is published after the target's own layers); `renames` rewrites name fragments; `zero_centered_keys` / `zero_centered_norms` list the weights stored around zero; `embedding_keys` / `head_keys` are the target tensors the draft shares |
| Gates | `rejected_config_flags`, `unimplemented` | checkpoint flags the description cannot express (they fail loudly), and what this build still lacks for the family |

Two of the input-fusion modes are **recorded but not executed** here: `ADD_PROJECTION` and
`INDEPENDENT_PROJECTION_SHARED_RESIDUAL` come from third-party MTP implementations (vLLM's
`residual_linear_shared` form and SGLang's broadcast-add NextN form), and this repository has not
reproduced them — only `CONCAT_PROJECTION` runs, so a description asking for the other two fails at
construction. The same reading applies to every `(recorded)` field above.

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
that implements that order; the description says so instead of silently producing wrong drafts.

### 6.4 What the build runs today

The registry in `python/infinilm/draft_spec.py` is the single place a family is declared. Each
description states what the family's checkpoint *needs*; the draft model registered under
`draft_model_type` is what executes it, so a description and its C++ block must agree.

A checkpoint is attributed to a family by two things together: its `config.json` `model_type` is
listed in that description's `target_model_types`, **and** it publishes the family's characteristic
draft keys. A checkpoint that publishes draft tensors no description covers is rejected at
construction with "no draft description for this model type" — the layout resemblance is reported,
never used as an identification:

- `qwen3_5_mtp` — Qwen3.5 dense checkpoints that embed `mtp.*` weights. Verified end to end
  (0.8B on a single consumer GPU, component-level forward/rollout checks on 2B).
- `minicpm_eagle` — standalone Eagle draft checkpoints, unchanged.
- `qwen_moe_mtp`, `deepseek_v3_mtp`, `mimo_mtp` — described and registered, but their draft blocks
  are not implemented in this build, so using one reports exactly which block is missing.

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
| `mimo_mtp` | MiMo-7B (`mimo`) | 16 `model.mtp_layers.0.*` tensors | dense full attention with q/k/v biases | block not implemented here: a draft block whose attention carries q/k/v biases; the recorded hidden-first concatenation order is **provisional** (taken from third-party code) | structurally extensible |
| — | Nemotron-3.5-Lightning-1.4B (`nemotron_h`) | `mtp.*` with DeepSeek-style names, **2 groups** against a count of 1 | full attention + MoE (`mtp_layers_block_type`) | not described yet: needs a MoE draft block | structurally extensible |
| — | Inkling (`inkling_mm_model`) | `mtp_config.num_nextn_predict_layers` | hybrid block, hidden-first fusion | not described yet: needs a hybrid draft block and the hidden-first fusion | not supported yet |
| — | Gemma 4 assistant (`gemma4_assistant`) | separate assistant checkpoint | consumes the target's KV states | not described yet (out of scope: this infrastructure has no target-KV channel) | not supported yet |
| — | Kimi-K2 | none published | — | not described yet: nothing to draft with (C1) | not supported yet |

"Structurally extensible" means the metadata was read from the published checkpoint and the shape of
the missing piece is known; it does **not** mean the family has been run. "Not supported yet" always
names a missing piece of this build, never a claim about the model.

---

*This document may lag behind code changes; for definitive behavior, refer to the source code in `csrc/models`.*
