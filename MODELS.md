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

*This document may lag behind code changes; for definitive behavior, refer to the source code in `csrc/models`.*
