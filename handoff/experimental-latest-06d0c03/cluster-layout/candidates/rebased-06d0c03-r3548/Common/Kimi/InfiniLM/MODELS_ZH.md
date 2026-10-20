# InfiniLM 新增模型参考文档

本文介绍如何在 InfiniLM C++推理框架中添加新模型 。


---

## 1. 主要内容

### 1.1 `InfinilmModel` 抽象类

推理模型需要继承 `infinilm::InfinilmModel`（`csrc/models/infinilm_model.hpp`），并至少实现：

- **`Output forward(const Input &input) const`（纯虚）**  
  前向计算入口：由 `Input` 得到 **`Output`**。

- **`void reset_cache(const cache::CacheConfig *cache_config)`（可重写，默认实现见基类）**  
  依据 `cache_config` **分配或重建各层 KV 张量**。


---

### 1.2 模型结构

- **模型目录**：每种模型有一个自己的目录，例如 `csrc/models/qwen3/`、`csrc/models/qwen3_next/`，目录名与 `config.json` 里的 `model_type` 命名对应。
- **模块拆分**：按模块分文件，常见包括 `<name>_for_causal_lm.hpp/.cpp`、`<name>_attention.*`、`<name>_decoderLayer.*`、`<name>_allocate_kv_cache_tensors.cpp`（仅自定义 KV 时需要）。
- **复用**：考虑使用 `csrc/layers/` 提供的 `TextDecoderLayer`、`TextModel`、`TextCausalLM`、`MLP` 等组件。

---

### 1.3 KV cache

- **实现和调用**： kv cache的实现在`default_allocate_kv_cache_tensors`函数中 或 **自定义 `<name>_allocate_kv_cache_tensors`** 函数中；调用在 `reset_cache(cache_config)`函数中。
- **自定义实现**：当默认kv cache的实现不适用时，可参考 `qwen3_next_allocate_kv_cache_tensors`、`minicpm_sala_allocate_kv_cache_tensors`等函数实现自己的函数。

---

### 1.4 `ModelConfig`对象

- **`ModelConfig`**对象 由  **`config.json`** 构造。
- **`create_<架构>_model_config`** 函数用于 **校验** `model_type`、**补全**缺失内容（如 `head_dim`、`layer_types`）。

---

### 1.5 模型注册

- 通过宏 **`INFINILM_REGISTER_CAUSAL_LM_MODEL(...)`** 注册 **模型信息** 。
- **位置**：注册调用位于 **`<name>_for_causal_lm.cpp` 末尾** 的匿名命名空间中。
- **约束**：注册所用字符串须与 **`config.json` 中 `model_type`** 一致。


---

## 2. 注意事项

### 2.1 命名规范
新增模型建议遵循下列约定：

- **目录名**：`csrc/models/<model_type>/`，与 `config.json` 中 `model_type` 一致（如 `qwen3`）。
- **命名空间**：`namespace infinilm::models::<model_type> { ... }`，以降低不同模型间的命名冲突。
- **核心文件**：
  - `<model_type>_for_causal_lm.hpp/.cpp`：顶层模型与注册入口。
  - `<model_type>_attention.hpp/.cpp`：自定义 attention（在不满足通用实现时新增）。
  - `<model_type>_decoderLayer.hpp/.cpp`：自定义 decoder layer（在模板不够用时新增）。
  - `<model_type>_allocate_kv_cache_tensors.cpp`：自定义 KV 创建（在默认实现不适配时新增）。
- **配置后处理函数**：`create_<model_type>_model_config(...)`，名称与注册宏中的函数保持一致。
- **注册宏写法**：`INFINILM_REGISTER_CAUSAL_LM_MODEL(qwen3, Qwen3ForCausalLM, create_qwen3_model_config)`（示例）。

### 2.2 代码复用
- **第一层：`csrc/layers/`**
  - 建议采用 `TextDecoderLayer`、`TextModel`、`TextCausalLM` 等模板。
  - 建议采用 `infinilm::layers::MLP`、`ReplicatedLinear`、通用 `AttentionLayer` 等模块。
  - 例：`using Qwen3MLP = infinilm::layers::MLP;`

- **第二层：同系列 `csrc/models/` 模块**
  - 与既有模型架构一致时，建议复用已稳定模块。
  - 例：`qwen3_moe` 通过 `using Qwen3MoeAttention = qwen3::Qwen3Attention` 的方式复用了已有的 `Qwen3Attention`模块。

- **第三层：新增实现**
  - 在所需模块与现有实现不兼容时，可以自定义 attention、decoder 或 cache 等相关代码。

### 2.3 避免修改框架
新增模型实现应集中于本模型目录（涉及到的代码包括：`模型结构搭建`、`forward`、`reset_cache`、`create_<name>_model_config`、模型注册），**非必要不修改**公共框架代码。

- **范围**：非必要不修改 `csrc/layers/`、`csrc/models/infinilm_model.*`、`models_registry.*`、`model_factory.*` 等。

- **变更说明**：如果必须对模型目录外文件的修改，则表明框架能力或接口不足以覆盖需求。这部分改动会被详细考虑，并可能需要加入框架级修改后rebase。

### 2.4 不参考/修改/使用 llama_legacy 目录

`csrc/models/llama_legacy/` 中的接入方式**非本仓库当前推荐路径**，可能会在一段时间后删除；新增模型实现请以 §3 所列模型为主要参照。

`python/infinilm/auto_config.py` 通常不需要任何修改。

---

## 3. 按参考模型选型

### 3.1 `fm9g`：已有模块组合

- **Attention**：通用 `infinilm::layers::attention::Attention`（依赖关系见 `common_modules.hpp` 与 `attention` 模块）。
- **MLP**：`infinilm::layers::MLP`。
- **类型别名**：`TextDecoderLayer<FM9GAttention, FM9GMLP>` → `TextModel` → `TextCausalLM`。
- **配置**：`create_fm9g_model_config` 可对 JSON 补全（如由维度推导 `head_dim`）。


**文件**：`csrc/models/fm9g/fm9g_for_causal_lm.hpp`、`.cpp`。

### 3.2 `qwen3`：自定义 Attention + 标准 MLP

- **Attention**：`Qwen3Attention`（`qwen3_attention.hpp`、`.cpp`）。
- **MLP**：`infinilm::layers::MLP`。
- **顶层**：`TextCausalLM<Qwen3Model>`。

**文件**：`csrc/models/qwen3/qwen3_for_causal_lm.hpp`、`.cpp`，以及 `qwen3_attention.*`。



### 3.3 `minicpm_sala`：自定义 KV 分配

- `MiniCPMSALAForCausalLM` 继承 `InfinilmModel`；`reset_cache` 调用 **`minicpm_sala_allocate_kv_cache_tensors`**。

**文件**：`csrc/models/minicpm_sala/minicpm_sala_for_causal_lm.hpp`、`.cpp`，`minicpm_sala_allocate_kv_cache_tensors.cpp` 等。



---

## 4. 实现步骤（C++）

### 4.1 新建目录

在 `csrc/models/<your_model>/` 下组织头文件与实现。下列为以 `qwen3` 为例的目录布局，可按实际需求增删文件：

```text
csrc/models/qwen3/
├── qwen3_for_causal_lm.hpp  
├── qwen3_for_causal_lm.cpp   
├── qwen3_attention.hpp
└── qwen3_attention.cpp
```

- `<name>_for_causal_lm.hpp` / `.cpp`：顶层 `ForCausalLM` 或 `TextCausalLM` 组合、**注册宏**所在翻译单元。
- 存在自定义子模块时，可增加 `<name>_attention.*`、`<name>_decoderLayer.*`、`<name>_allocate_kv_cache_tensors.cpp` 等。

### 4.2 实现 Decoder 模块

**（1）类型别名组合 `TextModel` / `TextCausalLM`（稠密模型示例：`qwen3`）**

```7:15:csrc/models/qwen3/qwen3_for_causal_lm.hpp
using Qwen3MLP = infinilm::layers::MLP;

using Qwen3Attention = infinilm::models::qwen3::Qwen3Attention;

using Qwen3DecoderLayer = infinilm::layers::causal_lm_templates::TextDecoderLayer<Qwen3Attention, Qwen3MLP>;

using Qwen3Model = infinilm::layers::causal_lm_templates::TextModel<Qwen3DecoderLayer>;

using Qwen3ForCausalLM = infinilm::layers::causal_lm_templates::TextCausalLM<Qwen3Model>;
```

**（2）子模块构造函数约定：`TextDecoderLayer` 要求 `Attention` 与 `MLP`（或 MoE 块）按固定参数注册**（完整实现见 `csrc/layers/causal_lm_templates/text_decoder_layer.hpp`）

```cpp
// 接口摘要（实现见源文件）
template <typename Attention, typename MLP>
class TextDecoderLayer : public infinicore::nn::Module {
public:
    TextDecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     size_t layer_idx,
                     const infinicore::Device &device);
    // 内部：register_module<Attention>("self_attn", model_config, layer_idx, device);
    //       register_module<MLP>("mlp", model_config, device);
};
```

自定义 `Attention` 需提供与上述一致的构造函数 `(model_config, layer_idx, device)`；FFN 位需提供 `(model_config, device)`。

**（3）`TextCausalLM`：注册 `model` 与 `lm_head`，`forward` 为 hidden → logits**（完整实现见 `csrc/layers/causal_lm_templates/text_causal_lm.hpp`）

```cpp
// 接口摘要（构造函数与 forward 实现见源文件）
template <typename Model>
class TextCausalLM : public InfinilmModel {
public:
    TextCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                 const infinicore::Device &device);
    Output forward(const Input &input) const override;
};
```

若无法直接使用 `TextCausalLM` 作为顶层，须自定义子类并显式执行 `INFINICORE_NN_MODULE_INIT(model, ...)` 与 `lm_head` 初始化。

### 4.3 配置后处理 `create_<name>_model_config`

签名为 `std::shared_ptr<infinilm::config::ModelConfig> create_<name>_model_config(std::shared_ptr<infinilm::config::ModelConfig>)`（见 `models_registry.hpp`）。

**仅校验 `model_type`（示例：`qwen3`）**（`csrc/models/qwen3/qwen3_for_causal_lm.cpp`）；
**补全 JSON 字段（示例：`fm9g` 推导 `head_dim`）**（`csrc/models/fm9g/fm9g_for_causal_lm.cpp`）：
```cpp
std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);
```



### 4.4 静态注册

在 **`qwen3_for_causal_lm.cpp`** 末尾使用匿名命名空间，并 `#include` `models_registry.hpp`。结构示意（以 `qwen3` 为例，完整代码见 `csrc/models/qwen3/qwen3_for_causal_lm.cpp`）：

```cpp
#include "<name>_for_causal_lm.hpp"
#include "../models_registry.hpp"  // 相对路径依目录深度调整

namespace infinilm::models::qwen3 {
std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);
// create_xxx：model_type 校验等逻辑
} // namespace infinilm::models::qwen3

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    qwen3,
    infinilm::models::qwen3::Qwen3ForCausalLM,
    infinilm::models::qwen3::create_qwen3_model_config);
}
```

### 4.5 KV Cache

- **默认路径**：未重写 `reset_cache` 时，基类 `InfinilmModel::reset_cache` 调用 `default_allocate_kv_cache_tensors`（`infinilm_model.cpp`）。
- **KV 元素 dtype** 由 **`model_config_->get_kv_cache_dtype()`** 取得。
- **自定义路径**：子类重写 `reset_cache`，清空并写入 `global_state::get_forward_context().kv_cache_vec`。声明示例（`minicpm_sala`，实现见 `csrc/models/minicpm_sala/minicpm_sala_for_causal_lm.cpp`）：

```cpp
void MiniCPMSALAForCausalLM::reset_cache(const cache::CacheConfig *cache_config);
// 流程概要：cache_config 为空指针时转调基类 reset_cache(nullptr)；
// 否则对 cache_config 做 unique_copy、清空 kv_cache_vec、调用 minicpm_sala_allocate_kv_cache_tensors(...) 并赋值回写
```

---

*本文档随代码库变更可能滞后；具体行为以 `csrc/models` 源码为准。*
