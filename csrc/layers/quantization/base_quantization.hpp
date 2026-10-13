#pragma once
#include "../../weights/weights.hpp"
#include "runtime/tensor.hpp"
#include "nlohmann/json.hpp"
#include "quantization_scheme.hpp"
#include <infiniccl/infiniccl.h>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <optional>
#include <cstdint>

namespace infinilm::quantization {

class W4Execution;

struct ParamDescriptor {
    std::string name;
    std::vector<size_t> shape;
    infinilm::runtime::DataType dtype;
    int split_dim = -1;
    int tp_rank = 0;
    int tp_size = 1;
    int tp_num_heads = -1;
};

using ParamsMap = std::unordered_map<std::string, infinilm::runtime::Tensor>;

// Describes one shard of a fused linear (e.g., Q, K, V or gate, up)
struct SplitInfo {
    std::string prefix;    // "q_proj", "k_proj", "v_proj" or "gate_proj", "up_proj"
    size_t start;          // start offset along narrow_dim
    size_t size;           // size of this shard along narrow_dim
    size_t num_shards = 0; // number of logical shards for KV replication (0 = standard TP split)
};

// A named parameter produced by splitting a fused linear
struct SplitParam {
    std::string full_name; // "q_proj.weight", "gate_proj.qweight", etc.
    infinilm::weights::Weight param;
};

class BaseQuantization : public std::enable_shared_from_this<BaseQuantization> {
public:
    explicit BaseQuantization(const nlohmann::json &quant_config) : quant_config_(quant_config){};
    virtual ~BaseQuantization() = default;

    const nlohmann::json &get_config() const { return quant_config_; }

    virtual QuantScheme get_quant_scheme() const = 0;

    // Return the list of parameters this quantization scheme needs
    virtual std::vector<ParamDescriptor> get_param_layout(
        size_t in_features, size_t out_features,
        int split_dim, int tp_rank, int tp_size,
        int tp_num_heads,
        const infinilm::runtime::DataType &dtype,
        bool bias) const
        = 0;

    // Forward pass using the registered parameters
    virtual infinilm::runtime::Tensor forward(
        infinilm::runtime::Runtime &runtime,
        const ParamsMap &params,
        const infinilm::runtime::Tensor &input,
        bool has_bias,
        float alpha = 1.0f) const
        = 0;

    virtual infinilm::runtime::Tensor forward_allreduce(
        infinilm::runtime::Runtime &runtime,
        const ParamsMap &params,
        const infinilm::runtime::Tensor &input,
        bool has_bias,
        infinicclComm_t communicator,
        float alpha = 1.0f) const;

    // Dimension for fused-split (gate/up, q/k/v) of a column-parallel weight.
    // For NoneQuantization weight [out, in], split is on dim0.
    // For AWQ qweight [in, out/pack], split is on dim1.
    virtual int get_fused_split_dim() const { return 0; }

    // Logical output size along fused_split_dim from a parameter's raw dimension size.
    // For packed formats (AWQ, GPTQ), raw size needs to be multiplied by packing_num.
    // Default: raw size is already logical size.
    virtual size_t get_logical_dim_size(size_t raw_size) const { return raw_size; }

    // Split fused linear parameters into named sub-parameters (for QKV/GateUp)
    // params: the fused linear's registered parameters (by name)
    // splits: description of each shard
    // narrow_dim: 0=narrow dim0, 1=narrow dim1 (for weight-like params)
    // Returns a list of (full_name, Parameter) pairs
    virtual std::vector<SplitParam> split_params(
        const std::unordered_map<std::string, infinilm::weights::Weight> &params,
        const std::vector<SplitInfo> &splits,
        int narrow_dim,
        int tp_rank, int tp_size, int tp_num_heads) const
        = 0;

    virtual std::shared_ptr<BaseQuantization> process_weights_after_loading(
        infinilm::runtime::Runtime &, ParamsMap &, bool, int = -1) const {
        return nullptr;
    }

    // Finite W4 layer-only staging, separate from typed dense post-load processing.
    virtual std::shared_ptr<const W4Execution> prepare_w4_after_loading(
        infinilm::runtime::Runtime &, const ParamsMap &, int, bool,
        std::optional<int64_t> = std::nullopt) const {
        throw std::logic_error("quantization has no explicit W4 preparation path");
    }

    template <typename T>
    T get(const std::string &key) const {
        if (!quant_config_.contains(key)) {
            throw std::out_of_range("Key '" + key + "' not found in config.");
        }
        try {
            return quant_config_.at(key).get<T>();
        } catch (const nlohmann::json::type_error &e) {
            throw std::runtime_error("Type conversion failed for key '" + key + "': " + std::string(e.what()));
        }
    }

    template <typename T>
    T get_or(const std::string &key, const T &default_value) const {
        if (!quant_config_.contains(key) || quant_config_.at(key).is_null()) {
            return default_value;
        }
        try {
            return quant_config_.at(key).get<T>();
        } catch (const nlohmann::json::type_error &) {
            return default_value;
        }
    }

protected:
    nlohmann::json quant_config_;
};

} // namespace infinilm::quantization
