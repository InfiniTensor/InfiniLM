#pragma once
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"
#include "nlohmann/json.hpp"
#include "quantization_scheme.hpp"
#include <string>
#include <unordered_map>
#include <vector>

namespace infinilm::quantization {

struct ParamDescriptor {
    std::string name;
    std::vector<size_t> shape;
    infinicore::DataType dtype;
    int split_dim = -1;
    int tp_rank = 0;
    int tp_size = 1;
    int tp_num_heads = -1;
};

using ParamsMap = std::unordered_map<std::string, infinicore::Tensor>;

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
    infinicore::nn::Parameter param;
};

class BaseQuantization : public std::enable_shared_from_this<BaseQuantization> {
public:
    explicit BaseQuantization(const nlohmann::json &quant_config) : quant_config_(quant_config) {};
    virtual ~BaseQuantization() = default;

    const nlohmann::json &get_config() const { return quant_config_; }

    virtual QuantScheme get_quant_scheme() const = 0;

    // Return the list of parameters this quantization scheme needs
    virtual std::vector<ParamDescriptor> get_param_layout(
        size_t in_features, size_t out_features,
        int split_dim, int tp_rank, int tp_size,
        int tp_num_heads,
        const infinicore::DataType &dtype,
        bool bias) const = 0;

    // Forward pass using the registered parameters
    virtual infinicore::Tensor forward(
        const ParamsMap &params,
        const infinicore::Tensor &input,
        bool has_bias,
        float alpha = 1.0f) const = 0;

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
        const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
        const std::vector<SplitInfo> &splits,
        int narrow_dim,
        int tp_rank, int tp_size, int tp_num_heads) const = 0;

    // Post-loading weight processing (e.g., GPTQ->GPTQ_QY conversion).
    // Returns a replacement quantization object if the scheme changed (e.g. GPTQ -> GPTQ_QY),
    // or nullptr if no replacement is needed.
    virtual std::shared_ptr<BaseQuantization> process_weights_after_loading(
        ParamsMap &params,
        const infinicore::Device &device,
        int split_dim = -1) const {
        (void)params;
        (void)device;
        (void)split_dim;
        return nullptr;
    }

    // Reset transient buffers whose contents affect the next kernel launch.
    // Marlin currently uses a zero-initialized lock/workspace region; stale
    // lock values from warmup, graph capture, or a previous graph replay can
    // make the next launch wait forever. Quantization schemes without such
    // runtime state can keep the default no-op implementation.
    virtual void reset_runtime_state() const {}

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
