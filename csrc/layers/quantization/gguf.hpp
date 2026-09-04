#pragma once

#include "base_quantization.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>

namespace infinilm::quantization {

// GGUF block quantization stores the original block bytes in safetensors.
// Each row has row_bytes(in_features, type) bytes rather than in_features
// scalar elements, so the checkpoint tensor name determines the layout.
//
// config.json:quantization_config.ggml_types maps safetensors names to either
// a ggml type id or "dense_bf16" for tensors dequantized during conversion.
// key_prefix is removed once because nested modules do not know their absolute
// checkpoint path.
class GGUFBlockQuantization : public BaseQuantization {
public:
    // Sentinel for entries stored as dense BF16 rather than GGUF blocks.
    static constexpr int64_t DENSE_BF16 = -1;
    // Checkpoint suffix for raw block data; shared with scripts/gguf_mapping.py.
    static constexpr const char *BLOB_SUFFIX = "weight_bytes";
    static constexpr const char *DENSE_SUFFIX = "weight";
    // Parameter-key prefix used for fused Linear shards.
    static constexpr const char *SHARD_PREFIX = "shard";

    explicit GGUFBlockQuantization(const nlohmann::json &quant_config);

    ~GGUFBlockQuantization() override;

    QuantScheme get_quant_scheme() const override {
        return QuantScheme::GGUF_BLOCK;
    }

    // A stem is required to resolve the ggml type.
    std::vector<ParamDescriptor> get_param_layout(
        size_t in_features, size_t out_features,
        int split_dim, int tp_rank, int tp_size,
        int tp_num_heads,
        const infinicore::DataType &dtype,
        bool bias) const override;

    std::vector<ParamDescriptor> get_param_layout(
        size_t in_features, size_t out_features,
        int split_dim, int tp_rank, int tp_size,
        int tp_num_heads,
        const infinicore::DataType &dtype,
        bool bias,
        const std::string &stem) const override;

    infinicore::Tensor forward(
        const ParamsMap &params,
        const infinicore::Tensor &input,
        bool has_bias,
        float alpha = 1.0f) const override;

    infinicore::Tensor forward(
        const ParamsMap &params,
        const infinicore::Tensor &input,
        bool has_bias,
        float alpha,
        const std::string &stem) const override;

    // Fused Linear entry point. Every shard resolves its own ggml type from
    // its checkpoint stem; a shared group stem is insufficient.
    infinicore::Tensor forward(
        const ParamsMap &params,
        const infinicore::Tensor &input,
        bool has_bias,
        float alpha,
        const std::string &stem,
        const std::vector<std::string> &shard_stems) const override;

    // Map shard<i> names to <prefix>.<suffix>. Fused GGUF shards already have
    // independent buffers, so this method does not slice or modify data.
    std::vector<SplitParam> split_params(
        const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
        const std::vector<SplitInfo> &splits,
        int narrow_dim,
        int tp_rank, int tp_size, int tp_num_heads) const override;

    // Validate raw block buffers without modifying their bytes.
    std::shared_ptr<BaseQuantization> process_weights_after_loading(
        ParamsMap &params,
        const infinicore::Device &device,
        int split_dim = -1) const override;

    // Resolve a stem to a ggml type id or DENSE_BF16. Missing and ambiguous
    // matches fail closed instead of silently selecting a dense path.
    // matched_key receives the actual normalized table key when requested.
    int64_t resolve(const std::string &stem, std::string *matched_key = nullptr) const;
    size_t row_bytes(size_t in_features, int64_t type_id) const;
    bool has_group(const std::string &group_stem) const;
    size_t table_size() const { return types_.size(); }

    static bool is_known_type(int64_t type_id);

private:
    // type_id selects the execution path; table_key identifies the exact
    // checkpoint entry in diagnostics.
    infinicore::Tensor forward_shard(
        const std::string &suffix,
        const infinicore::Tensor &weight,
        const infinicore::Tensor &input,
        float alpha,
        int64_t type_id,
        const std::string &table_key) const;

    // Runtime value-head permutation for weights whose columns were exported
    // in tiled order while the GDN kernel produces grouped activations. The
    // converter records rules in quantization_config.activation_vperm because
    // permuting quantized columns would require requantization.
    struct ActVPerm {
        std::string suffix; // Suffix including the trailing '.', for example "linear_attn.out_proj.".
        size_t n_k;         // Number of key heads.
        size_t r;           // Value heads per key head.
        size_t hd;          // value head_dim
    };

    // Match by suffix because layer indices are not part of this local contract.
    const ActVPerm *vperm_rule(const std::string &stem) const;

    // Convert [..., n_k*r*hd] grouped order to [..., r*n_k*hd] tiled order.
    static infinicore::Tensor gather_grouped_to_tiled(
        const ActVPerm &rule, const infinicore::Tensor &input, const std::string &name);

    std::string describe(const std::string &stem) const;

    // Keep the original config key for diagnostics even though lookups use a
    // normalized key with key_prefix removed.
    struct TypeEntry {
        int64_t id;
        std::string name;
    };

    std::unordered_map<std::string, TypeEntry> types_;
    std::string key_prefix_;
    std::vector<ActVPerm> vperm_; // Empty when the converted model needs no permutation.
    // Mutable because layout queries are logically const.
    mutable size_t n_blob_ = 0;
    mutable size_t n_dense_ = 0;
    mutable size_t n_group_ = 0;
};

} // namespace infinilm::quantization
