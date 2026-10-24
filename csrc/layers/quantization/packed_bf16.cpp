#include "packed_bf16.hpp"

#include "infinicore/ops/linear_allreduce.hpp"
#include "infinicore/ops/linear.hpp"

#include <optional>
#include <string>
#include <stdexcept>

namespace infinilm::quantization {
namespace {
std::string layout_format(const BaseQuantization &quantization) {
    return quantization.get_config().value("format", "lossless-layout");
}

bool packed_projection(size_t in_features, size_t out_features, const std::string &format) {
    const bool qwen_qkv = in_features == 4096 && out_features == 6144;
    const bool qwen35_qkv = in_features == 4096 && out_features == 10240;
    const bool qwen_down = in_features == 12288;
    if (format == "lossless-layout") {
        return true;
    }
    if (format == "qkv-v1") {
        return qwen_qkv;
    }
    if (format == "down-v1") {
        return qwen_down;
    }
    if (format == "mixed-v1") {
        return qwen_qkv || qwen35_qkv || qwen_down;
    }
    return false;
}

bool packed_weight(
    const infinicore::Tensor &weight,
    const std::string &format) {
    if (format == "lossless-layout") {
        return true;
    }
    if (format == "qkv-v1") {
        return weight->size(1) == 6144;
    }
    if (format == "down-v1") {
        return weight->size(0) == 12288;
    }
    if (format == "mixed-v1") {
        return weight->size(1) == 6144 || weight->size(1) == 10240 || weight->size(0) == 12288;
    }
    return false;
}
} // namespace

std::vector<ParamDescriptor> PackedBF16::get_param_layout(
    size_t in_features,
    size_t out_features,
    int split_dim,
    int tp_rank,
    int tp_size,
    int tp_num_heads,
    const infinicore::DataType &dtype,
    bool bias) const {
    if (tp_size != 1) {
        throw std::runtime_error("PackedBF16 currently supports TP size 1 only");
    }
    (void)split_dim;
    (void)tp_rank;
    (void)tp_num_heads;

    std::vector<ParamDescriptor> descs;
    if (packed_projection(in_features, out_features, layout_format(*this))) {
        descs.push_back({"weight", {in_features, out_features}, dtype, -1, 0, 1});
    } else {
        descs.push_back({"weight", {out_features, in_features}, dtype, -1, 0, 1});
    }
    if (bias) {
        descs.push_back({"bias", {out_features}, dtype, -1, 0, 1});
    }
    return descs;
}

infinicore::Tensor PackedBF16::forward(
    const ParamsMap &params,
    const infinicore::Tensor &input,
    bool has_bias,
    float alpha) const {
    auto input_contiguous = input->is_contiguous() ? input : input->contiguous();
    auto weight = params.at("weight");
    std::optional<infinicore::Tensor> bias;
    if (has_bias) {
        bias = params.at("bias");
    }
    if (packed_weight(weight, layout_format(*this))) {
        return infinicore::op::linear_packed(
            input_contiguous, weight, bias, alpha);
    }
    return infinicore::op::linear(
        input_contiguous, weight, bias, alpha);
}

infinicore::Tensor PackedBF16::forward_allreduce(
    const ParamsMap &params,
    const infinicore::Tensor &input,
    bool has_bias,
    infinicclComm_t communicator,
    float alpha) const {
    auto input_contiguous = input->is_contiguous() ? input : input->contiguous();
    auto weight = params.at("weight");
    std::optional<infinicore::Tensor> bias;
    if (has_bias) {
        bias = params.at("bias");
    }
    if (alpha != 1.0f) {
        return BaseQuantization::forward_allreduce(
            params, input, has_bias, communicator, alpha);
    }
    if (packed_weight(weight, layout_format(*this))) {
        return infinicore::op::linear_allreduce_packed(
            input_contiguous, weight, bias, communicator);
    }
    return infinicore::op::linear_allreduce(
        input_contiguous, weight, bias, communicator);
}

int PackedBF16::get_fused_split_dim() const {
    return use_full_layout() ? 1 : 0;
}
bool PackedBF16::use_full_layout() const {
    return layout_format(*this) == "lossless-layout";
}
std::vector<SplitParam> PackedBF16::split_params(
    const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
    const std::vector<SplitInfo> &splits,
    int narrow_dim,
    int tp_rank,
    int tp_size,
    int tp_num_heads) const {
    if (tp_size != 1) {
        throw std::runtime_error("PackedBF16 currently supports TP size 1 only");
    }

    auto weight_it = params.find("weight");
    auto bias_it = params.find("bias");
    const int packed_dim =
        (narrow_dim == 0 && weight_it->second->size(0) <
                     weight_it->second->size(1))
            ? 1
            : narrow_dim;
    std::vector<SplitParam> result;
    result.reserve(splits.size());
    for (const auto &split : splits) {
        result.push_back(
            {split.prefix + ".weight",
             infinicore::nn::Parameter(
                 weight_it->second->narrow(
                     {{static_cast<size_t>(packed_dim), split.start, split.size}}),
                 packed_dim,
                 tp_rank,
                 tp_size,
                 split.num_shards)});
        if (bias_it != params.end()) {
            result.push_back(
                {split.prefix + ".bias",
                 infinicore::nn::Parameter(
                     bias_it->second->narrow({{0, split.start, split.size}}),
                     0,
                     tp_rank,
                     tp_size,
                     split.num_shards)});
        }
    }
    return result;
}

} // namespace infinilm::quantization
