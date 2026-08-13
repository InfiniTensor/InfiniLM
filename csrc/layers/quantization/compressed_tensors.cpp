#include "compressed_tensors.hpp"
#include "infinicore/ops/linear_w8a8i8.hpp"
#include "infinicore/ops/mul_scalar.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <optional>
#include <string>

namespace infinilm::quantization {
namespace {

std::string lower_string(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool string_field_equals(const nlohmann::json &json, const char *key, const char *expected) {
    auto it = json.find(key);
    return it != json.end() && it->is_string() && lower_string(it->get<std::string>()) == expected;
}

bool bool_field_equals(const nlohmann::json &json, const char *key, bool expected) {
    auto it = json.find(key);
    return it != json.end() && it->is_boolean() && it->get<bool>() == expected;
}

bool integer_field_equals(const nlohmann::json &json, const char *key, int expected) {
    auto it = json.find(key);
    return it != json.end() && it->is_number_integer() && it->get<int>() == expected;
}

bool has_linear_or_moe_target(const nlohmann::json &group) {
    auto targets = group.find("targets");
    if (targets == group.end() || !targets->is_array()) {
        return false;
    }
    for (const auto &target : *targets) {
        if (!target.is_string()) {
            continue;
        }
        const auto value = lower_string(target.get<std::string>());
        if (value == "linear" || value == "fusedmoe") {
            return true;
        }
    }
    return false;
}

bool is_dynamic_token_w8a8_group(const nlohmann::json &group) {
    auto weights_it = group.find("weights");
    auto input_it = group.find("input_activations");
    if (weights_it == group.end() || input_it == group.end() ||
        !weights_it->is_object() || !input_it->is_object()) {
        return false;
    }
    const auto &weights = *weights_it;
    const auto &input = *input_it;
    const bool weight_ok =
        string_field_equals(weights, "type", "int") &&
        string_field_equals(weights, "strategy", "channel") &&
        integer_field_equals(weights, "num_bits", 8) &&
        bool_field_equals(weights, "symmetric", true);
    const bool input_ok =
        string_field_equals(input, "type", "int") &&
        string_field_equals(input, "strategy", "token") &&
        integer_field_equals(input, "num_bits", 8) &&
        bool_field_equals(input, "dynamic", true) &&
        bool_field_equals(input, "symmetric", true);
    return weight_ok && input_ok;
}

} // namespace

std::vector<ParamDescriptor> CompressedTensors::get_param_layout(
    size_t in_features, size_t out_features,
    int split_dim, int tp_rank, int tp_size,
    int /*tp_num_heads*/,
    const infinicore::DataType &dtype,
    bool bias) const {

    std::vector<ParamDescriptor> descs;
    descs.push_back({"weight", {out_features, in_features}, infinicore::DataType::I8, split_dim, tp_rank, tp_size});
    // weight_scale is per-output-channel [out_features, 1]; always split on
    // dim0 (output dimension) for ColumnParallel, and don't split for RowParallel.
    int scale_split_dim = (split_dim == 0) ? 0 : -1;
    int scale_tp_size = (split_dim == 0) ? tp_size : 1;
    int scale_tp_rank = (split_dim == 0) ? tp_rank : 0;
    descs.push_back({"weight_scale", {out_features, 1}, infinicore::DataType::F32, scale_split_dim, scale_tp_rank, scale_tp_size});
    if (bias) {
        descs.push_back({"bias", {out_features}, dtype, -1, 0, 1});
    }
    return descs;
}

std::string CompressedTensors::get_moe_weight_method(const infinicore::Device &device) const {
    if (device.getType() != infinicore::Device::Type::HYGON || !quant_config_.is_object()) {
        return "dense";
    }
    if (!string_field_equals(quant_config_, "quant_method", "compressed-tensors")) {
        return "dense";
    }
    auto groups = quant_config_.find("config_groups");
    if (groups == quant_config_.end() || !groups->is_object()) {
        return "dense";
    }
    for (const auto &item : groups->items()) {
        const auto &group = item.value();
        if (group.is_object() && has_linear_or_moe_target(group) && is_dynamic_token_w8a8_group(group)) {
            return "slimquant_marlin";
        }
    }
    return "dense";
}

infinicore::Tensor CompressedTensors::forward(
    const ParamsMap &params,
    const infinicore::Tensor &input,
    bool has_bias,
    float alpha) const {

    auto input_contiguous = input->is_contiguous() ? input : input->contiguous();
    auto weight = params.at("weight");
    auto weight_scale = params.at("weight_scale");

    std::optional<infinicore::Tensor> bias_opt;
    if (has_bias) {
        bias_opt = params.at("bias");
    }

    auto effective_weight_scale = weight_scale;
    if (std::fabs(alpha - 1.0f) > 1e-7f) {
        effective_weight_scale = infinicore::op::mul_scalar(weight_scale, static_cast<double>(alpha));
    }

    return infinicore::op::linear_w8a8i8(input_contiguous->contiguous(), weight, effective_weight_scale, bias_opt);
}

std::vector<SplitParam> CompressedTensors::split_params(
    const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
    const std::vector<SplitInfo> &splits,
    int narrow_dim,
    int tp_rank, int tp_size, int /*tp_num_heads*/) const {

    std::vector<SplitParam> result;
    auto weight_it = params.find("weight");
    auto scale_it = params.find("weight_scale");
    auto bias_it = params.find("bias");

    for (const auto &s : splits) {
        result.push_back({s.prefix + ".weight",
                          infinicore::nn::Parameter(
                              weight_it->second->narrow({{static_cast<size_t>(narrow_dim), s.start, s.size}}),
                              narrow_dim, tp_rank, tp_size, s.num_shards)});
        result.push_back({s.prefix + ".weight_scale",
                          infinicore::nn::Parameter(
                              scale_it->second->narrow({{static_cast<size_t>(narrow_dim), s.start, s.size}}),
                              narrow_dim, tp_rank, tp_size, s.num_shards)});
        if (bias_it != params.end()) {
            result.push_back({s.prefix + ".bias",
                              infinicore::nn::Parameter(
                                  bias_it->second->narrow({{0, s.start, s.size}}),
                                  0, tp_rank, tp_size, s.num_shards)});
        }
    }
    return result;
}

} // namespace infinilm::quantization
