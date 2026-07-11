#include "none_quantization.hpp"
#include "../../engine/workspace/tensor_allocator.hpp"
#include "infinicore/ops/linear.hpp"
#include <optional>
#include <stdexcept>

namespace infinilm::quantization {

NoneQuantization::NoneQuantization() : NoneQuantization(nlohmann::json()) {}

std::vector<ParamDescriptor> NoneQuantization::get_param_layout(
    size_t in_features, size_t out_features,
    int split_dim, int tp_rank, int tp_size,
    int /*tp_num_heads*/,
    const infinicore::DataType &dtype,
    bool bias) const {

    std::vector<ParamDescriptor> descs;
    descs.push_back({"weight", {out_features, in_features}, dtype, split_dim, tp_rank, tp_size});
    if (bias) {
        descs.push_back({"bias", {out_features}, dtype, split_dim >= 0 ? 0 : -1, split_dim >= 0 ? tp_rank : 0, split_dim >= 0 ? tp_size : 1});
    }
    return descs;
}

infinicore::Tensor NoneQuantization::forward(
    const ParamsMap &params,
    const infinicore::Tensor &input,
    bool has_bias,
    float alpha) const {
    return forward(params, input, has_bias, alpha, nullptr);
}

infinicore::Tensor NoneQuantization::forward(
    const ParamsMap &params,
    const infinicore::Tensor &input,
    bool has_bias,
    float alpha,
    const infinicore::Tensor *preallocated_output) const {

    auto input_contiguous = input->is_contiguous() ? input : input->contiguous();
    auto weight = params.at("weight");

    std::optional<infinicore::Tensor> bias_opt;
    if (has_bias) {
        bias_opt = params.at("bias");
    }

    auto input_for_linear = input_contiguous->contiguous();
    auto weight_for_linear = weight->contiguous();
    auto output_shape = input_for_linear->shape();
    output_shape.back() = weight_for_linear->shape()[0];

    infinicore::Tensor output;
    if (preallocated_output != nullptr) {
        output = *preallocated_output;
        if (output->shape() != output_shape || output->dtype() != input_for_linear->dtype() || output->device() != input_for_linear->device()) {
            throw std::runtime_error("preallocated NoneQuantization output does not match linear output shape, dtype, or device");
        }
    } else {
        output = infinilm::engine::allocate_inference_tensor(
            "linear.none.output",
            output_shape,
            input_for_linear->dtype(),
            input_for_linear->device());
    }
    infinicore::op::linear_(output, input_for_linear, weight_for_linear, bias_opt, alpha);
    return output;
}

std::vector<SplitParam> NoneQuantization::split_params(
    const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
    const std::vector<SplitInfo> &splits,
    int narrow_dim,
    int tp_rank, int tp_size, int /*tp_num_heads*/) const {

    std::vector<SplitParam> result;
    auto weight_it = params.find("weight");
    auto bias_it = params.find("bias");

    for (const auto &s : splits) {
        result.push_back({s.prefix + ".weight",
                          infinicore::nn::Parameter(
                              weight_it->second->narrow({{static_cast<size_t>(narrow_dim), s.start, s.size}}),
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
