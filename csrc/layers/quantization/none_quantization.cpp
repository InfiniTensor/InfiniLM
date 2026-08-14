#include "none_quantization.hpp"
#include "../../global_state/global_state.hpp"
#include "infinicore/ops/linear.hpp"
#include "infinicore/ops/linear_allreduce.hpp"
#include <optional>

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

    auto input_contiguous = input->is_contiguous() ? input : input->contiguous();
    auto weight = params.at("weight");

    std::optional<infinicore::Tensor> bias_opt;
    if (has_bias) {
        bias_opt = params.at("bias");
    }

    // Ascend path: weight was pre-packed to [IC, OC] in process_weights_after_loading.
    // Use linear_packed to skip the runtime permute({1,0}).
    if (weight_prepacked_) {
        return infinicore::op::linear_packed(input_contiguous, weight, bias_opt, alpha);
    }
    return infinicore::op::linear(input_contiguous->contiguous(), weight->contiguous(), bias_opt, alpha);
}

infinicore::Tensor NoneQuantization::forward_allreduce(
    const ParamsMap &params,
    const infinicore::Tensor &input,
    bool has_bias,
    infinicclComm_t communicator,
    float alpha) const {
    if (alpha != 1.0f) {
        return BaseQuantization::forward_allreduce(
            params, input, has_bias, communicator, alpha);
    }

    auto input_contiguous = input->is_contiguous()
                              ? input
                              : input->contiguous();
    auto weight = params.at("weight");
    std::optional<infinicore::Tensor> bias_opt;
    if (has_bias) {
        bias_opt = params.at("bias");
    }

    if (weight_prepacked_) {
        return infinicore::op::linear_allreduce_packed(
            input_contiguous, weight, bias_opt, communicator);
    }
    return infinicore::op::linear_allreduce(
        input_contiguous, weight->contiguous(), bias_opt, communicator);
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

std::shared_ptr<BaseQuantization> NoneQuantization::process_weights_after_loading(
    ParamsMap &params,
    const infinicore::Device &device,
    int /*split_dim*/) const {

    // Controlled by --pre-transpose CLI flag, default off.
    if (!global_state::get_infinilm_config().pre_transpose) {
        return nullptr;
    }

    auto weight_it = params.find("weight");
    if (weight_it != params.end()) {
        // Transpose weight from [OC, IC] to [IC, OC] once.
        // contiguous() materializes the transposed layout so that
        // subsequent forwards can feed it directly to GEMM.
        params["weight"] = weight_it->second->permute({1, 0})->contiguous();

        // Mark as pre-packed so forward() uses linear_packed.
        weight_prepacked_ = true;
    }

    // Must return non-null so that BaseLinear::process_weights_after_loading
    // writes the modified params back into parameters_.
    // Returning shared_from_this() triggers the "quantization changed" path
    // which calls parameters_.clear() + re-insert from params.
    return std::const_pointer_cast<BaseQuantization>(shared_from_this());
}

} // namespace infinilm::quantization
