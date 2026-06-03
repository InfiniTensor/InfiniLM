#include "awq_marlin.hpp"
#include "marlin_utils.hpp"

#include "infinicore/ops/add.hpp"
#include "infinicore/ops/gptq_marlin_gemm.hpp"

#include <stdexcept>

namespace infinilm::quantization {

std::vector<ParamDescriptor> AWQMarlin::get_param_layout(
    size_t, size_t, int, int, int, int, const infinicore::DataType &, bool) const {
    return {};
}

infinicore::Tensor AWQMarlin::forward(
    const ParamsMap &params,
    const infinicore::Tensor &input,
    bool has_bias,
    float /*alpha*/) const {
    auto input_contiguous = input->is_contiguous() ? input : input->contiguous();
    const auto &shape = input_contiguous->shape();
    const size_t k = shape.back();
    const size_t m = input_contiguous->numel() / k;
    auto flat_input = input_contiguous->view({m, k});
    auto output = infinicore::Tensor::empty({m, output_size_per_partition_}, input->dtype(), input->device());

    auto qweight = params.at("qweight");
    auto scales = params.at("scales");
    auto qzeros = params.at("qzeros");
    auto g_idx = params.at("g_idx");
    auto perm = params.at("perm");
    auto global_scales = params.at("global_scales");

    infinicore::op::gptq_marlin_gemm_(
        output,
        flat_input,
        qweight,
        scales,
        global_scales,
        qzeros,
        g_idx,
        perm,
        marlin::UINT4_ID,
        true,
        false,
        true,
        false);

    if (has_bias) {
        auto bias = params.at("bias");
        infinicore::op::add_(output, output, bias->as_strided(output->shape(), {0, 1}));
    }

    auto out_shape = shape;
    out_shape.back() = output_size_per_partition_;
    return output->view(out_shape);
}

std::vector<SplitParam> AWQMarlin::split_params(
    const std::unordered_map<std::string, infinicore::nn::Parameter> &,
    const std::vector<SplitInfo> &,
    int,
    int, int, int) const {
    return {};
}

} // namespace infinilm::quantization
