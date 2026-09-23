#include "awq_marlin.hpp"

#include <stdexcept>

namespace infinilm::quantization {

std::vector<ParamDescriptor> AWQMarlin::get_param_layout(
    size_t, size_t, int, int, int, int, const infinicore::DataType &, bool) const {
    return {};
}

infinicore::Tensor AWQMarlin::forward(
    const ParamsMap &,
    const infinicore::Tensor &,
    bool,
    float) const {
    throw std::runtime_error(
        "AWQ Marlin quantization is unsupported because InfiniLM has no InfiniOps-backed Marlin GEMM path.");
}

std::vector<SplitParam> AWQMarlin::split_params(
    const std::unordered_map<std::string, infinicore::nn::Parameter> &,
    const std::vector<SplitInfo> &,
    int,
    int, int, int) const {
    return {};
}

void AWQMarlin::reset_runtime_state() const {}

} // namespace infinilm::quantization
