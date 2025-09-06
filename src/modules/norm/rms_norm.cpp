#include "rms_norm.hpp"

#include "../../context/inference_context.hpp"

namespace infinicore::nn::module {
std::shared_ptr<RMSNorm> RMSNorm::init(size_t dim, infiniDtype_t dtype) {
    auto rms_norm = std::shared_ptr<RMSNorm>(new RMSNorm());
    rms_norm->weight = Tensor::weight(nullptr, dtype, {dim});
    return rms_norm;
}

void RMSNorm::register_weights(infinicore::weights::Loader &loader, const std::string &name_prefix, int rank) {
    loader.register_weight(name_prefix + ".weight", weight, rank, weights::DistributionType::FULL);
}

void RMSNorm::forward(std::shared_ptr<Tensor> output, std::shared_ptr<Tensor> input) {
    getInferenceContext().rmsnorm(output, input, this->weight, 1e-6);
}
} // namespace infinicore::nn::module