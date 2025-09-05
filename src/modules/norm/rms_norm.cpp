#include "rms_norm.hpp"

#include "../../context/inference_context.hpp"

namespace infinicore::nn::module {
std::shared_ptr<RMSNorm> RMSNorm::init(size_t dim, infiniDtype_t dtype) {
    auto rms_norm = std::shared_ptr<RMSNorm>(new RMSNorm());

    return rms_norm;
}
void RMSNorm::register_weights(infinicore::weights::Loader &loader, const std::string &name_prefix, int rank) {}
void RMSNorm::forward(std::shared_ptr<Tensor> output, std::shared_ptr<Tensor> input) {}
} // namespace infinicore::nn::module