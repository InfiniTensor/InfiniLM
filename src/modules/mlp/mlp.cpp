#include "mlp.hpp"

#include "../../context/inference_context.hpp"

namespace infinicore::nn::module {
std::shared_ptr<MLP> MLP::init(size_t hidden_size, size_t intermediate_size, infiniDtype_t dtype, int nranks) {
    auto mlp = std::shared_ptr<MLP>(new MLP());

    return mlp;
}

void MLP::register_weights(infinicore::weights::Loader &loader, const std::string &name_prefix, int rank) {
}
void MLP::forward(std::shared_ptr<Tensor> output, std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> residual, std::shared_ptr<Tensor> gate_up_buf) {
}

} // namespace infinicore::nn::module