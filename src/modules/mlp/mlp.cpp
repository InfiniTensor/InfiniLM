#include "mlp.hpp"

#include "../../context/inference_context.hpp"

namespace infinicore::nn::module {
std::shared_ptr<MLP> MLP::init(size_t hidden_size, size_t intermediate_size, infiniDtype_t dtype, int nranks) {
    auto mlp = std::shared_ptr<MLP>(new MLP());
    // Initialize the three linear layers
    mlp->gate = Linear::init(hidden_size, intermediate_size, dtype, false, false,
                             weights::DistributionType::COLUMN, nranks);
    mlp->up = Linear::init(hidden_size, intermediate_size, dtype, false, false,
                           weights::DistributionType::COLUMN, nranks);
    mlp->down = Linear::init(intermediate_size, hidden_size, dtype, false, false,
                             weights::DistributionType::ROW, nranks);

    return mlp;
}

void MLP::register_weights(infinicore::weights::Loader &loader, const std::string &name_prefix, int rank) {
    if (gate != nullptr) {
        gate->register_weights(loader, name_prefix + ".gate_proj.weight", rank);
    }
    if (up != nullptr) {
        up->register_weights(loader, name_prefix + ".up_proj.weight", rank);
    }
    if (down != nullptr) {
        down->register_weights(loader, name_prefix + ".down_proj.weight", rank);
    }
}

void MLP::forward(std::shared_ptr<Tensor> output, std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> residual, std::shared_ptr<Tensor> gate_up_buf) {
    gate->forward(gate_up_buf, input, nullptr);
    size_t di = gate_up_buf->shape()[0] / 2;
    auto gate_buf = gate_up_buf->slice(1, 0, di);
    auto up_buf = gate_up_buf->slice(1, di, di);
    getInferenceContext().swiglu(gate_buf, up_buf, gate_buf);
    down->forward(output, gate_buf, residual);
}

} // namespace infinicore::nn::module