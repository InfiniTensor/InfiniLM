#include "minicpm_sala_mlp.hpp"

#include "infinicore/ops.hpp"

namespace infinilm::models::minicpm_sala {

MiniCPMSALAMLP::MiniCPMSALAMLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                               const infinicore::Device &device) {
    // Match parameter dtype with checkpoint `torch_dtype` (e.g. BF16 for MiniCPM-SALA).
    const auto dtype = model_config->get_dtype();
    hidden_size_ = model_config->get<size_t>("hidden_size");
    intermediate_size_ = model_config->get<size_t>("intermediate_size");

    INFINICORE_NN_MODULE_INIT(gate_proj, hidden_size_, intermediate_size_, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(up_proj, hidden_size_, intermediate_size_, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(down_proj, intermediate_size_, hidden_size_, false, dtype, device);
}

infinicore::Tensor MiniCPMSALAMLP::forward(const infinicore::Tensor &x) const {
    auto x_mut = x;
    auto gate = gate_proj_->forward(x_mut);
    auto up = up_proj_->forward(x_mut);

    // SwiGLU: silu(gate) * up — fused single kernel (swiglu(a,b) = a*b*sigmoid(b) => swiglu(up,gate))
    auto act = infinicore::op::swiglu(up, gate);

    auto act_mut = act;
    return down_proj_->forward(act_mut);
}

} // namespace infinilm::models::minicpm_sala

