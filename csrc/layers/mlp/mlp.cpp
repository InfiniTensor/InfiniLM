#include "mlp.hpp"
#include "../../engine/parallel_state.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/ops.hpp"
#include <cstddef>

namespace infinilm::layers::mlp {

MLP::MLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
         const infinicore::Device &device)
    : model_config_(model_config) {

    hidden_size_ = model_config->get<size_t>("hidden_size");
    intermediate_size_ = model_config->get<size_t>("intermediate_size");
    use_bias_ = model_config->get_or<bool>("mlp_bias", false);

    const auto &dtype{model_config_->get_dtype()};

    const engine::distributed::RankInfo &rank_info = infinilm::engine::get_tensor_model_parallel_rank_info();
    int tp_rank = rank_info.tp_rank;
    int tp_size = rank_info.tp_size;

    // Initialize projection layers
    auto quant_scheme = this->model_config_->get_quant_scheme();
    auto quantization_method = this->model_config_->get_quantization_method();

    switch (quant_scheme) {
    case infinicore::quantization::QuantScheme::COMPRESSED_TENSOR_W8A8I8:
        INFINILM_GATE_UP_LINEAR_W8A8_INIT(gate_up_proj, "gate_proj", "up_proj", hidden_size_, intermediate_size_, quantization_method,
                                          use_bias_, dtype, device, rank_info);
        INFINICORE_NN_MODULE_INIT(down_proj, intermediate_size_, hidden_size_, quantization_method, use_bias_,
                                  dtype, device, tp_rank, tp_size, rank_info.comm);
        break;
    case infinicore::quantization::QuantScheme::AWQ_W4A16:
        INFINILM_GATE_UP_LINEAR_W4A16AWQ_INIT(gate_up_proj, "gate_proj", "up_proj", hidden_size_, intermediate_size_, quantization_method,
                                              use_bias_, dtype, device, rank_info);
        INFINICORE_NN_MODULE_INIT(down_proj, intermediate_size_, hidden_size_, quantization_method, use_bias_,
                                  dtype, device, tp_rank, tp_size, rank_info.comm);
        break;

    default:
        INFINILM_GATE_UP_LINEAR_INIT(gate_up_proj, "gate_proj", "up_proj", hidden_size_, intermediate_size_, quantization_method,
                                     use_bias_, dtype, device, rank_info);
        INFINICORE_NN_MODULE_INIT(down_proj, intermediate_size_, hidden_size_, quantization_method, use_bias_,
                                  dtype, device, tp_rank, tp_size, rank_info.comm);
        break;
    }
}

infinicore::Tensor MLP::forward(const infinicore::Tensor &hidden_states) const {
    // 1. Project to gate and up
    auto hidden_states_mutable = hidden_states;
    auto [gate, up] = gate_up_proj_->forward_split(hidden_states_mutable);
    // 2. Apply SwiGLU: silu(gate) * up
    // Note: swiglu kernel expects (up, gate) and computes gate * sigmoid(gate) * up
    // So we pass (up, gate) to get the correct result: gate * sigmoid(gate) * up
    auto intermediate = infinicore::op::swiglu(up, gate);

    // 3. Project down
    auto output = down_proj_->forward(intermediate);
    return output;
}
} // namespace infinilm::layers::mlp
