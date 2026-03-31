#include "mlp.hpp"
#include "../../global_state/global_state.hpp"
#include "infinicore/ops.hpp"

namespace infinilm::layers::mlp {

MLP::MLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
         const infinicore::Device &device) {

    const auto &dtype{model_config->get_dtype()};
    hidden_size_ = model_config->get<size_t>("hidden_size");
    intermediate_size_ = model_config->get<size_t>("intermediate_size");
    use_bias_ = model_config->get_or<bool>("mlp_bias", false);

    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    int tp_rank = rank_info.tp_rank;
    int tp_size = rank_info.tp_size;

    auto quant_scheme = model_config->get_quant_scheme();
    auto quantization_method = model_config->get_quantization_method();
    switch (quant_scheme) {
    case infinicore::quantization::QuantScheme::NONE: {
        INFINILM_GATE_UP_LINEAR_INIT(gate_up_proj, "gate_proj", "up_proj", hidden_size_, intermediate_size_, quantization_method,
                                     use_bias_, dtype, device, rank_info);
        INFINICORE_NN_MODULE_INIT(down_proj, intermediate_size_, hidden_size_, quantization_method,
                                  use_bias_, dtype, device, tp_rank, tp_size, rank_info.comm);
        break;
    }
    case infinicore::quantization::QuantScheme::COMPRESSED_TENSOR_W8A8I8: {
        INFINILM_GATE_UP_LINEAR_W8A8_INIT(gate_up_proj, "gate_proj", "up_proj", hidden_size_, intermediate_size_, quantization_method,
                                          use_bias_, dtype, device, rank_info);
        INFINICORE_NN_MODULE_INIT(down_proj, intermediate_size_, hidden_size_, quantization_method,
                                  use_bias_, dtype, device, tp_rank, tp_size, rank_info.comm);
        break;
    }
    case infinicore::quantization::QuantScheme::AWQ_W4A16: {
        INFINILM_GATE_UP_LINEAR_W4A16AWQ_INIT(gate_up_proj, "gate_proj", "up_proj", hidden_size_, intermediate_size_, quantization_method,
                                              use_bias_, dtype, device, rank_info);
        INFINICORE_NN_MODULE_INIT(down_proj, intermediate_size_, hidden_size_, quantization_method,
                                  use_bias_, dtype, device, tp_rank, tp_size, rank_info.comm);
        break;
    }
    default: {
        throw std::runtime_error("infinilm::layers::mlp::MLP: unsupported quantization scheme");
        break;
    }
    }
}

infinicore::Tensor MLP::forward(const infinicore::Tensor &hidden_states) const {
    // 1. Project to gate and up
    auto hidden_states_mutable = hidden_states;
    auto [gate, up] = gate_up_proj_->forward_split(hidden_states_mutable);
    // 2. Apply SwiGLU: silu(gate) * up
    auto intermediate = infinicore::op::swiglu(up, gate);
    // 3. Project down
    auto output = down_proj_->forward(intermediate);
    return output;
}
} // namespace infinilm::layers::mlp
