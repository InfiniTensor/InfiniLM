#include "deepseek_v4_mlp.hpp"

#include "../../global_state/global_state.hpp"
#include "deepseek_v4_linear.hpp"
#include "infinicore/ops.hpp"

#include <stdexcept>
#include <string>

namespace infinilm::models::deepseek_v4 {

DeepseekV4MLP::DeepseekV4MLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                             const infinicore::Device &device)
    : DeepseekV4MLP(model_config, model_config->get<size_t>("moe_intermediate_size"), device) {
}

DeepseekV4MLP::DeepseekV4MLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                             size_t intermediate_size,
                             const infinicore::Device &device)
    : hidden_size_(model_config->get<size_t>("hidden_size")),
      intermediate_size_(intermediate_size) {
    const auto &dtype = model_config->get_dtype();
    const auto hidden_act = model_config->get_or<std::string>("hidden_act", "silu");
    if (hidden_act != "silu") {
        throw std::runtime_error("DeepseekV4MLP only supports silu activation");
    }

    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    auto quantization_method = deepseek_v4_linear_quantization(model_config, true);
    INFINICORE_NN_MODULE_INIT(w1, hidden_size_, intermediate_size_, quantization_method, false, dtype, device,
                              rank_info.tp_rank, rank_info.tp_size);
    INFINICORE_NN_MODULE_INIT(w2, intermediate_size_, hidden_size_, quantization_method, false, dtype, device,
                              rank_info.tp_rank, rank_info.tp_size, rank_info.comm);
    INFINICORE_NN_MODULE_INIT(w3, hidden_size_, intermediate_size_, quantization_method, false, dtype, device,
                              rank_info.tp_rank, rank_info.tp_size);
}

infinicore::Tensor DeepseekV4MLP::forward(const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    auto gate = w1_->forward(hidden_states_mutable);
    auto up = w3_->forward(hidden_states_mutable);
    auto intermediate = [&]() {
        // dsv4 op test: dsv4_silu_and_mul
        if (false) {
            return infinicore::op::dsv4_silu_and_mul(gate, up);
        }
        return infinicore::op::swiglu(up, gate);
    }();
    return w2_->forward(intermediate);
}

infinicore::Tensor DeepseekV4MLP::forward_without_allreduce(const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    auto gate = w1_->forward(hidden_states_mutable);
    auto up = w3_->forward(hidden_states_mutable);
    auto intermediate = [&]() {
        // dsv4 op test: dsv4_silu_and_mul
        if (false) {
            return infinicore::op::dsv4_silu_and_mul(gate, up);
        }
        return infinicore::op::swiglu(up, gate);
    }();
    return w2_->forward_without_allreduce(intermediate);
}

} // namespace infinilm::models::deepseek_v4
