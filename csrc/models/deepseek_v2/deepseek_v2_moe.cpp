#include "deepseek_v2_moe.hpp"

#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"

#include "spdlog/spdlog.h"
#include <stdexcept>

namespace infinilm::models::deepseek_v2 {
namespace {

bool supports_fused_deepseek_moe(infinicore::Device::Type device_type) {
    switch (device_type) {
    case infinicore::Device::Type::kNvidia:
    case infinicore::Device::Type::kHygon:
    case infinicore::Device::Type::kIluvatar:
    case infinicore::Device::Type::kMetax:
    case infinicore::Device::Type::kMoore:
        return true;
    default:
        return false;
    }
}

} // namespace

DeepseekV2TopKRouter::DeepseekV2TopKRouter(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                           const infinicore::Device &device) {
    const auto &dtype{model_config->get_dtype()};
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    num_experts_ = model_config->get<size_t>("num_experts");
    num_experts_per_tok_ = model_config->get<size_t>("num_experts_per_tok");
    norm_topk_prob_ = model_config->get<bool>("norm_topk_prob");

    ASSERT((num_experts_ > 0) && (num_experts_per_tok_ > 0) && (num_experts_per_tok_ <= num_experts_));
    INFINICORE_NN_PARAMETER_INIT(weight, ({num_experts_, hidden_size}, dtype, device));
}

std::tuple<infinicore::Tensor, infinicore::Tensor>
DeepseekV2TopKRouter::forward(const infinicore::Tensor &hidden_states) const {
    ASSERT(hidden_states->ndim() == 2);
    const size_t ntoken = hidden_states->shape()[0];
    auto router_logits = infinicore::op::linear(hidden_states, weight_, std::nullopt, 1.0f);
    auto router_scores = infinicore::Tensor::empty({ntoken, num_experts_per_tok_}, infinicore::DataType::kFloat32, hidden_states->device());
    auto router_indices = infinicore::Tensor::empty({ntoken, num_experts_per_tok_}, infinicore::DataType::kInt32, hidden_states->device());
    infinicore::op::topksoftmax(router_scores, router_indices, router_logits, num_experts_per_tok_, norm_topk_prob_);
    return {router_scores, router_indices};
}

DeepseekV2Experts::DeepseekV2Experts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                     const infinicore::Device &device) {
    hidden_size_ = model_config->get<size_t>("hidden_size");
    moe_intermediate_size_ = model_config->get<size_t>("moe_intermediate_size");
    num_experts_ = model_config->get<size_t>("num_experts");
    num_experts_per_tok_ = model_config->get<size_t>("num_experts_per_tok");
    ASSERT((num_experts_ > 0) && (num_experts_per_tok_ > 0) && (num_experts_per_tok_ <= num_experts_));
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    tp_size_ = static_cast<size_t>(rank_info.tp_size);
    communicator_ = rank_info.comm;

    experts_.reserve(num_experts_);
    gate_weights_.reserve(num_experts_);
    up_weights_.reserve(num_experts_);
    down_weights_.reserve(num_experts_);
    for (size_t i = 0; i < num_experts_; ++i) {
        auto expert = this->register_module<DeepseekV2ExpertMLP>(std::to_string(i), model_config, device);
        gate_weights_.push_back(expert->gate_weight());
        up_weights_.push_back(expert->up_weight());
        down_weights_.push_back(expert->down_weight());
        experts_.push_back(std::move(expert));
    }
    local_moe_intermediate_size_ = gate_weights_.empty() ? moe_intermediate_size_ : gate_weights_.front()->shape()[0];
}

infinicore::Tensor DeepseekV2Experts::forward_cpu_routed_(const infinicore::Tensor &hidden_states,
                                                          const infinicore::Tensor &top_k_index,
                                                          const infinicore::Tensor &top_k_weights) const {
    auto top_k_weights_cpu = top_k_weights->to(infinicore::Device::Type::kCpu);
    auto top_k_index_cpu = top_k_index->to(infinicore::Device::Type::kCpu);
    auto *top_k_index_ptr = reinterpret_cast<int *>(top_k_index_cpu->data());
    auto *top_k_weights_ptr = reinterpret_cast<float *>(top_k_weights_cpu->data());

    const size_t ntoken = hidden_states->shape()[0];
    auto final_hidden_states = infinicore::Tensor::empty(hidden_states->shape(), hidden_states->dtype(), hidden_states->device());
    for (size_t itok = 0; itok < ntoken; ++itok) {
        auto hidden_states_i = hidden_states->narrow({{0, itok, 1}});
        const size_t route_row = itok * num_experts_per_tok_;

        infinicore::Tensor final_hidden_states_i;
        for (size_t k = 0; k < num_experts_per_tok_; ++k) {
            const int index = top_k_index_ptr[route_row + k];
            const float score = top_k_weights_ptr[route_row + k];
            ASSERT(index >= 0 && static_cast<size_t>(index) < num_experts_);
            experts_[index]->set_alpha(score);
            auto expert_out = experts_[index]->forward(hidden_states_i);
            if (k == 0) {
                final_hidden_states_i = expert_out;
            } else {
                infinicore::op::add_(final_hidden_states_i, final_hidden_states_i, expert_out);
            }
        }
        final_hidden_states->narrow({{0, itok, 1}})->copy_from(final_hidden_states_i);
    }
    return final_hidden_states;
}

infinicore::Tensor DeepseekV2Experts::forward(const infinicore::Tensor &hidden_states,
                                              const infinicore::Tensor &top_k_index,
                                              const infinicore::Tensor &top_k_weights) const {
    ASSERT(hidden_states->ndim() == 2);
    if (supports_fused_deepseek_moe(hidden_states->device().type())
        && (hidden_states->dtype() == infinicore::DataType::kBFloat16
            || hidden_states->dtype() == infinicore::DataType::kFloat16)) {
        try {
            auto output = infinicore::op::deepseek_moe(hidden_states, top_k_index, top_k_weights,
                                                       gate_weights_, up_weights_, down_weights_,
                                                       local_moe_intermediate_size_, num_experts_);
            if (tp_size_ > 1 && communicator_ != nullptr) {
                infinicore::op::distributed::allreduce_(output, output, infinicclSum, communicator_);
            }
            return output;
        } catch (const std::exception &e) {
            spdlog::warn("DeepseekV2Experts: deepseek_moe unavailable on {}, falling back to CPU-routed experts: {}",
                         static_cast<int>(hidden_states->device().type()), e.what());
        }
    }
    return forward_cpu_routed_(hidden_states, top_k_index, top_k_weights);
}

DeepseekV2MoE::DeepseekV2MoE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                             const infinicore::Device &device) {
    INFINICORE_NN_MODULE_INIT(gate, model_config, device);
    INFINICORE_NN_MODULE_INIT(experts, model_config, device);

    const size_t n_shared_experts = model_config->get_or<size_t>("n_shared_experts", 0);
    has_shared_experts_ = n_shared_experts > 0;
    if (has_shared_experts_) {
        auto shared_config_json = model_config->get_config_json();
        shared_config_json["intermediate_size"] = model_config->get<size_t>("moe_intermediate_size") * n_shared_experts;
        auto shared_config = std::make_shared<infinilm::config::ModelConfig>(shared_config_json);
        INFINICORE_NN_MODULE_INIT(shared_experts, shared_config, device);
    }
}

infinicore::Tensor DeepseekV2MoE::forward(const infinicore::Tensor &hidden_states) const {
    ASSERT(hidden_states->ndim() == 3);
    const auto shape = hidden_states->shape();
    auto hidden_states_reshaped = hidden_states->view({shape[0] * shape[1], shape[2]});

    auto [routing_weights, selected_experts] = gate_->forward(hidden_states_reshaped);
    auto final_hidden_states = experts_->forward(hidden_states_reshaped, selected_experts, routing_weights)->view(shape);
    if (has_shared_experts_) {
        auto shared_out = shared_experts_->forward(hidden_states);
        final_hidden_states = infinicore::op::add(final_hidden_states, shared_out);
    }
    return final_hidden_states;
}

} // namespace infinilm::models::deepseek_v2
