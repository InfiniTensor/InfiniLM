#include "qwen3_moe_experts.hpp"

#include "infinicore/ops.hpp"

#include <string>

namespace infinilm::models::qwen3_moe {

Qwen3MoeExperts::Qwen3MoeExperts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                 const infinicore::Device &device) {

    num_experts_ = model_config->get<size_t>("num_experts");
    num_experts_per_tok_ = model_config->get<size_t>("num_experts_per_tok");

    ASSERT((num_experts_ > 0) && (num_experts_per_tok_ > 0) && (num_experts_per_tok_ <= num_experts_));

    for (size_t i = 0; i < num_experts_; ++i) {
        experts_.push_back(this->register_module<Qwen3MoeMLP>(std::to_string(i), model_config, device));
    }
}

infinicore::Tensor Qwen3MoeExperts::forward(const infinicore::Tensor &hidden_states,
                                            const infinicore::Tensor &top_k_index,
                                            const infinicore::Tensor &top_k_weights) const {
    ASSERT(hidden_states->ndim() == 2);

    auto top_k_weights_cpu = top_k_weights->to(infinicore::Device::Type::CPU);
    auto top_k_index_cpu = top_k_index->to(infinicore::Device::Type::CPU);

    int *top_k_index_ptr = reinterpret_cast<int *>(top_k_index_cpu->data());
    float *top_k_weights_ptr = reinterpret_cast<float *>(top_k_weights_cpu->data());

    size_t ntoken = hidden_states->shape()[0];
    int index;
    float score;

    auto final_hidden_states = infinicore::Tensor::empty(hidden_states->shape(), hidden_states->dtype(), hidden_states->device());
    for (size_t itok = 0; itok < ntoken; ++itok) {
        auto hidden_states_i = hidden_states->narrow({{0, itok, 1}});
        const size_t route_row = itok * num_experts_per_tok_;

        infinicore::Tensor final_hidden_states_i;
        for (size_t k = 0; k < num_experts_per_tok_; ++k) {
            index = top_k_index_ptr[route_row + k];
            score = top_k_weights_ptr[route_row + k];

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

} // namespace infinilm::models::qwen3_moe
