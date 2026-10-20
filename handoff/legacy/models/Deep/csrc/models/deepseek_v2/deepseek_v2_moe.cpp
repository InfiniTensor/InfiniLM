#include "execution/submit.hpp"
#include "../../weights/weights.hpp"
#include "../../layers/linear/dense.hpp"
#include "deepseek_v2_moe.hpp"
#include <exception>

#include "../../utils.hpp"
#include "runtime/runtime.hpp"
#include <cstdint>
#include <vector>
#include "../../engine/distributed/communication_group.hpp"
#include "execution/graph/graph.hpp"

#include "spdlog/spdlog.h"
#include <stdexcept>

namespace infinilm::models::deepseek_v2 {
namespace {

bool supports_fused_deepseek_moe(infinilm::runtime::Device::Type device_type) {
    switch (device_type) {
    case infinilm::runtime::Device::Type::kNvidia:
    case infinilm::runtime::Device::Type::kHygon:
    case infinilm::runtime::Device::Type::kIluvatar:
    case infinilm::runtime::Device::Type::kMetax:
    case infinilm::runtime::Device::Type::kMoore:
        return true;
    default:
        return false;
    }
}

} // namespace

DeepseekV2TopKRouter::DeepseekV2TopKRouter(infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("DeepseekV2TopKRouter construction requires eager Runtime");
    if (!model_config) throw std::invalid_argument("Deepseek construction requires configuration");
    try {
    const auto &dtype{model_config->get_dtype()};
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    num_experts_ = model_config->get<size_t>("num_experts");
    num_experts_per_tok_ = model_config->get<size_t>("num_experts_per_tok");
    norm_topk_prob_ = model_config->get<bool>("norm_topk_prob");

        if (!num_experts_ || !num_experts_per_tok_ || num_experts_per_tok_ > num_experts_)
            throw std::invalid_argument("Deepseek router expert/topk geometry is invalid");
        weight_ = infinilm::weights::Weight(runtime, {num_experts_, hidden_size}, dtype);
        this->bind_weight("weight", weight_);
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor>
DeepseekV2TopKRouter::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const {
    runtime.assertOwnerThread();
    ASSERT(hidden_states->ndim() == 2);
    const size_t ntoken = hidden_states->shape()[0];
    auto router_logits = infinilm::layers::linear::dense_projection(runtime, hidden_states, weight_.tensor(), std::nullopt, 1.0f);
    auto router_scores = infinilm::runtime::Tensor::empty(runtime, {ntoken, num_experts_per_tok_}, infinilm::runtime::DataType::kFloat32);
    auto router_indices = infinilm::runtime::Tensor::empty(runtime, {ntoken, num_experts_per_tok_}, infinilm::runtime::DataType::kInt32);
    auto topk_scratch = infinilm::runtime::Tensor::empty(runtime, router_scores->shape(), infinilm::runtime::DataType::kInt32);
    infinilm::execution::submit<infini::ops::TopkSoftmax>(
        runtime, infinilm::execution::RecordingPolicy::Unqualified,
        infini::ops::TopkSoftmax::DefaultConfig(runtime.device().type()),
        router_logits, std::optional<infinilm::runtime::Tensor>{},
        std::optional<infinilm::runtime::Tensor>{}, norm_topk_prob_,
        router_scores, router_indices, topk_scratch);
    INFINILM_TRACE_OP(Topksoftmax, router_scores);
    return {router_scores, router_indices};
}

DeepseekV2Experts::DeepseekV2Experts(infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const engine::distributed::RankInfo &rank_info) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("DeepseekV2Experts construction requires eager Runtime");
    if (!model_config || rank_info.device != runtime.device() || rank_info.tp_size <= 0
        || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size
        || (rank_info.tp_size > 1 && !rank_info.comm))
        throw std::invalid_argument("Invalid explicit Deepseek construction resources");
    try {
    hidden_size_ = model_config->get<size_t>("hidden_size");
    moe_intermediate_size_ = model_config->get<size_t>("moe_intermediate_size");
    num_experts_ = model_config->get<size_t>("num_experts");
    num_experts_per_tok_ = model_config->get<size_t>("num_experts_per_tok");
        if (!num_experts_ || !num_experts_per_tok_ || num_experts_per_tok_ > num_experts_)
            throw std::invalid_argument("Deepseek experts/topk geometry is invalid");
    tp_size_ = static_cast<size_t>(rank_info.tp_size);
    communicator_ = rank_info.comm;

    experts_.reserve(num_experts_);
    gate_weights_.reserve(num_experts_);
    up_weights_.reserve(num_experts_);
    down_weights_.reserve(num_experts_);
    for (size_t i = 0; i < num_experts_; ++i) {
            // Publish the member owner before registry/vector operations can throw.
            experts_.push_back(std::make_shared<DeepseekV2ExpertMLP>(runtime, model_config, rank_info));
            this->bind_child(std::to_string(i), experts_.back());
            gate_weights_.push_back(experts_.back()->gate_weight());
            up_weights_.push_back(experts_.back()->up_weight());
            down_weights_.push_back(experts_.back()->down_weight());
    }
    local_moe_intermediate_size_ = gate_weights_.empty() ? moe_intermediate_size_ : gate_weights_.front()->shape()[0];
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor DeepseekV2Experts::forward_cpu_routed_(infinilm::runtime::Runtime &runtime,
                                                          const infinilm::runtime::Tensor &hidden_states,
                                                          const infinilm::runtime::Tensor &top_k_index,
                                                          const infinilm::runtime::Tensor &top_k_weights) const {
    runtime.assertOwnerThread();
    auto final_hidden_states = infinilm::runtime::Tensor::empty(runtime, hidden_states->shape(), hidden_states->dtype());
    // One body for eager and explicit host replay. Never read routes while recording.
    const auto execute = [](infinilm::runtime::Runtime &runtime,
                            const infinilm::runtime::Tensor &hidden_states,
                            const infinilm::runtime::Tensor &top_k_index,
                            const infinilm::runtime::Tensor &top_k_weights,
                            infinilm::runtime::Tensor final_hidden_states,
                            const std::vector<std::shared_ptr<DeepseekV2ExpertMLP>> &experts,
                            size_t num_experts_per_tok) {
    runtime.assertOwnerThread();
    auto contiguous_weights = top_k_weights->contiguous(runtime);
    auto contiguous_indices = top_k_index->contiguous(runtime);
    std::vector<float> host_weights(contiguous_weights->numel());
    std::vector<std::int32_t> host_indices(contiguous_indices->numel());
    // memcpyD2H completes this owner stream before host route reads, and
    // drains before unwind on failure while both destination vectors live.
    runtime.memcpyD2H(host_weights.data(), contiguous_weights->data(), contiguous_weights->nbytes());
    runtime.memcpyD2H(host_indices.data(), contiguous_indices->data(), contiguous_indices->nbytes());
    const auto *top_k_index_ptr = host_indices.data();
    const auto *top_k_weights_ptr = host_weights.data();

    const size_t ntoken = hidden_states->shape()[0];
    for (size_t itok = 0; itok < ntoken; ++itok) {
        auto hidden_states_i = hidden_states->narrow({{0, itok, 1}});
        const size_t route_row = itok * num_experts_per_tok;

        infinilm::runtime::Tensor final_hidden_states_i;
        for (size_t k = 0; k < num_experts_per_tok; ++k) {
            const int index = top_k_index_ptr[route_row + k];
            const float score = top_k_weights_ptr[route_row + k];
            ASSERT(index >= 0 && static_cast<size_t>(index) < experts.size());
            experts[index]->set_alpha(score);
            auto expert_out = experts[index]->forward(runtime, hidden_states_i);
            if (k == 0) {
                final_hidden_states_i = expert_out;
            } else {
                infinilm::execution::submit<infini::ops::Add>(
                    runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), final_hidden_states_i, expert_out, final_hidden_states_i);
                INFINILM_TRACE_OP(Add, final_hidden_states_i);
            }
        }
        final_hidden_states->narrow({{0, itok, 1}})->copy_from(runtime, final_hidden_states_i);
    }
    };
    if (runtime.isGraphRecording()) {
        infinilm::execution::graph::GraphTensor hidden{runtime, hidden_states};
        infinilm::execution::graph::GraphTensor indices{runtime, top_k_index};
        infinilm::execution::graph::GraphTensor weights{runtime, top_k_weights};
        infinilm::execution::graph::GraphTensor output{runtime, final_hidden_states};
        runtime.addGraphOperator(std::make_shared<infinilm::execution::graph::RecordedCall>(
            [execute, hidden, indices, weights, output,
             experts = experts_, k = num_experts_per_tok_,
             owner = runtime.shared_from_this()]() mutable {
                execute(*owner, hidden, indices, weights, output, experts, k);
            }, false));
    } else {
        execute(runtime, hidden_states, top_k_index, top_k_weights,
                final_hidden_states, experts_, num_experts_per_tok_);
    }
    return final_hidden_states;
}

infinilm::runtime::Tensor DeepseekV2Experts::forward(infinilm::runtime::Runtime &runtime,
                                              const infinilm::runtime::Tensor &hidden_states,
                                              const infinilm::runtime::Tensor &top_k_index,
                                              const infinilm::runtime::Tensor &top_k_weights) const {
    runtime.assertOwnerThread();
    ASSERT(hidden_states->ndim() == 2);
    if (hidden_states->device() != runtime.device() ||
        top_k_index->device() != runtime.device() || top_k_weights->device() != runtime.device()) {
        throw std::invalid_argument("Deepseek MoE inputs must belong to the supplied Runtime device");
    }
    if (supports_fused_deepseek_moe(hidden_states->device().type())
        && (hidden_states->dtype() == infinilm::runtime::DataType::kBFloat16
            || hidden_states->dtype() == infinilm::runtime::DataType::kFloat16)) {
        // Fallback is selected before entering this branch. Once fused
        // execution is attempted, propagate compute and collective errors.
        auto output = infinilm::runtime::Tensor::empty(runtime, hidden_states->shape(), hidden_states->dtype());
        infinilm::execution::submit<infini::ops::DeepseekMoe>(
            runtime, infinilm::execution::RecordingPolicy::AllowHostReplay,
            infini::ops::DeepseekMoe::DefaultConfig(runtime.device().type()),
            output, hidden_states, top_k_index, top_k_weights,
                                                       gate_weights_, up_weights_, down_weights_,
                                                       local_moe_intermediate_size_, num_experts_);
        INFINILM_TRACE_OP(DeepseekMoe, output);
            if (tp_size_ > 1 && communicator_ != nullptr) {
            infinilm::engine::distributed::allreduce_(runtime, output, output, infinicclSum, communicator_);
            }
            return output;
        }
    return forward_cpu_routed_(runtime, hidden_states, top_k_index, top_k_weights);
}

DeepseekV2MoE::DeepseekV2MoE(infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("DeepseekV2MoE construction requires eager Runtime");
    if (!model_config || rank_info.device != runtime.device() || rank_info.tp_size <= 0
        || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size
        || (rank_info.tp_size > 1 && !rank_info.comm))
        throw std::invalid_argument("Invalid explicit Deepseek construction resources");
    try {
        gate_ = this->bind_child("gate", std::make_shared<std::remove_reference_t<decltype(*gate_ )>>(runtime, model_config));
        experts_ = this->bind_child("experts", std::make_shared<std::remove_reference_t<decltype(*experts_ )>>(runtime, model_config, rank_info));

    const size_t n_shared_experts = model_config->get_or<size_t>("n_shared_experts", 0);
    has_shared_experts_ = n_shared_experts > 0;
    if (has_shared_experts_) {
        auto shared_config_json = model_config->get_config_json();
        shared_config_json["intermediate_size"] = model_config->get<size_t>("moe_intermediate_size") * n_shared_experts;
        auto shared_config = std::make_shared<infinilm::config::ModelConfig>(shared_config_json);
            shared_experts_ = this->bind_child("shared_experts", std::make_shared<std::remove_reference_t<decltype(*shared_experts_ )>>(runtime, shared_config, layer_idx, rank_info, attention_backend, ep_config));
        }
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor DeepseekV2MoE::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const {
    runtime.assertOwnerThread();
    ASSERT(hidden_states->ndim() == 3);
    const auto shape = hidden_states->shape();
    auto hidden_states_reshaped = hidden_states->view({shape[0] * shape[1], shape[2]});

    auto [routing_weights, selected_experts] = gate_->forward(runtime, hidden_states_reshaped);
    auto final_hidden_states = experts_->forward(runtime, hidden_states_reshaped, selected_experts, routing_weights)->view(shape);
    if (has_shared_experts_) {
        auto shared_out = shared_experts_->forward(runtime, hidden_states);
        {
            auto ops_output = infinilm::runtime::Tensor::empty(runtime, final_hidden_states->shape(), final_hidden_states->dtype());
            infinilm::execution::submit<infini::ops::Add>(
                runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), final_hidden_states, shared_out, ops_output);
            INFINILM_TRACE_OP(Add, ops_output);
            final_hidden_states = std::move(ops_output);
        }
    }
    return final_hidden_states;
}

} // namespace infinilm::models::deepseek_v2
