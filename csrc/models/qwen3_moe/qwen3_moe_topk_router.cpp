#include "qwen3_moe_topk_router.hpp"
#include "../../execution/submit.hpp"
#include "../../layers/linear/dense.hpp"
#include "../../weights/weights.hpp"
#include "../models_registry.hpp"
#include <base/topk_softmax.h>
#include <exception>

#include "../../layers/moe/ep/ep_config.hpp"
#include "runtime/runtime.hpp"
#include <exception>
#include <stdexcept>

namespace infinilm::models::qwen3_moe {

Qwen3MoeTopKRouter::Qwen3MoeTopKRouter(infinilm::runtime::Runtime &runtime, std::shared_ptr<config::ModelConfig> model_config,
                                       const engine::distributed::RankInfo &rank_info, const layers::moe::EPConfig &ep_config) {
    models::validate_runtime_construction(model_config, runtime, rank_info);
    try {

        runtime_borrowed_ = &runtime; // caller/worker retains the actual owner.
        const auto &dtype{model_config->get_dtype()};

        size_t hidden_size = model_config->get<size_t>("hidden_size");
        size_t num_experts = model_config->get<size_t>("num_experts");
        num_experts_per_tok_ = model_config->get<size_t>("num_experts_per_tok");
        norm_topk_prob_ = model_config->get<bool>("norm_topk_prob");

        ASSERT((num_experts > 0) && (num_experts_per_tok_ > 0) && (num_experts_per_tok_ <= num_experts));

        weight_ = infinilm::weights::Weight(runtime, {num_experts, hidden_size}, dtype);
        this->bind_weight("weight", weight_);
        if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) {
            runtime.syncStream();
        }
    } catch (...) {
        if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) {
            try {
                runtime.syncStream();
            } catch (...) { std::terminate(); }
        }
        throw;
    }
}

void Qwen3MoeTopKRouter::validate_runtime(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const {
    runtime.assertOwnerThread();
    if (&runtime != runtime_borrowed_) {
        throw std::invalid_argument("Legacy MoE router Runtime differs from construction owner");
    }
    if (!hidden_states || hidden_states->device() != runtime.device() || weight_->device() != runtime.device()) {
        throw std::invalid_argument("Legacy MoE router input/weight device mismatch");
    }
    if (runtime.isGraphRecording()) {
        throw std::logic_error("Legacy MoE requires eager execution");
    }
}
std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor> Qwen3MoeTopKRouter::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const {
    validate_runtime(runtime, hidden_states);
    ASSERT(hidden_states->ndim() == 2);
    const size_t ntoken = hidden_states->shape()[0];
    infinilm::runtime::Tensor logits, scores, indices, scratch;
    try {
        logits = infinilm::runtime::Tensor::empty(runtime, {ntoken, weight_->size(0)}, hidden_states->dtype());
        infinilm::layers::linear::dense_projection(runtime, hidden_states, weight_.tensor(), std::nullopt, 1.0f, std::nullopt, false, logits);
        scores = infinilm::runtime::Tensor::empty(runtime, {ntoken, num_experts_per_tok_}, infinilm::runtime::DataType::kFloat32);
        indices = infinilm::runtime::Tensor::empty(runtime, {ntoken, num_experts_per_tok_}, infinilm::runtime::DataType::kInt32);
        scratch = infinilm::runtime::Tensor::empty(runtime, scores->shape(), infinilm::runtime::DataType::kInt32);
        infini::ops::TopkSoftmaxConfig topk_config{0.0f, 0.0f};
        topk_config.set_implementation_index(
            infini::ops::TopkSoftmax::DefaultConfig(runtime.device().type()).implementation_index());
        infinilm::execution::submit<infini::ops::TopkSoftmax>(
            runtime, infinilm::execution::RecordingPolicy::AllowHostReplay, topk_config,
            logits, std::optional<infinilm::runtime::Tensor>{},
            std::optional<infinilm::runtime::Tensor>{}, norm_topk_prob_, scores, indices, scratch);
        INFINILM_TRACE_OP(MoeTopkSoftmax, scores);
        runtime.syncStream();
        return std::make_tuple(scores, indices);
    } catch (...) {
        if (runtime.isGraphRecording()) {
            runtime.cancelGraphRecording();
            if (runtime.isGraphRecording()) {
                std::terminate();
            }
        }
        try {
            runtime.syncStream();
        } catch (...) { std::terminate(); }
        throw;
    }
}

} // namespace infinilm::models::qwen3_moe
