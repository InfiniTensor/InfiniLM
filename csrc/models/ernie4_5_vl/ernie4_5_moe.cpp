#include "../../weights/weights.hpp"
#include "../../layers/mlp/fused_activation.hpp"
#include "../../execution/submit.hpp"
#include <base/gemm.h>
#include <base/cast.h>
#include <base/topk_softmax.h>
#include "ernie4_5_moe.hpp"

#include "../../global_state/forward_context.hpp"
#include "../models_registry.hpp"
#include <exception>
#include "runtime/runtime.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::ernie4_5_vl {
namespace {

std::vector<size_t> get_size_list(const nlohmann::json &config, const char *key) {
    std::vector<size_t> values;
    if (!config.contains(key) || config.at(key).is_null()) {
        return values;
    }
    const auto &value = config.at(key);
    if (value.is_array()) {
        values.reserve(value.size());
        for (const auto &item : value) {
            values.push_back(item.get<size_t>());
        }
    } else {
        values.push_back(value.get<size_t>());
    }
    return values;
}

} // namespace

Ernie45TopKRouter::Ernie45TopKRouter(infinilm::runtime::Runtime &runtime,
        std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("Ernie45 construction requires eager Runtime");
    if (!model_config) throw std::invalid_argument("Ernie45 construction requires configuration");
    try {
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t num_experts = model_config->get<size_t>("num_experts");
    const auto expert_counts = get_size_list(model_config->get_config_json(), "moe_num_experts");
    const size_t num_router_groups = expert_counts.empty() ? 1 : expert_counts.size();
    num_experts_per_tok_ = model_config->get<size_t>("num_experts_per_tok");
    norm_topk_prob_ = model_config->get<bool>("norm_topk_prob");

    ASSERT((num_experts > 0) && (num_router_groups <= 2) && (num_experts_per_tok_ > 0) && (num_experts_per_tok_ <= num_experts * num_router_groups));

        // ERNIE stores router weights as [hidden_size, group_num_experts], while Qwen3 stores
        // [num_experts, hidden_size]. matmul(hidden, weight) matches ERNIE directly.
        const auto dtype = infinilm::runtime::DataType::kFloat32;
        weight_ = infinilm::weights::Weight(runtime, {hidden_size, num_experts}, dtype); this->bind_weight("weight", weight_);
    if (num_router_groups > 1) {
            weight_1_ = infinilm::weights::Weight(runtime, {hidden_size, num_experts}, dtype); this->bind_weight("weight_1", weight_1_);
        }
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor> Ernie45TopKRouter::forward(infinilm::runtime::Runtime &runtime,
                                                                              const infinilm::runtime::Tensor &hidden_states,
                                                                              const infinilm::runtime::Tensor &correction_bias,
                                                                              size_t group_idx) const {
    ASSERT(hidden_states->ndim() == 2);
    const auto router_weight = group_idx == 1 && weight_1_ ? static_cast<infinilm::runtime::Tensor>(weight_1_) : static_cast<infinilm::runtime::Tensor>(weight_);
    runtime.assertOwnerThread();
    if (!hidden_states || hidden_states->device() != runtime.device())
        throw std::invalid_argument("Cast input and Runtime devices must match");
    auto router_input = hidden_states;
    if (router_input->dtype() != infinilm::runtime::DataType::kFloat32) {
        auto router_output = infinilm::runtime::Tensor::empty(runtime, hidden_states->shape(), infinilm::runtime::DataType::kFloat32);
        infinilm::execution::submit<infini::ops::Cast>(
            runtime, infinilm::execution::RecordingPolicy::AllowHostReplay,
            infini::ops::Cast::DefaultConfig(runtime.device().type()), hidden_states, router_output);
        router_input = std::move(router_output);
    }
    auto router_logits_shape = router_input->shape();
    router_logits_shape[router_input->ndim() - 1] = router_weight->size(router_input->ndim() - 1);
    auto router_logits = infinilm::runtime::Tensor::empty(runtime, router_logits_shape, router_input->dtype());
    infinilm::execution::submit<infini::ops::Gemm>(
        runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
        infini::ops::Gemm::DefaultConfig(runtime.device().type()), router_input, router_weight,
        std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{1.0F},
        std::optional<float>{0.0F}, std::optional<int>{}, std::optional<int>{}, router_logits);
    INFINILM_TRACE_OP(Gemm, router_logits);
    if (!router_logits || router_logits->ndim() != 2 ||
        num_experts_per_tok_ == 0 || num_experts_per_tok_ > router_logits->size(1) ||
        runtime.device() != router_logits->device()) {
        throw std::invalid_argument("moe_topk_softmax input or topk is invalid");
    }
    auto router_scores = infinilm::runtime::Tensor::empty(runtime, {router_logits->size(0), num_experts_per_tok_}, infinilm::runtime::DataType::kFloat32);
    auto router_indices = infinilm::runtime::Tensor::empty(runtime, router_scores->shape(), infinilm::runtime::DataType::kInt32);
    auto topk_scratch = infinilm::runtime::Tensor::empty(runtime, router_scores->shape(), infinilm::runtime::DataType::kInt32);
    infini::ops::TopkSoftmaxConfig topk_config{0.0f, 1e-12f};
    topk_config.set_implementation_index(
        infini::ops::TopkSoftmax::DefaultConfig(runtime.device().type()).implementation_index());
    const auto topk_bias = correction_bias
        ? std::optional<infinilm::runtime::Tensor>{correction_bias}
        : std::optional<infinilm::runtime::Tensor>{};
    infinilm::execution::submit<infini::ops::TopkSoftmax>(
        runtime, infinilm::execution::RecordingPolicy::AllowHostReplay, topk_config,
        router_logits, topk_bias, std::optional<infinilm::runtime::Tensor>{},
        norm_topk_prob_, router_scores, router_indices, topk_scratch);
    INFINILM_TRACE_OP(MoeTopkSoftmax, router_scores);
    return {router_scores, router_indices};
}

Ernie45ExpertMLP::Ernie45ExpertMLP(infinilm::runtime::Runtime &runtime,
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                   size_t intermediate_size,
        const engine::distributed::RankInfo &rank_info) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("Ernie45 construction requires eager Runtime");
    if (!model_config) throw std::invalid_argument("Ernie45 construction requires configuration");
    if (rank_info.device != runtime.device() || rank_info.tp_size <= 0 || rank_info.tp_rank < 0
        || rank_info.tp_rank >= rank_info.tp_size || (rank_info.tp_size > 1 && !rank_info.comm))
        throw std::invalid_argument("Invalid explicit Ernie45 construction resources");
    try {
    const auto &dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const bool use_bias = model_config->get_or<bool>("mlp_bias", false);
    const int tp_rank = rank_info.tp_rank;
    const int tp_size = rank_info.tp_size;
    auto quantization_method = model_config->get_quantization_method();

        gate_proj_ = std::make_shared<std::remove_reference_t<decltype(*gate_proj_)>>(runtime, hidden_size, intermediate_size, quantization_method, use_bias, dtype, tp_rank, tp_size);
        this->bind_child("gate_proj", gate_proj_);
        up_proj_ = std::make_shared<std::remove_reference_t<decltype(*up_proj_)>>(runtime, hidden_size, intermediate_size, quantization_method, use_bias, dtype, tp_rank, tp_size);
        this->bind_child("up_proj", up_proj_);
        down_proj_ = std::make_shared<std::remove_reference_t<decltype(*down_proj_)>>(runtime, intermediate_size, hidden_size, quantization_method, use_bias, dtype, tp_rank, tp_size, rank_info.comm);
        this->bind_child("down_proj", down_proj_);
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor Ernie45ExpertMLP::forward(infinilm::runtime::Runtime &runtime,
                                            const infinilm::runtime::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    auto gate = gate_proj_->forward(runtime, hidden_states_mutable);
    auto up = up_proj_->forward(runtime, hidden_states_mutable);
    auto intermediate = infinilm::runtime::Tensor::empty(runtime, up->shape(), up->dtype());
    infinilm::layers::mlp::branch_activation_into(runtime, intermediate, up, gate);
    return down_proj_->forward(runtime, intermediate);
}

Ernie45Experts::Ernie45Experts(infinilm::runtime::Runtime &runtime,
        std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("Ernie45 construction requires eager Runtime");
    if (!model_config) throw std::invalid_argument("Ernie45 construction requires configuration");
    use_bias_ = model_config->get_or<bool>("mlp_bias", false);
    if (model_config->get_quant_scheme() != infinilm::quantization::QuantScheme::NONE)
        throw std::invalid_argument("Ernie45Experts: quantization must be NONE");
    const auto &dtype = model_config->get_dtype();

    const auto &config_json = model_config->get_config_json();
    auto expert_counts = get_size_list(config_json, "moe_num_experts");
    auto intermediate_sizes = get_size_list(config_json, "moe_intermediate_size");
    if (expert_counts.empty()) expert_counts.push_back(model_config->get<size_t>("num_experts"));
    if (intermediate_sizes.empty()) intermediate_sizes.push_back(model_config->get<size_t>("moe_intermediate_size"));
    if (intermediate_sizes.size() == 1 && expert_counts.size() > 1)
        intermediate_sizes.resize(expert_counts.size(), intermediate_sizes.front());
    if (expert_counts.size() != intermediate_sizes.size() || expert_counts.size() > 2)
        throw std::invalid_argument("Ernie45Experts: expected one or two matching expert groups");
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    num_experts_per_tok_ = model_config->get<size_t>("num_experts_per_tok");
    const size_t int_limit = static_cast<size_t>(std::numeric_limits<int32_t>::max());
    for (size_t i = 0; i < expert_counts.size(); ++i) {
        const size_t e = expert_counts[i], m = intermediate_sizes[i];
        if (!hidden_size || hidden_size > int_limit
            || !m || m > int_limit / 2 || !e
            || !num_experts_per_tok_ || num_experts_per_tok_ > e
            || hidden_size > std::numeric_limits<size_t>::max() / (2 * m)
            || e > std::numeric_limits<size_t>::max() / (2 * m * hidden_size) / 2)
            throw std::invalid_argument("Ernie45Experts: unsupported dense runner geometry");
    }
    num_text_experts_ = expert_counts.front();
    num_vision_experts_ = expert_counts.size() > 1 ? expert_counts[1] : 0;
    const size_t text_intermediate_size = intermediate_sizes.front();
    const size_t vision_intermediate_size = intermediate_sizes.size() > 1 ? intermediate_sizes[1] : text_intermediate_size;
    intermediate_sizes_[0] = text_intermediate_size;
    intermediate_sizes_[1] = vision_intermediate_size;
    runtime_owner_ = runtime.shared_from_this();
    try {
        w1_ = infinilm::weights::Weight(runtime, {num_text_experts_, 2 * text_intermediate_size, hidden_size}, dtype); this->bind_weight("w1", w1_);
        w2_ = infinilm::weights::Weight(runtime, {num_text_experts_, hidden_size, text_intermediate_size}, dtype); this->bind_weight("w2", w2_);
    if (use_bias_) {
            b1_ = infinilm::weights::Weight(runtime, {num_text_experts_, 2 * text_intermediate_size}, dtype); this->bind_weight("b1", b1_);
            b2_ = infinilm::weights::Weight(runtime, {num_text_experts_, hidden_size}, dtype); this->bind_weight("b2", b2_);
    }
    if (num_vision_experts_ > 0) {
            w1_1_ = infinilm::weights::Weight(runtime, {num_vision_experts_, 2 * vision_intermediate_size, hidden_size}, dtype); this->bind_weight("w1_1", w1_1_);
            w2_1_ = infinilm::weights::Weight(runtime, {num_vision_experts_, hidden_size, vision_intermediate_size}, dtype); this->bind_weight("w2_1", w2_1_);
        if (use_bias_) {
                b1_1_ = infinilm::weights::Weight(runtime, {num_vision_experts_, 2 * vision_intermediate_size}, dtype); this->bind_weight("b1_1", b1_1_);
                b2_1_ = infinilm::weights::Weight(runtime, {num_vision_experts_, hidden_size}, dtype); this->bind_weight("b2_1", b2_1_);
        }
        }
        for (size_t i = 0; i < expert_counts.size(); ++i)
            runners_[i] = std::make_unique<layers::moe::CudaFusedMoeRunner>(
                expert_counts[i], hidden_size, intermediate_sizes[i], 16);
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor Ernie45Experts::forward(infinilm::runtime::Runtime &runtime,
                                           const infinilm::runtime::Tensor &hidden_states,
                                           const infinilm::runtime::Tensor &top_k_index,
                                           const infinilm::runtime::Tensor &top_k_weights,
                                           size_t group_idx) const {
    runtime.assertOwnerThread();
    if (&runtime != runtime_owner_.get())
        throw std::invalid_argument("Ernie45Experts: forward Runtime differs from construction Runtime");
    if (group_idx >= runners_.size() || !runners_[group_idx])
        throw std::invalid_argument("Ernie45Experts: requested expert group does not exist");
    const auto w1 = group_idx == 1 ? static_cast<infinilm::runtime::Tensor>(w1_1_) : static_cast<infinilm::runtime::Tensor>(w1_);
    const auto w2 = group_idx == 1 ? static_cast<infinilm::runtime::Tensor>(w2_1_) : static_cast<infinilm::runtime::Tensor>(w2_);
    if (!hidden_states || !top_k_index || !top_k_weights || !w1 || !w2
        || hidden_states->device() != runtime.device() || top_k_index->device() != runtime.device()
        || top_k_weights->device() != runtime.device() || w1->device() != runtime.device() || w2->device() != runtime.device()
        || hidden_states->ndim() != 2 || !hidden_states->size(0) || hidden_states->size(1) != w1->size(2)
        || hidden_states->dtype() != w1->dtype() || w2->dtype() != w1->dtype()
        || !hidden_states->is_contiguous() || !w1->is_contiguous() || !w2->is_contiguous()
        || top_k_index->shape() != infinilm::runtime::Shape{hidden_states->size(0), num_experts_per_tok_}
        || top_k_weights->shape() != top_k_index->shape()
        || top_k_index->dtype() != infinilm::runtime::DataType::kInt32 || top_k_weights->dtype() != infinilm::runtime::DataType::kFloat32
        || !top_k_index->is_contiguous() || !top_k_weights->is_contiguous()
        || reinterpret_cast<uintptr_t>(w1->data()) % 16 || reinterpret_cast<uintptr_t>(w2->data()) % 16
        || (hidden_states->size(0) == 1 && reinterpret_cast<uintptr_t>(hidden_states->data()) % 16))
        throw std::invalid_argument("Ernie45Experts: dense runner inputs have unsupported shape, dtype, layout or device");
    const size_t int_limit = static_cast<size_t>(std::numeric_limits<int32_t>::max());
    const size_t n = hidden_states->size(0), h = hidden_states->size(1), m = intermediate_sizes_[group_idx];
    if (n > int_limit / num_experts_per_tok_)
        throw std::invalid_argument("Ernie45Experts: dense runner token capacity exceeds int32 range");
    const size_t pairs = n * num_experts_per_tok_;
    const size_t align_experts = w1->size(0) + 1;
    const size_t padded = pairs < align_experts ? pairs * 16 : pairs + align_experts * 15;
    const size_t sorted = (padded + 3) / 4 * 4;
    if (sorted > int_limit || sorted > int_limit / h || sorted > int_limit / (2 * m))
        throw std::invalid_argument("Ernie45Experts: dense runner token capacity exceeds int32 range");

    auto &workspace = workspaces_[group_idx];
    // Fresh output per call preserves retained results, including across both groups.
    // Recorded operators retain their Tensor/Runtime leases independently of this slot.
    workspace.fused_moe_output = infinilm::runtime::Tensor::empty(runtime, hidden_states->shape(), hidden_states->dtype());
    const layers::moe::DispatchOutput input{
        layers::moe::DispatchOutputFormat::Standard, hidden_states, {},
        layers::moe::TopKOutput{top_k_weights, top_k_index, {}}, {}};
    const auto bias1 = use_bias_ ? (group_idx == 1 ? static_cast<infinilm::runtime::Tensor>(b1_1_) : static_cast<infinilm::runtime::Tensor>(b1_)) : infinilm::runtime::Tensor();
    const auto bias2 = use_bias_ ? (group_idx == 1 ? static_cast<infinilm::runtime::Tensor>(b2_1_) : static_cast<infinilm::runtime::Tensor>(b2_)) : infinilm::runtime::Tensor();
    const layers::moe::MoeWeights weights{w1, w2, bias1, bias2};
    // The existing runner drains/cancels on failure while inputs and workspace live.
    return runners_[group_idx]->run(runtime, input, weights, workspace).hidden_states;
}

Ernie45MoE::Ernie45MoE(infinilm::runtime::Runtime &runtime,
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        size_t layer_idx,
        const engine::distributed::RankInfo &rank_info,
        backends::AttentionBackend attention_backend,
        const layers::moe::EPConfig &ep_config) {
    models::validate_runtime_construction(model_config, runtime, rank_info);
    try {
    num_experts_ = model_config->get<size_t>("num_experts");
    const auto expert_counts = get_size_list(model_config->get_config_json(), "moe_num_experts");
    num_vision_experts_ = expert_counts.size() > 1 ? expert_counts[1] : 0;
        gate_ = std::make_shared<std::remove_reference_t<decltype(*gate_)>>(runtime, model_config);
        this->bind_child("gate", gate_);
        experts_ = std::make_shared<std::remove_reference_t<decltype(*experts_)>>(runtime, model_config);
        this->bind_child("experts", experts_);

    if (model_config->get_or<bool>("moe_use_aux_free", false)) {
        size_t num_expert_groups = 1;
        const auto &config_json = model_config->get_config_json();
        if (config_json.contains("moe_num_experts") && config_json["moe_num_experts"].is_array()) {
            num_expert_groups = config_json["moe_num_experts"].size();
        }
        const size_t num_experts = model_config->get<size_t>("num_experts");
            e_score_correction_bias_ = infinilm::weights::Weight(runtime, {num_expert_groups, num_experts}, infinilm::runtime::DataType::kFloat32);
            this->bind_weight("moe_statics.e_score_correction_bias", e_score_correction_bias_);
    }

    const size_t n_shared_experts = model_config->get_or<size_t>("moe_num_shared_experts", 0);
    has_shared_experts_ = n_shared_experts > 0;
    if (has_shared_experts_) {
        auto shared_config_json = model_config->get_config_json();
        auto intermediate_sizes = get_size_list(shared_config_json, "moe_intermediate_size");
        if (intermediate_sizes.empty()) {
            throw std::runtime_error("Ernie45MoE: moe_intermediate_size is required for shared experts");
        }
        shared_config_json["intermediate_size"] = intermediate_sizes.front() * n_shared_experts;
        auto shared_config = std::make_shared<infinilm::config::ModelConfig>(shared_config_json);
            shared_experts_ = std::make_shared<std::remove_reference_t<decltype(*shared_experts_)>>(runtime, shared_config, layer_idx, rank_info, attention_backend, ep_config);
            this->bind_child("shared_experts", shared_experts_);
        }
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor Ernie45MoE::forward_group(infinilm::runtime::Runtime &runtime,
                                            const infinilm::runtime::Tensor &hidden_states_2d, size_t group_idx) const {
    const size_t bias_row = group_idx == 1 ? 1 : 0;
    auto correction_bias = e_score_correction_bias_ ? e_score_correction_bias_->narrow({{0, bias_row, 1}})->view({num_experts_}) : infinilm::runtime::Tensor();
    auto [routing_weights, selected_experts] = gate_->forward(runtime, hidden_states_2d, correction_bias, group_idx);
    const auto routing_dtype = hidden_states_2d->dtype();
    runtime.assertOwnerThread();
    if (!routing_weights || routing_weights->device() != runtime.device())
        throw std::invalid_argument("Cast input and Runtime devices must match");
    if (routing_weights->dtype() != routing_dtype) {
        auto converted_weights = infinilm::runtime::Tensor::empty(runtime, routing_weights->shape(), routing_dtype);
        infinilm::execution::submit<infini::ops::Cast>(
            runtime, infinilm::execution::RecordingPolicy::AllowHostReplay,
            infini::ops::Cast::DefaultConfig(runtime.device().type()), routing_weights, converted_weights);
        routing_weights = std::move(converted_weights);
    }
    if (routing_weights->dtype() != infinilm::runtime::DataType::kFloat32) {
        auto converted_weights = infinilm::runtime::Tensor::empty(runtime, routing_weights->shape(), infinilm::runtime::DataType::kFloat32);
        infinilm::execution::submit<infini::ops::Cast>(
            runtime, infinilm::execution::RecordingPolicy::AllowHostReplay,
            infini::ops::Cast::DefaultConfig(runtime.device().type()), routing_weights, converted_weights);
        routing_weights = std::move(converted_weights);
    }
    return experts_->forward(runtime, hidden_states_2d, selected_experts, routing_weights, group_idx);
}

infinilm::runtime::Tensor Ernie45MoE::forward(const infinilm::runtime::Tensor &hidden_states,
                                      global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    ASSERT(hidden_states->ndim() == 3);
    const auto shape = hidden_states->shape();
    auto hidden_states_reshaped = hidden_states->view({shape[0] * shape[1], shape[2]});

    infinilm::runtime::Tensor final_hidden_states_2d;
    const auto &mm_metadata = forward_context.mm_metadata;
    const bool has_visual_ranges = mm_metadata.visual_token_ranges.has_value()
                                && !mm_metadata.visual_token_ranges->empty()
                                && num_vision_experts_ > 0;
    if (!has_visual_ranges) {
        final_hidden_states_2d = forward_group(runtime, hidden_states_reshaped, 0);
    } else {
        final_hidden_states_2d = infinilm::runtime::Tensor::empty(runtime, hidden_states_reshaped->shape(), hidden_states_reshaped->dtype());
        const auto &ranges = mm_metadata.visual_token_ranges.value();
        const size_t total_tokens = hidden_states_reshaped->size(0);
        size_t cursor = 0;
        for (size_t i = 0; i + 1 < ranges.size(); i += 2) {
            const size_t start = std::min(ranges[i], total_tokens);
            const size_t end = std::min(ranges[i + 1], total_tokens);
            if (cursor < start) {
                const size_t len = start - cursor;
                final_hidden_states_2d->narrow({{0, cursor, len}})->copy_from(runtime, forward_group(runtime, hidden_states_reshaped->narrow({{0, cursor, len}}), 0));
            }
            if (start < end) {
                const size_t len = end - start;
                final_hidden_states_2d->narrow({{0, start, len}})->copy_from(runtime, forward_group(runtime, hidden_states_reshaped->narrow({{0, start, len}}), 1));
            }
            cursor = std::max(cursor, end);
        }
        if (cursor < total_tokens) {
            const size_t len = total_tokens - cursor;
            final_hidden_states_2d->narrow({{0, cursor, len}})->copy_from(runtime, forward_group(runtime, hidden_states_reshaped->narrow({{0, cursor, len}}), 0));
        }
    }

    auto final_hidden_states = final_hidden_states_2d->view(shape);
    if (has_shared_experts_) {
        {
            const auto ops_other = shared_experts_->forward(runtime, hidden_states);
            auto ops_output = infinilm::runtime::Tensor::empty(runtime, final_hidden_states->shape(), final_hidden_states->dtype());
            infinilm::execution::submit<infini::ops::Add>(
                runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), final_hidden_states, ops_other, ops_output);
            INFINILM_TRACE_OP(Add, ops_output);
            final_hidden_states = std::move(ops_output);
        }
    }
    return final_hidden_states;
}

} // namespace infinilm::models::ernie4_5_vl
