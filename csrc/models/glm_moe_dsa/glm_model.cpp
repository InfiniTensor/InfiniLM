#include "glm_model.hpp"
#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"
#include "../models_registry.hpp"
#include "glm_dsa_allocate_cache_tensors.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/cat.hpp"
#include "infinicore/ops/distributed/p2p.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace infinilm::models::glm_moe_dsa {
namespace {
void debug_dump(const infinicore::Tensor &tensor, const std::string &name) {
    if (std::getenv("INFINILM_GLM_DEBUG_DUMP") == nullptr) {
        return;
    }
    const auto &rank = infinilm::global_state::get_tensor_model_parallel_rank_info();
    if (rank.global_rank != 0) {
        return;
    }
    tensor->debug("/tmp/glmdbg_" + name + ".bin");
}

infinicore::Tensor i32_tensor_on_device(
    const std::vector<int32_t> &values,
    const infinicore::Device &device) {
    auto tensor = infinicore::Tensor::empty(
        {values.size()}, infinicore::DataType::I32, device);
    infinicore::context::memcpyH2D(
        tensor->data(), values.data(), values.size() * sizeof(int32_t), false);
    return tensor;
}
} // namespace

GlmDecoder::GlmDecoder(std::shared_ptr<infinilm::config::ModelConfig> c, size_t i, const infinicore::Device &d) {
    layer_idx_ = i;
    auto h = c->get<size_t>("hidden_size");
    auto e = c->get<double>("rms_norm_eps");
    auto dt = c->get_dtype();
    INFINICORE_NN_MODULE_INIT(input_layernorm, h, e, dt, d);
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, h, e, dt, d);
    INFINICORE_NN_MODULE_INIT(self_attn, c, i, d);
    moe_ = i >= c->get_or<size_t>("first_k_dense_replace", 0);
    if (moe_) {
        moe_mlp_ = register_module<GlmMoE>("mlp", c, d);
    } else {
        dense_mlp_ = register_module<GlmDenseMLP>("mlp", c, d);
    }
}
void GlmDecoder::forward(const infinicore::Tensor &p, infinicore::Tensor &x, infinicore::Tensor &r) const {
    input_layernorm_->forward_inplace(x, r);
    debug_dump(x, "layer_" + std::to_string(layer_idx_) + "_input_norm");
    x = self_attn_->forward(p, x);
    debug_dump(x, "layer_" + std::to_string(layer_idx_) + "_attn");
    post_attention_layernorm_->forward_inplace(x, r);
    debug_dump(x, "layer_" + std::to_string(layer_idx_) + "_post_attn_norm");
    x = moe_ ? moe_mlp_->forward(x) : dense_mlp_->forward(x);
    debug_dump(x, "layer_" + std::to_string(layer_idx_) + "_mlp");
}

namespace {
std::vector<size_t> make_pipeline_boundaries(
    const std::shared_ptr<infinilm::config::ModelConfig> &config,
    int pp_size) {
    const size_t num_layers = config->get<size_t>("num_hidden_layers");
    if (pp_size < 1 || static_cast<size_t>(pp_size) > num_layers) {
        throw std::runtime_error(
            "GLM pipeline parallel size must be within the model layer count");
    }
    const auto indexer_types = config->get_config_json()
                                   .at("indexer_types")
                                   .get<std::vector<std::string>>();
    std::vector<size_t> boundaries(static_cast<size_t>(pp_size) + 1, 0);
    boundaries.back() = num_layers;
    for (int stage = 1; stage < pp_size; ++stage) {
        const size_t ideal = num_layers * static_cast<size_t>(stage)
                           / static_cast<size_t>(pp_size);
        const size_t min_layer = boundaries[static_cast<size_t>(stage) - 1] + 1;
        const size_t max_layer = num_layers
                               - static_cast<size_t>(pp_size - stage);
        size_t best = ideal;
        size_t best_distance = std::numeric_limits<size_t>::max();
        for (size_t layer = min_layer; layer <= max_layer; ++layer) {
            const bool fresh_indexer = layer >= indexer_types.size()
                                    || indexer_types[layer] != "shared";
            const size_t distance = layer > ideal ? layer - ideal : ideal - layer;
            if (fresh_indexer && distance < best_distance) {
                best = layer;
                best_distance = distance;
            }
        }
        if (best_distance == std::numeric_limits<size_t>::max()) {
            best = std::max(min_layer, std::min(ideal, max_layer));
        }
        boundaries[static_cast<size_t>(stage)] = best;
    }
    return boundaries;
}
} // namespace

GlmModel::GlmModel(std::shared_ptr<infinilm::config::ModelConfig> c, const infinicore::Device &d) {
    dtype_ = c->get_dtype();
    hidden_size_ = c->get<size_t>("hidden_size");
    const size_t num_layers = c->get<size_t>("num_hidden_layers");
    const auto &rank = infinilm::global_state::get_tensor_model_parallel_rank_info();
    pp_size_ = rank.pp_size;
    pp_rank_ = rank.pp_rank;
    pp_comm_ = rank.pp_comm;
    const auto boundaries = make_pipeline_boundaries(c, pp_size_);
    layer_start_ = boundaries[static_cast<size_t>(pp_rank_)];
    layer_end_ = boundaries[static_cast<size_t>(pp_rank_) + 1];
    const auto indexer_types = c->get_config_json()
                                   .at("indexer_types")
                                   .get<std::vector<std::string>>();
    stage_boundary_needs_topk_.reserve(
        static_cast<size_t>(std::max(0, pp_size_ - 1)));
    for (int stage = 0; stage + 1 < pp_size_; ++stage) {
        const size_t next_layer = boundaries[static_cast<size_t>(stage) + 1];
        stage_boundary_needs_topk_.push_back(
            next_layer < indexer_types.size()
            && indexer_types[next_layer] == "shared");
    }

    if (rank.is_pipeline_first_stage()) {
        INFINICORE_NN_MODULE_INIT(
            embed_tokens, c->get<size_t>("vocab_size"), hidden_size_, dtype_, d);
    }
    for (size_t i = layer_start_; i < layer_end_; ++i) {
        layers_.push_back(register_module<GlmDecoder>("layers." + std::to_string(i), c, i, d));
    }
    if (rank.is_pipeline_last_stage()) {
        INFINICORE_NN_MODULE_INIT(
            norm, hidden_size_, c->get<double>("rms_norm_eps"), dtype_, d);
    }
    index_topk_ = c->get<size_t>("index_topk");
}

void GlmModel::begin_pipeline_batch() const {
    pipeline_send_lifetimes_.clear();
}

void GlmModel::transfer_pipeline_state_(
    size_t source_stage,
    infinicore::Tensor &hidden_states,
    infinicore::Tensor &residual,
    bool transfer_indexer_state) const {
    if (pp_size_ == 1) {
        return;
    }
    if (pp_comm_ == nullptr) {
        throw std::runtime_error("GLM pipeline stage requires a PP communicator");
    }
    if (!hidden_states || hidden_states->ndim() != 3
        || hidden_states->size(0) != 1
        || hidden_states->size(2) != hidden_size_) {
        throw std::runtime_error("GLM pipeline hidden state has an invalid shape");
    }

    const bool is_source = static_cast<size_t>(pp_rank_) == source_stage;
    const bool is_destination = static_cast<size_t>(pp_rank_) == source_stage + 1;
    if (!is_source && !is_destination) {
        return;
    }
    if (is_source) {
        if (!residual || residual->shape() != hidden_states->shape()) {
            throw std::runtime_error("GLM pipeline residual state is unavailable");
        }
    } else {
        hidden_states = infinicore::Tensor::empty(
            {1, hidden_states->size(1), hidden_size_}, dtype_,
            hidden_states->device());
        residual = infinicore::Tensor::empty(
            hidden_states->shape(), dtype_, hidden_states->device());
    }

    std::vector<infinicore::Tensor> pipeline_state{hidden_states, residual};
    if (transfer_indexer_state) {
        auto &forward_context = infinilm::global_state::get_forward_context();
        if (!forward_context.dsa_topk_indices.has_value()) {
            throw std::runtime_error("GLM pipeline indexer state is unavailable");
        }
        auto &topk = forward_context.dsa_topk_indices.value();
        pipeline_state.push_back(topk);
    }

    if (is_source) {
        infinicore::op::distributed::send_grouped(
            pipeline_state, static_cast<int>(source_stage + 1), pp_comm_);
        pipeline_send_lifetimes_.insert(
            pipeline_send_lifetimes_.end(),
            pipeline_state.begin(), pipeline_state.end());
    } else {
        infinicore::op::distributed::recv_grouped_(
            pipeline_state, static_cast<int>(source_stage), pp_comm_);
        if (!infinicore::context::isGraphRecording()) {
            infinicore::context::syncStream();
        }
    }
}

infinicore::Tensor GlmModel::forward(const infinilm::InfinilmModel::Input &i) const {
    if (!i.input_ids.has_value() || !i.position_ids.has_value()) {
        throw std::runtime_error("GLM pipeline requires input and position IDs");
    }
    const size_t num_tokens = i.position_ids.value()->numel();
    infinicore::Tensor x;
    if (pp_rank_ == 0) {
        x = embed_tokens_->forward(i.input_ids.value());
        debug_dump(x, "embed");
    } else {
        x = infinicore::Tensor::empty(
            {1, num_tokens, hidden_size_}, dtype_, i.position_ids.value()->device());
        set_zeros_device_async(x);
    }
    auto &forward_context = infinilm::global_state::get_forward_context();
    forward_context.dsa_topk_indices = infinicore::Tensor::empty(
        {num_tokens, index_topk_},
        infinicore::DataType::I32,
        x->device());
    set_minus_one_device_async(forward_context.dsa_topk_indices.value());
    infinicore::Tensor r;
    for (int stage = 0; stage < pp_size_; ++stage) {
        if (stage == pp_rank_) {
            for (auto &l : layers_) {
                l->forward(i.position_ids.value(), x, r);
            }
        }
        if (stage + 1 < pp_size_) {
            transfer_pipeline_state_(
                static_cast<size_t>(stage), x, r,
                stage_boundary_needs_topk_[static_cast<size_t>(stage)]);
        }
    }
    if (pp_rank_ == pp_size_ - 1) {
        norm_->forward_inplace(x, r);
        debug_dump(x, "final");
    }
    return x;
}
GlmForCausalLM::GlmForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> c, const infinicore::Device &d) {
    model_config_ = c;
    INFINICORE_NN_MODULE_INIT(model, c, d);
    const auto &rank = infinilm::global_state::get_tensor_model_parallel_rank_info();
    is_output_stage_ = rank.is_pipeline_last_stage();
    if (is_output_stage_) {
        INFINICORE_NN_MODULE_INIT(lm_head, c->get<size_t>("hidden_size"), c->get<size_t>("vocab_size"), false, c->get_dtype(), d);
    }
}
infinilm::InfinilmModel::Output GlmForCausalLM::forward(const Input &i) const {
    auto select_logits_input = [](infinicore::Tensor x, const Input &input) {
        if (!input.sample_all_positions && input.input_offsets.has_value()) {
            const auto &input_offsets = input.input_offsets.value();
            if (input_offsets->numel() < 2) {
                throw std::runtime_error(
                    "GLM logits selection requires at least one request");
            }
            const size_t num_requests = input_offsets->numel() - 1;
            const size_t num_tokens = x->size(0) * x->size(1);
            if (num_tokens != num_requests) {
                if (num_requests == 1) {
                    x = x->narrow({{1, x->size(1) - 1, 1}});
                } else {
                    auto selected = infinicore::Tensor::empty(
                        {1, num_requests, x->size(2)}, x->dtype(), x->device());
                    infinicore::op::select_last_token_hidden_(
                        selected, x, input_offsets);
                    x = selected;
                }
            }
        }
        return x;
    };

    const auto &rank = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const size_t num_requests = i.total_sequence_lengths.has_value()
                                  ? i.total_sequence_lengths.value()->numel()
                                  : 1;
    const size_t num_tokens = i.position_ids.has_value()
                                ? i.position_ids.value()->numel()
                                : 0;
    const bool decode_batch = num_tokens == num_requests;
    model_->begin_pipeline_batch();
    if (rank.pp_size == 1 || num_requests <= 1 || decode_batch) {
        auto x = model_->forward(i);
        if (!is_output_stage_) {
            return {x};
        }
        return {lm_head_->forward(select_logits_input(x, i))};
    }
    if (!i.input_ids.has_value() || !i.position_ids.has_value()
        || !i.input_offsets.has_value() || !i.request_ids.has_value()
        || !i.cu_seqlens.has_value() || !i.block_tables.has_value()
        || !i.slot_mapping.has_value()) {
        throw std::runtime_error(
            "GLM PP microbatching requires complete paged-attention metadata");
    }

    std::vector<int32_t> input_offsets(num_requests + 1);
    std::vector<int32_t> cu_seqlens(num_requests + 1);
    if (decode_batch) {
        for (size_t request = 0; request <= num_requests; ++request) {
            input_offsets[request] = static_cast<int32_t>(request);
            cu_seqlens[request] = static_cast<int32_t>(request);
        }
    } else {
        infinicore::context::syncStream();
        infinicore::context::memcpyD2H(
            input_offsets.data(), i.input_offsets.value()->data(),
            input_offsets.size() * sizeof(int32_t));
        infinicore::context::memcpyD2H(
            cu_seqlens.data(), i.cu_seqlens.value()->data(),
            cu_seqlens.size() * sizeof(int32_t));
    }

    auto &forward_context = infinilm::global_state::get_forward_context();
    const auto full_attn_metadata = forward_context.attn_metadata;
    infinicore::Tensor output_hidden;
    infinicore::Tensor local_output;
    constexpr size_t long_prefill_threshold = 512;
    constexpr size_t long_prefill_microbatch_tokens = 2048;
    const bool group_requests = num_tokens >= long_prefill_threshold;
    size_t request_begin = 0;
    while (request_begin < num_requests) {
        size_t request_end = request_begin;
        while (request_end < num_requests) {
            const size_t candidate_tokens = static_cast<size_t>(
                input_offsets[request_end + 1] - input_offsets[request_begin]);
            if (request_end > request_begin
                && (!group_requests
                    || candidate_tokens > long_prefill_microbatch_tokens)) {
                break;
            }
            ++request_end;
            if (group_requests
                && candidate_tokens >= long_prefill_microbatch_tokens) {
                break;
            }
        }

        const size_t micro_requests = request_end - request_begin;
        const size_t token_start = static_cast<size_t>(input_offsets[request_begin]);
        const size_t token_end = static_cast<size_t>(input_offsets[request_end]);
        if (token_end <= token_start || token_end > num_tokens) {
            throw std::runtime_error("GLM PP microbatch has invalid token offsets");
        }
        const size_t token_count = token_end - token_start;
        std::vector<int32_t> micro_input_offsets(micro_requests + 1);
        std::vector<int32_t> micro_cu_seqlens(micro_requests + 1);
        std::vector<int32_t> micro_request_ids(token_count);
        for (size_t local_request = 0; local_request < micro_requests;
             ++local_request) {
            const size_t global_request = request_begin + local_request;
            micro_input_offsets[local_request] = input_offsets[global_request] - input_offsets[request_begin];
            micro_cu_seqlens[local_request] = cu_seqlens[global_request] - cu_seqlens[request_begin];
            const size_t local_token_begin = static_cast<size_t>(
                input_offsets[global_request] - input_offsets[request_begin]);
            const size_t local_token_end = static_cast<size_t>(
                input_offsets[global_request + 1] - input_offsets[request_begin]);
            std::fill(
                micro_request_ids.begin() + local_token_begin,
                micro_request_ids.begin() + local_token_end,
                static_cast<int32_t>(local_request));
        }
        micro_input_offsets[micro_requests] = static_cast<int32_t>(token_count);
        micro_cu_seqlens[micro_requests] = cu_seqlens[request_end] - cu_seqlens[request_begin];
        Input micro = i;
        micro.input_ids = i.input_ids.value()->narrow({{1, token_start, token_count}});
        micro.position_ids = i.position_ids.value()->ndim() == 1
                               ? i.position_ids.value()->narrow({{0, token_start, token_count}})
                               : i.position_ids.value()->narrow({{1, token_start, token_count}});
        if (i.past_sequence_lengths.has_value()) {
            micro.past_sequence_lengths = i.past_sequence_lengths.value()->narrow(
                {{0, request_begin, micro_requests}});
        }
        micro.total_sequence_lengths = i.total_sequence_lengths.value()->narrow(
            {{0, request_begin, micro_requests}});
        micro.block_tables = i.block_tables.value()->narrow(
            {{0, request_begin, micro_requests}});
        micro.slot_mapping = i.slot_mapping.value()->narrow({{0, token_start, token_count}});
        micro.input_offsets = i32_tensor_on_device(
            micro_input_offsets, micro.position_ids.value()->device());
        micro.request_ids = i32_tensor_on_device(
            micro_request_ids, micro.position_ids.value()->device());
        micro.cu_seqlens = i32_tensor_on_device(
            micro_cu_seqlens, micro.position_ids.value()->device());
        if (infinicore::context::isGraphRecording()) {
            graph_microbatch_constants_.push_back(micro.input_offsets.value());
            graph_microbatch_constants_.push_back(micro.request_ids.value());
            graph_microbatch_constants_.push_back(micro.cu_seqlens.value());
        }

        forward_context.attn_metadata = {
            micro.past_sequence_lengths,
            micro.total_sequence_lengths,
            micro.input_offsets,
            micro.request_ids,
            micro.cu_seqlens,
            micro.block_tables,
            micro.slot_mapping,
            full_attn_metadata.max_context_len,
            false};
        auto x = model_->forward(micro);
        local_output = x;
        if (is_output_stage_) {
            auto selected = select_logits_input(x, micro);
            if (!output_hidden) {
                output_hidden = infinicore::Tensor::empty(
                    {1, num_requests, selected->size(2)},
                    selected->dtype(), selected->device());
            }
            output_hidden
                ->narrow({{1, request_begin, micro_requests}})
                ->copy_from(selected);
        }
        request_begin = request_end;
    }
    forward_context.attn_metadata = full_attn_metadata;
    if (!is_output_stage_) {
        return {local_output};
    }
    return {lm_head_->forward(output_hidden)};
}

void GlmForCausalLM::reset_cache(const cache::CacheConfig *cache_config) {
    auto &forward_context = infinilm::global_state::get_forward_context();
    forward_context.dsa_topk_indices.reset();
    forward_context.kv_cache_vec.clear();
    forward_context.mla_vendor_cache_vec.clear();
    forward_context.indexer_cache_vec.clear();
    infinicore::context::syncStream();
    infinicore::context::trimMemory();
    if (cache_config == nullptr) {
        cache_config_.reset();
        return;
    }
    cache_config_ = cache_config->unique_copy();
    auto caches = glm_dsa_allocate_cache_tensors(
        cache_config,
        model_config_,
        global_state::get_infinilm_config().attention_backend,
        model_->layer_start(),
        model_->layer_end());
    forward_context.kv_cache_vec = std::move(caches.mla);
    forward_context.mla_vendor_cache_vec = std::move(caches.mla_vendor);
    forward_context.indexer_cache_vec = std::move(caches.indexer);
    forward_context.dsa_topk_indices.reset();
}
std::shared_ptr<infinilm::config::ModelConfig> create_glm_config(std::shared_ptr<infinilm::config::ModelConfig> c) {
    auto j = c->get_config_json();
    auto qh = j.at("qk_nope_head_dim").get<size_t>() + j.at("qk_rope_head_dim").get<size_t>();
    j["head_dim"] = qh;
    if (!j.contains("rope_theta") && j.contains("rope_parameters")) {
        j["rope_theta"] = j["rope_parameters"].value("rope_theta", 10000.0);
    }
    j["partial_rotary_factor"] = double(j.at("qk_rope_head_dim").get<size_t>()) / double(qh);
    j["num_experts"] = j.at("n_routed_experts");
    j["mlp_bias"] = false;
    j["quantization_config"] = {{"quant_method", "glm_w8a8"}};
    auto n = std::make_shared<infinilm::config::ModelConfig>(j);
    n->set_rope_algo(infinicore::nn::RoPE::Algo::GPT_J);
    return n;
}
} // namespace infinilm::models::glm_moe_dsa
namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(glm_moe_dsa, infinilm::models::glm_moe_dsa::GlmForCausalLM, infinilm::models::glm_moe_dsa::create_glm_config);
}
