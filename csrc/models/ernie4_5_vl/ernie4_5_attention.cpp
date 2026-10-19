#include "ernie4_5_attention.hpp"
#include "../../weights/weights.hpp"
#include "../models_registry.hpp"
#include "runtime/runtime.hpp"
#include <exception>

#include "../../execution/submit.hpp"
#include "../../global_state/forward_context.hpp"
#include "../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace infinilm::models::ernie4_5_vl {
namespace {

void validate_grouped_mrope_section(const std::vector<int> &section, size_t rotary_dim) {
    if (section.size() != 3 || rotary_dim % 2 != 0) {
        throw std::invalid_argument("Ernie MRoPE requires three sections covering rotary pairs");
    }
    std::vector<bool> assigned(rotary_dim / 2, false);
    for (size_t axis = 0; axis < section.size(); ++axis) {
        if (section[axis] < 0 || static_cast<size_t>(section[axis]) > assigned.size()) {
            throw std::invalid_argument("Ernie MRoPE section is outside the rotary dimension");
        }
        const size_t first = axis < 2 ? axis : static_cast<size_t>(section[0]) + section[1];
        const size_t stride = axis < 2 ? 2 : 1;
        for (size_t i = 0; i < static_cast<size_t>(section[axis]); ++i) {
            const size_t pair = first + i * stride;
            if (pair >= assigned.size() || assigned[pair]) {
                throw std::invalid_argument("Ernie grouped MRoPE sections overlap or exceed the rotary dimension");
            }
            assigned[pair] = true;
        }
    }
    if (std::find(assigned.begin(), assigned.end(), false) != assigned.end()) {
        throw std::invalid_argument("Ernie grouped MRoPE sections leave unassigned rotary pairs");
    }
}

// Source vectors remain alive through completion or fail-stop in this local upload.
void upload_custom_rope_cache(infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor table, const infinilm::runtime::Tensor &host) {
    try {
        table->copy_from(runtime, host);
        runtime.syncStream(); // cache creation only; never forward
    } catch (...) {
        const auto failure = std::current_exception();
        if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) {
            try {
                runtime.syncStream();
            } catch (...) { std::terminate(); }
        }
        std::rethrow_exception(failure);
    }
}

infinilm::runtime::Tensor build_group_rope_cache(infinilm::runtime::Runtime &runtime, size_t max_seq_len,
                                                 size_t rotary_dim,
                                                 size_t group_pairs,
                                                 size_t first_pair_idx,
                                                 size_t pair_stride,
                                                 double theta,
                                                 const infinilm::runtime::DataType &dtype,
                                                 const infinilm::runtime::Device &device) {
    runtime.assertOwnerThread();
    if (runtime.device() != device) {
        throw std::invalid_argument("Ernie grouped RoPE cache Runtime and device must agree");
    }
    const size_t cache_dim = group_pairs * 2;
    const size_t numel = max_seq_len * cache_dim;
    std::vector<float> cache_data(numel);
    for (size_t pos = 0; pos < max_seq_len; ++pos) {
        for (size_t group_idx = 0; group_idx < group_pairs; ++group_idx) {
            const size_t pair_idx = first_pair_idx + group_idx * pair_stride;
            const float inv_freq = 1.0f / std::pow(static_cast<float>(theta), 2.0f * static_cast<float>(pair_idx) / static_cast<float>(rotary_dim));
            const float angle = static_cast<float>(pos) * inv_freq;
            const size_t offset = pos * cache_dim + group_idx;
            cache_data[offset] = std::cos(angle);
            cache_data[offset + group_pairs] = std::sin(angle);
        }
    }

    const auto cpu = infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu};
    auto cache = infinilm::runtime::Tensor::empty(runtime, {max_seq_len, cache_dim}, dtype);
    if (dtype == infinilm::runtime::DataType::kFloat32) {
        auto cache_cpu = infinilm::runtime::Tensor::from_blob(cache_data.data(), {max_seq_len, cache_dim}, infinilm::runtime::DataType::kFloat32, cpu);
        upload_custom_rope_cache(runtime, cache, cache_cpu);
        return cache;
    }
    if (dtype == infinilm::runtime::DataType::kBFloat16) {
        std::vector<uint16_t> cache_bf16(numel);
        for (size_t i = 0; i < numel; ++i) {
            cache_bf16[i] = f32_to_bf16(cache_data[i]);
        }
        auto cache_cpu = infinilm::runtime::Tensor::from_blob(cache_bf16.data(), {max_seq_len, cache_dim}, infinilm::runtime::DataType::kBFloat16, cpu);
        upload_custom_rope_cache(runtime, cache, cache_cpu);
        return cache;
    }
    if (dtype == infinilm::runtime::DataType::kFloat16) {
        std::vector<uint16_t> cache_f16(numel);
        for (size_t i = 0; i < numel; ++i) {
            cache_f16[i] = f32_to_f16(cache_data[i]);
        }
        auto cache_cpu = infinilm::runtime::Tensor::from_blob(cache_f16.data(), {max_seq_len, cache_dim}, infinilm::runtime::DataType::kFloat16, cpu);
        upload_custom_rope_cache(runtime, cache, cache_cpu);
        return cache;
    }
    throw std::runtime_error("infinilm::models::ernie4_5_vl::Ernie45Attention: unsupported RoPE cache dtype");
}

infinilm::runtime::Tensor axis_positions_for_rope(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &position_ids, size_t axis, bool has_batch_dim) {
    const auto pos_shape = position_ids->shape();
    if (pos_shape.size() != 3 || pos_shape[2] < 3) {
        throw std::runtime_error("infinilm::models::ernie4_5_vl::Ernie45Attention: ERNIE MRoPE expects [batch, seq, 3] position_ids");
    }
    if (has_batch_dim) {
        return position_ids->narrow({{2, axis, 1}})
            ->contiguous(runtime)
            ->view({pos_shape[0], pos_shape[1]});
    }
    return position_ids->narrow({{0, 0, 1}, {2, axis, 1}})
        ->contiguous(runtime)
        ->view({pos_shape[1]});
}

void apply_grouped_rope_one(infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor &x,
                            const infinilm::runtime::Tensor &position_ids,
                            size_t axis,
                            size_t group_pairs,
                            size_t first_pair_idx,
                            size_t pair_stride,
                            const infinilm::runtime::Tensor &cos_sin_cache) {
    const size_t ndim = x->ndim();
    if (ndim != 3 && ndim != 4) {
        throw std::runtime_error("infinilm::models::ernie4_5_vl::Ernie45Attention: ERNIE grouped RoPE expects 3D or 4D q/k");
    }
    const size_t last_dim = ndim - 1;
    auto group_shape = x->shape();
    group_shape[last_dim] = group_pairs * 2;
    auto group = infinilm::runtime::Tensor::empty(runtime, group_shape, x->dtype());

    for (size_t group_idx = 0; group_idx < group_pairs; ++group_idx) {
        const size_t src_pair_idx = first_pair_idx + group_idx * pair_stride;
        group->narrow({{last_dim, 2 * group_idx, 2}})->copy_from(runtime, x->narrow({{last_dim, 2 * src_pair_idx, 2}}));
    }

    auto positions = axis_positions_for_rope(runtime, position_ids, axis, ndim == 4);
    execution::submit<infini::ops::RotaryEmbedding>(
        runtime, execution::RecordingPolicy::AllowHostReplay,
        infini::ops::RotaryEmbedding::DefaultConfig(runtime.device().type()),
        positions, group, std::optional<infinilm::runtime::Tensor>{}, cos_sin_cache,
        static_cast<int64_t>(group_pairs * 2), false, int64_t{0}, false);
    INFINILM_TRACE_OP(RotaryEmbedding, group);

    for (size_t group_idx = 0; group_idx < group_pairs; ++group_idx) {
        const size_t dst_pair_idx = first_pair_idx + group_idx * pair_stride;
        x->narrow({{last_dim, 2 * dst_pair_idx, 2}})->copy_from(runtime, group->narrow({{last_dim, 2 * group_idx, 2}}));
    }
}

void apply_ernie_grouped_mrope(infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor &q,
                               infinilm::runtime::Tensor &k,
                               const infinilm::runtime::Tensor &position_ids,
                               const std::vector<int> &section,
                               const infinilm::runtime::Tensor &cos_sin_h,
                               const infinilm::runtime::Tensor &cos_sin_w,
                               const infinilm::runtime::Tensor &cos_sin_t) {
    const size_t h_pairs = static_cast<size_t>(section[0]);
    const size_t w_pairs = static_cast<size_t>(section[1]);
    const size_t t_pairs = static_cast<size_t>(section[2]);
    const size_t t_first_pair = h_pairs + w_pairs;

    apply_grouped_rope_one(runtime, q, position_ids, 1, h_pairs, 0, 2, cos_sin_h);
    apply_grouped_rope_one(runtime, q, position_ids, 2, w_pairs, 1, 2, cos_sin_w);
    apply_grouped_rope_one(runtime, q, position_ids, 0, t_pairs, t_first_pair, 1, cos_sin_t);
    apply_grouped_rope_one(runtime, k, position_ids, 1, h_pairs, 0, 2, cos_sin_h);
    apply_grouped_rope_one(runtime, k, position_ids, 2, w_pairs, 1, 2, cos_sin_w);
    apply_grouped_rope_one(runtime, k, position_ids, 0, t_pairs, t_first_pair, 1, cos_sin_t);
}

} // namespace

void Ernie45MropeCache::require_owner(infinilm::runtime::Runtime &runtime) const {
    runtime.assertOwnerThread();
    if (runtime_owner.get() != &runtime) {
        throw std::invalid_argument("Ernie grouped RoPE cache belongs to another Runtime owner");
    }
}

std::shared_ptr<const Ernie45MropeCache> build_ernie45_mrope_cache(infinilm::runtime::Runtime &runtime, std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                                   const infinilm::runtime::Device &device) {
    runtime.assertOwnerThread();
    if (runtime.device() != device) {
        throw std::invalid_argument("Ernie grouped RoPE cache Runtime and device must agree");
    }
    auto cache = std::make_shared<Ernie45MropeCache>();
    cache->runtime_owner = runtime.shared_from_this();
    const size_t rotary_dim = model_config->get_rotary_dim();
    const double rope_theta = model_config->get<double>("rope_theta");
    const auto &config_json = model_config->get_config_json();
    if (config_json.contains("rope_parameters") && config_json["rope_parameters"].contains("mrope_section")) {
        cache->section = config_json["rope_parameters"]["mrope_section"].get<std::vector<int>>();
    } else if (config_json.contains("rope_scaling") && config_json["rope_scaling"].contains("mrope_section")) {
        cache->section = config_json["rope_scaling"]["mrope_section"].get<std::vector<int>>();
    }
    validate_grouped_mrope_section(cache->section, rotary_dim);

    const auto &dtype = model_config->get_dtype();
    const size_t max_position_embeddings = model_config->get<size_t>("max_position_embeddings");
    auto h_cache = build_group_rope_cache(runtime, max_position_embeddings, rotary_dim, static_cast<size_t>(cache->section[0]), 0, 2, rope_theta, dtype, device);
    auto w_cache = build_group_rope_cache(runtime, max_position_embeddings, rotary_dim, static_cast<size_t>(cache->section[1]), 1, 2, rope_theta, dtype, device);
    auto t_cache = build_group_rope_cache(runtime, max_position_embeddings,
                                          rotary_dim,
                                          static_cast<size_t>(cache->section[2]),
                                          static_cast<size_t>(cache->section[0] + cache->section[1]),
                                          1,
                                          rope_theta,
                                          dtype,
                                          device);
    cache->cos_sin_h = h_cache;
    cache->cos_sin_w = w_cache;
    cache->cos_sin_t = t_cache;
    return cache;
}

Ernie45Attention::Ernie45Attention(infinilm::runtime::Runtime &runtime,
                                   std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                   size_t layer_idx,
                                   std::shared_ptr<const Ernie45MropeCache> mrope_cache,
                                   const engine::distributed::RankInfo &rank_info,
                                   backends::AttentionBackend attention_backend)
    : layer_idx_(layer_idx),
      mrope_cache_(std::move(mrope_cache)) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) {
        throw std::logic_error("Ernie45 construction requires eager Runtime");
    }
    if (!model_config) {
        throw std::invalid_argument("Ernie45 construction requires configuration");
    }
    if (rank_info.device != runtime.device() || rank_info.tp_size <= 0 || rank_info.tp_rank < 0
        || rank_info.tp_rank >= rank_info.tp_size || (rank_info.tp_size > 1 && !rank_info.comm)) {
        throw std::invalid_argument("Invalid explicit Ernie45 construction resources");
    }
    try {
        hidden_size_ = model_config->get<size_t>("hidden_size");
        head_dim_ = model_config->get_head_dim();
        rotary_dim_ = model_config->get_rotary_dim();
        if (!mrope_cache_) {
            throw std::runtime_error("infinilm::models::ernie4_5_vl::Ernie45Attention: mrope_cache is required");
        }

        mrope_cache_->require_owner(runtime);
        const auto &dtype = model_config->get_dtype();
        const size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
        const size_t total_num_kv_heads = model_config->get<size_t>("num_key_value_heads");
        const bool use_bias = model_config->get_or<bool>("use_bias", false);

        attention_backend_ = attention_backend;
        const int tp_rank = rank_info.tp_rank;
        const int tp_size = rank_info.tp_size;
        if (head_dim_ == 0 || total_num_heads == 0) {
            throw std::invalid_argument("Ernie45Attention requires positive head geometry");
        }
        if ((total_num_kv_heads < static_cast<size_t>(tp_size)) || (0 != (total_num_kv_heads % static_cast<size_t>(tp_size)))) {
            throw std::runtime_error("infinilm::models::ernie4_5_vl::Ernie45Attention: num_key_value_heads must be divisible by tp_size");
        }

        num_attention_heads_ = total_num_heads / static_cast<size_t>(tp_size);
        num_key_value_heads_ = total_num_kv_heads / static_cast<size_t>(tp_size);

        auto quantization_method = model_config->get_quantization_method();
        q_proj_ = std::make_shared<std::remove_reference_t<decltype(*q_proj_)>>(runtime, hidden_size_, total_num_heads * head_dim_, quantization_method, use_bias, dtype, tp_rank, tp_size);
        this->bind_child("q_proj", q_proj_);
        k_proj_ = std::make_shared<std::remove_reference_t<decltype(*k_proj_)>>(runtime, hidden_size_, total_num_kv_heads * head_dim_, quantization_method, use_bias, dtype, tp_rank, tp_size);
        this->bind_child("k_proj", k_proj_);
        v_proj_ = std::make_shared<std::remove_reference_t<decltype(*v_proj_)>>(runtime, hidden_size_, total_num_kv_heads * head_dim_, quantization_method, use_bias, dtype, tp_rank, tp_size);
        this->bind_child("v_proj", v_proj_);
        o_proj_ = std::make_shared<std::remove_reference_t<decltype(*o_proj_)>>(runtime, total_num_heads * head_dim_, hidden_size_, quantization_method, use_bias, dtype, tp_rank, tp_size, rank_info.comm);
        this->bind_child("o_proj", o_proj_);

        rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(runtime, model_config, runtime.device());
        auto register_fn = [this](const std::string &n, infinilm::weights::Weight p) { this->bind_weight(n, std::move(p)); };
        infinilm::layers::attention::init_kv_cache_quant_params(runtime, register_fn, kv_cache_k_scale_, kv_cache_v_scale_,
                                                                model_config->get_kv_quant_scheme());

        const float scaling = 1.0f / std::sqrt(static_cast<float>(head_dim_));
        attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(num_attention_heads_, head_dim_, scaling, num_key_value_heads_, layer_idx_,
                                                                              kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_,
                                                                              model_config->get_kv_quant_scheme());
    } catch (...) {
        try {
            runtime.syncStream();
        } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor Ernie45Attention::forward(const infinilm::runtime::Tensor &positions,
                                                    const infinilm::runtime::Tensor &hidden_states,
                                                    global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    mrope_cache_->require_owner(runtime);
    if (infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
        return forward_static_(positions, hidden_states, forward_context);
    }
    return forward_paged_(positions, hidden_states, forward_context);
}

infinilm::runtime::Tensor Ernie45Attention::forward_static_(const infinilm::runtime::Tensor &position_ids,
                                                            const infinilm::runtime::Tensor &hidden_states,
                                                            global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    auto hidden_states_mutable = hidden_states;
    const auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];

    auto q = q_proj_->forward(runtime, hidden_states_mutable)->view({batch_size, seq_len, num_attention_heads_, head_dim_});
    auto k = k_proj_->forward(runtime, hidden_states_mutable)->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    auto v = v_proj_->forward(runtime, hidden_states_mutable)->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

    infinilm::runtime::Tensor pos_ids_for_rope = position_ids;
    const auto pos_shape = position_ids->shape();
    if (pos_shape.size() == 2) {
        ASSERT_EQ(pos_shape[0], batch_size);
        ASSERT_EQ(pos_shape[1], seq_len);
        pos_ids_for_rope = position_ids->contiguous(runtime);
    } else if (pos_shape.size() == 3) {
        pos_ids_for_rope = position_ids->narrow({{0, 0, 1}, {2, 0, 1}})->view({pos_shape[1]});
    } else if (pos_shape.size() == 1) {
        ASSERT_EQ(batch_size, 1);
        ASSERT_EQ(pos_shape[0], seq_len);
        pos_ids_for_rope = position_ids->contiguous(runtime)->view({1, seq_len});
    } else {
        throw std::runtime_error("infinilm::models::ernie4_5_vl::Ernie45Attention: unsupported position_ids shape");
    }

    auto q_rotary = q->narrow({{3, 0, rotary_dim_}});
    auto k_rotary = k->narrow({{3, 0, rotary_dim_}});
    if (pos_shape.size() == 3) {
        apply_ernie_grouped_mrope(runtime, q, k, position_ids, mrope_cache_->section, mrope_cache_->cos_sin_h, mrope_cache_->cos_sin_w, mrope_cache_->cos_sin_t);
    } else {
        rotary_emb_->forward(runtime, q_rotary, pos_ids_for_rope, true);
        rotary_emb_->forward(runtime, k_rotary, pos_ids_for_rope, true);
    }

    auto attn_output = attn_->forward(q, k, v, forward_context);
    return o_proj_->forward(runtime, attn_output);
}

infinilm::runtime::Tensor Ernie45Attention::forward_paged_(const infinilm::runtime::Tensor &position_ids,
                                                           const infinilm::runtime::Tensor &hidden_states,
                                                           global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    auto hidden_states_mutable = hidden_states;
    const auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];

    ASSERT_EQ(batch_size, 1);

    auto q = q_proj_->forward(runtime, hidden_states_mutable)->view({seq_len, num_attention_heads_, head_dim_});
    auto k = k_proj_->forward(runtime, hidden_states_mutable)->view({seq_len, num_key_value_heads_, head_dim_});
    auto v = v_proj_->forward(runtime, hidden_states_mutable)->view({seq_len, num_key_value_heads_, head_dim_});

    infinilm::runtime::Tensor pos_ids_for_rope = position_ids;
    const auto pos_shape = position_ids->shape();
    if (pos_shape.size() == 2) {
        pos_ids_for_rope = position_ids->narrow({{0, 0, 1}})->view({pos_shape[1]});
    } else if (pos_shape.size() == 3) {
        pos_ids_for_rope = position_ids->narrow({{0, 0, 1}, {2, 0, 1}})->view({pos_shape[1]});
    } else if (pos_shape.size() != 1) {
        throw std::runtime_error("infinilm::models::ernie4_5_vl::Ernie45Attention: unsupported position_ids shape");
    }

    auto q_rotary = q->narrow({{2, 0, rotary_dim_}});
    auto k_rotary = k->narrow({{2, 0, rotary_dim_}});
    if (pos_shape.size() == 3) {
        apply_ernie_grouped_mrope(runtime, q, k, position_ids, mrope_cache_->section, mrope_cache_->cos_sin_h, mrope_cache_->cos_sin_w, mrope_cache_->cos_sin_t);
    } else {
        rotary_emb_->forward(runtime, q_rotary, pos_ids_for_rope, true);
        rotary_emb_->forward(runtime, k_rotary, pos_ids_for_rope, true);
    }

    auto attn_output = attn_->forward(q, k, v, forward_context);
    return o_proj_->forward(runtime, attn_output);
}

} // namespace infinilm::models::ernie4_5_vl
