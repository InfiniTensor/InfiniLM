#include "ernie4_5_attention.hpp"

#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/rope.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infinilm::models::ernie4_5_vl {
namespace {

std::pair<infinicore::Tensor, infinicore::Tensor> build_group_rope_cache(size_t max_seq_len,
                                                                         size_t rotary_dim,
                                                                         size_t group_pairs,
                                                                         size_t first_pair_idx,
                                                                         size_t pair_stride,
                                                                         double theta,
                                                                         const infinicore::DataType &dtype,
                                                                         const infinicore::Device &device) {
    const size_t numel = max_seq_len * group_pairs;
    std::vector<float> sin_data(numel);
    std::vector<float> cos_data(numel);
    for (size_t pos = 0; pos < max_seq_len; ++pos) {
        for (size_t group_idx = 0; group_idx < group_pairs; ++group_idx) {
            const size_t pair_idx = first_pair_idx + group_idx * pair_stride;
            const float inv_freq = 1.0f / std::pow(static_cast<float>(theta), 2.0f * static_cast<float>(pair_idx) / static_cast<float>(rotary_dim));
            const float angle = static_cast<float>(pos) * inv_freq;
            const size_t offset = pos * group_pairs + group_idx;
            sin_data[offset] = std::sin(angle);
            cos_data[offset] = std::cos(angle);
        }
    }

    const auto cpu = infinicore::Device::cpu();
    auto sin_cache = infinicore::Tensor::empty({max_seq_len, group_pairs}, dtype, device);
    auto cos_cache = infinicore::Tensor::empty({max_seq_len, group_pairs}, dtype, device);
    if (dtype == infinicore::DataType::F32) {
        auto sin_cpu = infinicore::Tensor::from_blob(sin_data.data(), {max_seq_len, group_pairs}, infinicore::DataType::F32, cpu);
        auto cos_cpu = infinicore::Tensor::from_blob(cos_data.data(), {max_seq_len, group_pairs}, infinicore::DataType::F32, cpu);
        sin_cache->copy_from(sin_cpu);
        cos_cache->copy_from(cos_cpu);
        return {sin_cache, cos_cache};
    }
    if (dtype == infinicore::DataType::BF16) {
        std::vector<uint16_t> sin_bf16(numel);
        std::vector<uint16_t> cos_bf16(numel);
        for (size_t i = 0; i < numel; ++i) {
            sin_bf16[i] = f32_to_bf16(sin_data[i]);
            cos_bf16[i] = f32_to_bf16(cos_data[i]);
        }
        auto sin_cpu = infinicore::Tensor::from_blob(sin_bf16.data(), {max_seq_len, group_pairs}, infinicore::DataType::BF16, cpu);
        auto cos_cpu = infinicore::Tensor::from_blob(cos_bf16.data(), {max_seq_len, group_pairs}, infinicore::DataType::BF16, cpu);
        sin_cache->copy_from(sin_cpu);
        cos_cache->copy_from(cos_cpu);
        return {sin_cache, cos_cache};
    }
    if (dtype == infinicore::DataType::F16) {
        std::vector<uint16_t> sin_f16(numel);
        std::vector<uint16_t> cos_f16(numel);
        for (size_t i = 0; i < numel; ++i) {
            sin_f16[i] = f32_to_f16(sin_data[i]);
            cos_f16[i] = f32_to_f16(cos_data[i]);
        }
        auto sin_cpu = infinicore::Tensor::from_blob(sin_f16.data(), {max_seq_len, group_pairs}, infinicore::DataType::F16, cpu);
        auto cos_cpu = infinicore::Tensor::from_blob(cos_f16.data(), {max_seq_len, group_pairs}, infinicore::DataType::F16, cpu);
        sin_cache->copy_from(sin_cpu);
        cos_cache->copy_from(cos_cpu);
        return {sin_cache, cos_cache};
    }
    throw std::runtime_error("infinilm::models::ernie4_5_vl::Ernie45Attention: unsupported RoPE cache dtype");
}

infinicore::Tensor axis_positions_for_rope(const infinicore::Tensor &position_ids, size_t axis, bool has_batch_dim) {
    const auto pos_shape = position_ids->shape();
    if (pos_shape.size() != 3 || pos_shape[2] < 3) {
        throw std::runtime_error("infinilm::models::ernie4_5_vl::Ernie45Attention: ERNIE MRoPE expects [batch, seq, 3] position_ids");
    }
    auto pos = position_ids->narrow({{0, 0, 1}, {2, axis, 1}})->contiguous();
    if (has_batch_dim) {
        return pos->view({pos_shape[0], pos_shape[1]});
    }
    return pos->view({pos_shape[1]});
}

void apply_grouped_rope_one(infinicore::Tensor &x,
                            const infinicore::Tensor &position_ids,
                            size_t axis,
                            size_t group_pairs,
                            size_t first_pair_idx,
                            size_t pair_stride,
                            const infinicore::Tensor &sin_cache,
                            const infinicore::Tensor &cos_cache) {
    const size_t ndim = x->ndim();
    if (ndim != 3 && ndim != 4) {
        throw std::runtime_error("infinilm::models::ernie4_5_vl::Ernie45Attention: ERNIE grouped RoPE expects 3D or 4D q/k");
    }
    const size_t last_dim = ndim - 1;
    auto group_shape = x->shape();
    group_shape[last_dim] = group_pairs * 2;
    auto group = infinicore::Tensor::empty(group_shape, x->dtype(), x->device());

    for (size_t group_idx = 0; group_idx < group_pairs; ++group_idx) {
        const size_t src_pair_idx = first_pair_idx + group_idx * pair_stride;
        group->narrow({{last_dim, 2 * group_idx, 2}})->copy_from(x->narrow({{last_dim, 2 * src_pair_idx, 2}}));
    }

    auto positions = axis_positions_for_rope(position_ids, axis, ndim == 4);
    infinicore::op::rope_(group, group, positions, sin_cache, cos_cache, infinicore::nn::RoPE::Algo::GPT_J);

    for (size_t group_idx = 0; group_idx < group_pairs; ++group_idx) {
        const size_t dst_pair_idx = first_pair_idx + group_idx * pair_stride;
        x->narrow({{last_dim, 2 * dst_pair_idx, 2}})->copy_from(group->narrow({{last_dim, 2 * group_idx, 2}}));
    }
}

void apply_ernie_grouped_mrope(infinicore::Tensor &q,
                               infinicore::Tensor &k,
                               const infinicore::Tensor &position_ids,
                               const std::vector<int> &section,
                               const infinicore::Tensor &sin_h,
                               const infinicore::Tensor &cos_h,
                               const infinicore::Tensor &sin_w,
                               const infinicore::Tensor &cos_w,
                               const infinicore::Tensor &sin_t,
                               const infinicore::Tensor &cos_t) {
    const size_t h_pairs = static_cast<size_t>(section[0]);
    const size_t w_pairs = static_cast<size_t>(section[1]);
    const size_t t_pairs = static_cast<size_t>(section[2]);
    const size_t t_first_pair = h_pairs + w_pairs;

    apply_grouped_rope_one(q, position_ids, 1, h_pairs, 0, 2, sin_h, cos_h);
    apply_grouped_rope_one(q, position_ids, 2, w_pairs, 1, 2, sin_w, cos_w);
    apply_grouped_rope_one(q, position_ids, 0, t_pairs, t_first_pair, 1, sin_t, cos_t);
    apply_grouped_rope_one(k, position_ids, 1, h_pairs, 0, 2, sin_h, cos_h);
    apply_grouped_rope_one(k, position_ids, 2, w_pairs, 1, 2, sin_w, cos_w);
    apply_grouped_rope_one(k, position_ids, 0, t_pairs, t_first_pair, 1, sin_t, cos_t);
}

} // namespace

std::shared_ptr<const Ernie45MropeCache> build_ernie45_mrope_cache(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                                   const infinicore::Device &device) {
    auto cache = std::make_shared<Ernie45MropeCache>();
    const size_t rotary_dim = model_config->get_rotary_dim();
    const double rope_theta = model_config->get<double>("rope_theta");
    const auto &config_json = model_config->get_config_json();
    if (config_json.contains("rope_parameters") && config_json["rope_parameters"].contains("mrope_section")) {
        cache->section = config_json["rope_parameters"]["mrope_section"].get<std::vector<int>>();
    } else if (config_json.contains("rope_scaling") && config_json["rope_scaling"].contains("mrope_section")) {
        cache->section = config_json["rope_scaling"]["mrope_section"].get<std::vector<int>>();
    }
    if (cache->section.size() != 3 || static_cast<size_t>(cache->section[0] + cache->section[1] + cache->section[2]) * 2 != rotary_dim) {
        throw std::runtime_error("infinilm::models::ernie4_5_vl::Ernie45Attention: invalid mrope_section");
    }

    const auto &dtype = model_config->get_dtype();
    const size_t max_position_embeddings = model_config->get<size_t>("max_position_embeddings");
    auto h_cache = build_group_rope_cache(max_position_embeddings, rotary_dim, static_cast<size_t>(cache->section[0]), 0, 2, rope_theta, dtype, device);
    auto w_cache = build_group_rope_cache(max_position_embeddings, rotary_dim, static_cast<size_t>(cache->section[1]), 1, 2, rope_theta, dtype, device);
    auto t_cache = build_group_rope_cache(max_position_embeddings,
                                          rotary_dim,
                                          static_cast<size_t>(cache->section[2]),
                                          static_cast<size_t>(cache->section[0] + cache->section[1]),
                                          1,
                                          rope_theta,
                                          dtype,
                                          device);
    cache->sin_h = h_cache.first;
    cache->cos_h = h_cache.second;
    cache->sin_w = w_cache.first;
    cache->cos_w = w_cache.second;
    cache->sin_t = t_cache.first;
    cache->cos_t = t_cache.second;
    return cache;
}

Ernie45Attention::Ernie45Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                   size_t layer_idx,
                                   std::shared_ptr<const Ernie45MropeCache> mrope_cache,
                                   const infinicore::Device &device)
    : layer_idx_(layer_idx),
      mrope_cache_(std::move(mrope_cache)) {
    hidden_size_ = model_config->get<size_t>("hidden_size");
    head_dim_ = model_config->get_head_dim();
    rotary_dim_ = model_config->get_rotary_dim();
    if (!mrope_cache_) {
        throw std::runtime_error("infinilm::models::ernie4_5_vl::Ernie45Attention: mrope_cache is required");
    }

    const auto &dtype = model_config->get_dtype();
    const size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
    const size_t total_num_kv_heads = model_config->get<size_t>("num_key_value_heads");
    const bool use_bias = model_config->get_or<bool>("use_bias", false);

    attention_backend_ = infinilm::global_state::get_infinilm_config().attention_backend;
    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const int tp_rank = rank_info.tp_rank;
    const int tp_size = rank_info.tp_size;
    if ((total_num_kv_heads < static_cast<size_t>(tp_size)) || (0 != (total_num_kv_heads % static_cast<size_t>(tp_size)))) {
        throw std::runtime_error("infinilm::models::ernie4_5_vl::Ernie45Attention: num_key_value_heads must be divisible by tp_size");
    }

    num_attention_heads_ = total_num_heads / static_cast<size_t>(tp_size);
    num_key_value_heads_ = total_num_kv_heads / static_cast<size_t>(tp_size);

    auto quantization_method = model_config->get_quantization_method();
    INFINICORE_NN_MODULE_INIT(q_proj, hidden_size_, total_num_heads * head_dim_, quantization_method, use_bias, dtype, device, tp_rank, tp_size);
    INFINICORE_NN_MODULE_INIT(k_proj, hidden_size_, total_num_kv_heads * head_dim_, quantization_method, use_bias, dtype, device, tp_rank, tp_size);
    INFINICORE_NN_MODULE_INIT(v_proj, hidden_size_, total_num_kv_heads * head_dim_, quantization_method, use_bias, dtype, device, tp_rank, tp_size);
    INFINICORE_NN_MODULE_INIT(o_proj, total_num_heads * head_dim_, hidden_size_, quantization_method, use_bias, dtype, device, tp_rank, tp_size, rank_info.comm);

    rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(model_config, device);
    const float scaling = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(num_attention_heads_, head_dim_, scaling, num_key_value_heads_, layer_idx_,
                                                                          kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_);

    auto register_fn = [this](const std::string &n, infinicore::nn::Parameter p) { this->register_parameter(n, std::move(p)); };
    infinilm::layers::attention::init_kv_cache_quant_params(register_fn, device, kv_cache_k_scale_, kv_cache_v_scale_);
}

infinicore::Tensor Ernie45Attention::forward(const infinicore::Tensor &positions,
                                             const infinicore::Tensor &hidden_states) const {
    if (infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
        return forward_static_(positions, hidden_states);
    }
    return forward_paged_(positions, hidden_states);
}

infinicore::Tensor Ernie45Attention::forward_static_(const infinicore::Tensor &position_ids,
                                                     const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    const auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];

    auto q = q_proj_->forward(hidden_states_mutable)->view({batch_size, seq_len, num_attention_heads_, head_dim_});
    auto k = k_proj_->forward(hidden_states_mutable)->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    auto v = v_proj_->forward(hidden_states_mutable)->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

    infinicore::Tensor pos_ids_for_rope = position_ids;
    const auto pos_shape = position_ids->shape();
    if (pos_shape.size() == 2) {
        pos_ids_for_rope = position_ids->narrow({{0, 0, 1}})->view({pos_shape[1]});
    } else if (pos_shape.size() == 3) {
        pos_ids_for_rope = position_ids->narrow({{0, 0, 1}, {2, 0, 1}})->view({pos_shape[1]});
    } else if (pos_shape.size() != 1) {
        throw std::runtime_error("infinilm::models::ernie4_5_vl::Ernie45Attention: unsupported position_ids shape");
    }

    auto q_rotary = q->narrow({{3, 0, rotary_dim_}});
    auto k_rotary = k->narrow({{3, 0, rotary_dim_}});
    if (pos_shape.size() == 3) {
        apply_ernie_grouped_mrope(q, k, position_ids, mrope_cache_->section, mrope_cache_->sin_h, mrope_cache_->cos_h, mrope_cache_->sin_w, mrope_cache_->cos_w, mrope_cache_->sin_t, mrope_cache_->cos_t);
    } else {
        rotary_emb_->forward(q_rotary, pos_ids_for_rope, true);
        rotary_emb_->forward(k_rotary, pos_ids_for_rope, true);
    }

    auto attn_output = attn_->forward(q, k, v);
    return o_proj_->forward(attn_output);
}

infinicore::Tensor Ernie45Attention::forward_paged_(const infinicore::Tensor &position_ids,
                                                    const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    const auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];

    ASSERT_EQ(batch_size, 1);

    auto q = q_proj_->forward(hidden_states_mutable)->view({seq_len, num_attention_heads_, head_dim_});
    auto k = k_proj_->forward(hidden_states_mutable)->view({seq_len, num_key_value_heads_, head_dim_});
    auto v = v_proj_->forward(hidden_states_mutable)->view({seq_len, num_key_value_heads_, head_dim_});

    infinicore::Tensor pos_ids_for_rope = position_ids;
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
        apply_ernie_grouped_mrope(q, k, position_ids, mrope_cache_->section, mrope_cache_->sin_h, mrope_cache_->cos_h, mrope_cache_->sin_w, mrope_cache_->cos_w, mrope_cache_->sin_t, mrope_cache_->cos_t);
    } else {
        rotary_emb_->forward(q_rotary, pos_ids_for_rope, true);
        rotary_emb_->forward(k_rotary, pos_ids_for_rope, true);
    }

    auto attn_output = attn_->forward(q, k, v);
    return o_proj_->forward(attn_output);
}

} // namespace infinilm::models::ernie4_5_vl
