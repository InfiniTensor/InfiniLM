#include "ernie4_5_attention.hpp"

#include "../../global_state/global_state.hpp"
#include "../../layers/attention/attention.hpp"
#include "../../utils.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace infinilm::models::ernie4_5_moe_vl {
namespace {

uint16_t fp32_to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1U;
    bits += 0x7FFFU + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

float bf16_to_fp32(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

float read_float(const void *src, size_t index, infinicore::DataType dtype) {
    if (dtype == infinicore::DataType::F32) {
        return reinterpret_cast<const float *>(src)[index];
    }
    if (dtype == infinicore::DataType::BF16) {
        return bf16_to_fp32(reinterpret_cast<const uint16_t *>(src)[index]);
    }
    throw std::runtime_error("Ernie4_5Attention: only float32 and bfloat16 3D RoPE are supported");
}

void write_float(void *dst, size_t index, infinicore::DataType dtype, float value) {
    if (dtype == infinicore::DataType::F32) {
        reinterpret_cast<float *>(dst)[index] = value;
        return;
    }
    if (dtype == infinicore::DataType::BF16) {
        reinterpret_cast<uint16_t *>(dst)[index] = fp32_to_bf16(value);
        return;
    }
    throw std::runtime_error("Ernie4_5Attention: only float32 and bfloat16 3D RoPE are supported");
}

} // namespace

Ernie4_5Attention::Ernie4_5Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                     size_t layer_idx,
                                     const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    hidden_size_ = model_config->get<size_t>("hidden_size");
    head_dim_ = model_config->get<size_t>("head_dim");
    use_rope_3d_ = model_config->get_or<bool>("rope_3d", false);
    freq_allocation_ = model_config->get_or<size_t>("freq_allocation", 20);
    rope_theta_ = model_config->get_or<double>("rope_theta", 10000.0);
    compression_ratio_ = model_config->get_or<double>("compression_ratio", 1.0);

    const auto &dtype{model_config->get_dtype()};
    size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
    size_t total_num_kv_heads = model_config->get<size_t>("num_key_value_heads");
    bool use_bias = model_config->get_or<bool>("attention_bias", false);
    bool use_output_bias = model_config->get_or<bool>("attention_output_bias", false);

    attention_backend_ = infinilm::global_state::get_infinilm_config().attention_backend;
    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
    int tp_size = infinilm::global_state::get_tensor_model_parallel_world_size();

    if ((total_num_kv_heads < static_cast<size_t>(tp_size)) || (total_num_kv_heads % static_cast<size_t>(tp_size) != 0)) {
        throw std::runtime_error("Ernie4_5Attention: num_key_value_heads must be divisible by tp_size");
    }

    num_attention_heads_ = total_num_heads / tp_size;
    num_key_value_heads_ = total_num_kv_heads / tp_size;

    auto quantization_method = model_config->get_quantization_method();
    auto register_fn = [this](const std::string &n, infinicore::nn::Parameter p) { this->register_parameter(n, std::move(p)); };
    qkv_proj_ = std::make_shared<layers::linear::QKVParallelLinear>(
        hidden_size_, head_dim_, total_num_heads, total_num_kv_heads,
        "q_proj", "k_proj", "v_proj", register_fn,
        quantization_method, use_bias, dtype, device, rank_info);
    o_proj_ = this->register_module<layers::linear::RowParallelLinear>(
        "o_proj", total_num_heads * head_dim_, hidden_size_, quantization_method,
        use_output_bias, dtype, device, tp_rank, tp_size, rank_info.comm);

    rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(model_config, device);

    float scaling = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(
        num_attention_heads_, head_dim_, scaling, num_key_value_heads_, layer_idx_,
        kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_);

    infinilm::layers::attention::init_kv_cache_quant_params(register_fn, device, kv_cache_k_scale_, kv_cache_v_scale_);
}

infinicore::Tensor Ernie4_5Attention::forward(const infinicore::Tensor &positions,
                                              const infinicore::Tensor &hidden_states) const {
    if (::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
        return forward_static_(positions, hidden_states);
    }
    return forward_paged_(positions, hidden_states);
}

infinicore::Tensor Ernie4_5Attention::position_ids_for_rope_(const infinicore::Tensor &position_ids) const {
    auto pos_shape = position_ids->shape();
    if (pos_shape.size() == 3) {
        auto text_axis = position_ids->narrow({{2, 0, 1}})->contiguous()->view({pos_shape[0], pos_shape[1]});
        return text_axis->narrow({{0, 0, 1}})->contiguous()->view({pos_shape[1]});
    }
    if (pos_shape.size() == 2) {
        return position_ids->narrow({{0, 0, 1}})->contiguous()->view({pos_shape[1]});
    }
    if (pos_shape.size() == 1) {
        return position_ids->contiguous();
    }
    throw std::runtime_error("Ernie4_5Attention: unexpected position_ids shape");
}

bool Ernie4_5Attention::should_use_rope_3d_(const infinicore::Tensor &position_ids) const {
    return use_rope_3d_ && position_ids->shape().size() == 3;
}

infinicore::Tensor Ernie4_5Attention::apply_rope_3d_(const infinicore::Tensor &states,
                                                     const infinicore::Tensor &position_ids) const {
    const auto state_shape = states->shape();
    if (state_shape.size() != 3 && state_shape.size() != 4) {
        throw std::runtime_error("Ernie4_5Attention: 3D RoPE expects [S,H,D] or [B,S,H,D] states");
    }
    ASSERT(head_dim_ % 2 == 0);
    const size_t half_dim = head_dim_ / 2;
    ASSERT(freq_allocation_ <= half_dim);

    const bool has_batch_dim = state_shape.size() == 4;
    const size_t batch = has_batch_dim ? state_shape[0] : 1;
    const size_t seq_len = has_batch_dim ? state_shape[1] : state_shape[0];
    const size_t num_heads = has_batch_dim ? state_shape[2] : state_shape[1];
    ASSERT_EQ(has_batch_dim ? state_shape[3] : state_shape[2], head_dim_);

    auto pos_cpu = position_ids->to(infinicore::Device::cpu());
    const auto pos_shape = pos_cpu->shape();
    ASSERT(pos_shape.size() == 3);
    ASSERT_EQ(pos_shape[2], 3);
    ASSERT(pos_shape[0] == batch || pos_shape[0] == 1);
    ASSERT(pos_shape[1] >= seq_len);

    auto states_cpu = states->contiguous()->to(infinicore::Device::cpu());
    auto out_shape = has_batch_dim
                       ? std::vector<size_t>{batch, num_heads, seq_len, head_dim_}
                       : states->shape();
    auto out_cpu = infinicore::Tensor::empty(out_shape, states->dtype(), infinicore::Device::cpu());
    const auto *pos = reinterpret_cast<const int64_t *>(pos_cpu->data());
    const void *src = states_cpu->data();
    void *dst = out_cpu->data();

    std::vector<float> inv_freq(half_dim);
    for (size_t j = 0; j < half_dim; ++j) {
        inv_freq[j] = 1.0f / std::pow(static_cast<float>(rope_theta_), static_cast<float>(2 * j) / static_cast<float>(head_dim_));
    }

    const size_t hw_freq_end = half_dim - freq_allocation_;
    for (size_t b = 0; b < batch; ++b) {
        const size_t pos_batch = pos_shape[0] == 1 ? 0 : b;
        for (size_t s = 0; s < seq_len; ++s) {
            const int64_t pos_t = pos[(pos_batch * pos_shape[1] + s) * 3];
            const int64_t pos_h = pos[(pos_batch * pos_shape[1] + s) * 3 + 1];
            const int64_t pos_w = pos[(pos_batch * pos_shape[1] + s) * 3 + 2];
            for (size_t h = 0; h < num_heads; ++h) {
                const size_t src_base = has_batch_dim
                                          ? ((b * seq_len + s) * num_heads + h) * head_dim_
                                          : (s * num_heads + h) * head_dim_;
                const size_t dst_base = has_batch_dim
                                          ? ((b * num_heads + h) * seq_len + s) * head_dim_
                                          : src_base;
                for (size_t j = 0; j < half_dim; ++j) {
                    int64_t position = pos_t;
                    if (j < hw_freq_end) {
                        position = (j % 2 == 0) ? pos_h : pos_w;
                    }
                    const float angle = (static_cast<float>(position) / static_cast<float>(compression_ratio_)) * inv_freq[j];
                    const float sn = std::sin(angle);
                    const float cs = std::cos(angle);
                    const size_t src_even_idx = src_base + 2 * j;
                    const size_t src_odd_idx = src_even_idx + 1;
                    const size_t dst_even_idx = dst_base + 2 * j;
                    const size_t dst_odd_idx = dst_even_idx + 1;
                    const float x0 = read_float(src, src_even_idx, states->dtype());
                    const float x1 = read_float(src, src_odd_idx, states->dtype());
                    write_float(dst, dst_even_idx, states->dtype(), x0 * cs - x1 * sn);
                    write_float(dst, dst_odd_idx, states->dtype(), x1 * cs + x0 * sn);
                }
            }
        }
    }

    auto out = out_cpu->to(states->device());
    if (has_batch_dim) {
        return out->permute({0, 2, 1, 3});
    }
    return out;
}

infinicore::Tensor Ernie4_5Attention::forward_static_(const infinicore::Tensor &position_ids,
                                                      const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    auto [q, k, v] = qkv_proj_->forward_split(hidden_states_mutable);

    auto q_reshaped = q->view({batch_size, seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

    infinicore::Tensor q_rope;
    if (should_use_rope_3d_(position_ids)) {
        q_rope = apply_rope_3d_(q_reshaped, position_ids);
        k_reshaped = apply_rope_3d_(k_reshaped, position_ids);
    } else {
        auto pos_ids_for_rope = position_ids_for_rope_(position_ids);
        q_rope = infinicore::Tensor::empty(
                     {batch_size, num_attention_heads_, seq_len, head_dim_},
                     q_reshaped->dtype(), q_reshaped->device())
                     ->permute({0, 2, 1, 3});
        rotary_emb_->forward(q_rope, q_reshaped, pos_ids_for_rope);
        rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);
    }

    auto attn_output = attn_->forward(q_rope, k_reshaped, v_reshaped);
    return o_proj_->forward(attn_output);
}

infinicore::Tensor Ernie4_5Attention::forward_paged_(const infinicore::Tensor &position_ids,
                                                     const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];
    ASSERT_EQ(batch_size, 1);

    auto [q, k, v] = qkv_proj_->forward_split(hidden_states_mutable);

    auto q_reshaped = q->view({seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k->view({seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({seq_len, num_key_value_heads_, head_dim_});

    if (should_use_rope_3d_(position_ids)) {
        q_reshaped = apply_rope_3d_(q_reshaped, position_ids);
        k_reshaped = apply_rope_3d_(k_reshaped, position_ids);
    } else {
        auto pos_ids_for_rope = position_ids_for_rope_(position_ids);
        rotary_emb_->forward(q_reshaped, pos_ids_for_rope, true);
        rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);
    }

    auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped);
    return o_proj_->forward(attn_output);
}

} // namespace infinilm::models::ernie4_5_moe_vl
