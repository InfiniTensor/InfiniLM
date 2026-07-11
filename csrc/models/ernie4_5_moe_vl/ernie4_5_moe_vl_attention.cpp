#include "ernie4_5_moe_vl_attention.hpp"
#include "../../global_state/global_state.hpp"
#include "../../layers/attention/attention.hpp"
#include "../../utils.hpp"
#include "infinicore/ops.hpp"

#include <cmath>
#include <tuple>
#include <utility>
#include <vector>

namespace infinilm::models::ernie4_5_moe_vl {

Ernie4_5_VLMoeAttention::Ernie4_5_VLMoeAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                 size_t layer_idx,
                                                 const infinicore::Device &device) {
    layer_idx_ = layer_idx;
    hidden_size_ = model_config->get<size_t>("hidden_size");
    head_dim_ = model_config->get<size_t>("head_dim"); // supplied by create_ernie4_5_moe_vl_model_config

    const auto &dtype{model_config->get_dtype()};
    size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
    size_t total_num_kv_heads = model_config->get<size_t>("num_key_value_heads");
    bool use_bias = model_config->get_or<bool>("use_bias", false);

    attention_backend_ = infinilm::global_state::get_infinilm_config().attention_backend;
    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
    int tp_size = infinilm::global_state::get_tensor_model_parallel_world_size();
    if ((total_num_kv_heads < static_cast<size_t>(tp_size)) || (0 != (total_num_kv_heads % tp_size))) {
        throw std::runtime_error("Ernie4_5_VLMoeAttention: num_key_value_heads must be divisible by tp_size");
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
        use_bias, dtype, device, tp_rank, tp_size, rank_info.comm);

    // Base 1D RoPE. ERNIE-4.5-VL uses GPT-J / interleaved rotary (HF
    // RopeEmbedding.apply_rotary: sin_pos=[θ0,θ0,θ1,θ1,...], rotate_half pairs
    // adjacent dims (q0,q1),(q2,q3),...), NOT the framework-default GPT-NEOX
    // (half-split). Used directly for text positions and reused (algo+theta+
    // head_dim) when building the 3D-mrope tables.
    // GPT-J/interleaved algo is set on the model config by
    // Ernie4_5_VLMoeForConditionalGeneration before the text model is built;
    // get_rope() picks it up from there.
    rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(model_config, device);
    rope_algo_ = rotary_emb_->algo();
    rope_theta_ = model_config->get<double>("rope_theta");
    // 3D mrope sections (time,height,width) from rope_scaling.mrope_section, e.g.
    // [22,22,20] summing to head_dim/2 = 64. Empty -> plain 1D rope.
    // VERIFY(GPU): confirm axis order and that mrope_section matches HF
    // modeling_ernie4_5_vl (Qwen2-VL-style section->axis assignment is assumed).
    {
        const auto &cfg = model_config->get_config_json();
        if (cfg.contains("rope_scaling") && cfg["rope_scaling"].is_object()
            && cfg["rope_scaling"].contains("mrope_section")) {
            for (const auto &v : cfg["rope_scaling"]["mrope_section"]) {
                mrope_section_.push_back(v.get<size_t>());
            }
        }
    }

    float scaling = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(
        num_attention_heads_, head_dim_, scaling, num_key_value_heads_, layer_idx_,
        kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_);

    infinilm::layers::attention::init_kv_cache_quant_params(register_fn, device, kv_cache_k_scale_, kv_cache_v_scale_);
}

std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor>
Ernie4_5_VLMoeAttention::build_mrope_(const infinicore::Tensor &position_ids,
                                      const infinicore::DataType &dtype,
                                      const infinicore::Device &device) const {
    // position_ids: [3, seq] int64 = (time, height, width).
    auto pos_cpu = position_ids->to(infinicore::Device::cpu())->contiguous();
    size_t seq = pos_cpu->shape()[1];
    const auto *pp = reinterpret_cast<const int64_t *>(pos_cpu->data());

    size_t cache_dim = head_dim_ / 2;

    // Per-frequency axis assignment, matching HF RopeEmbedding.apply_rotary_3d
    // (NOT a sequential mrope_section split): with freq_allocation = mrope_section
    // last entry (=20), the low frequencies [0, cache_dim-freq_allocation) alternate
    // height(even)/width(odd), and the high frequencies [.., cache_dim) are time.
    //   pos rows: 0 = time, 1 = height, 2 = width  (from processor get_rope_index).
    // For text (all 3 rows equal) the assignment is irrelevant; it only matters for
    // image/video where the axes differ.
    size_t freq_allocation = mrope_section_.empty() ? 0 : mrope_section_.back();
    size_t split = (freq_allocation <= cache_dim) ? (cache_dim - freq_allocation) : cache_dim;
    std::vector<size_t> axis_of(cache_dim, 0);
    for (size_t j = 0; j < cache_dim; ++j) {
        if (j < split) {
            axis_of[j] = (j % 2 == 0) ? 1 /*height*/ : 2 /*width*/;
        } else {
            axis_of[j] = 0 /*time*/;
        }
    }

    // GPT-J style inverse frequency theta^(-2j/head_dim); the rope algo only
    // controls dimension pairing in the kernel (kept == rotary_emb_'s algo).
    std::vector<float> sin_f(seq * cache_dim);
    std::vector<float> cos_f(seq * cache_dim);
    for (size_t j = 0; j < cache_dim; ++j) {
        float inv_freq = 1.0f / std::pow(static_cast<float>(rope_theta_),
                                         2.0f * static_cast<float>(j) / static_cast<float>(head_dim_));
        size_t ax = axis_of[j];
        for (size_t i = 0; i < seq; ++i) {
            float p = static_cast<float>(pp[ax * seq + i]);
            float angle = p * inv_freq;
            sin_f[i * cache_dim + j] = std::sin(angle);
            cos_f[i * cache_dim + j] = std::cos(angle);
        }
    }

    auto to_table = [&](const std::vector<float> &f) -> infinicore::Tensor {
        auto out = infinicore::Tensor::empty({seq, cache_dim}, dtype, device);
        if (dtype == infinicore::DataType::F32) {
            auto cpu = infinicore::Tensor::from_blob(const_cast<float *>(f.data()), {seq, cache_dim},
                                                     infinicore::DataType::F32, infinicore::Device::cpu());
            out->copy_from(cpu);
        } else if (dtype == infinicore::DataType::BF16) {
            std::vector<uint16_t> h(f.size());
            for (size_t i = 0; i < f.size(); ++i) {
                h[i] = f32_to_bf16(f[i]);
            }
            auto cpu = infinicore::Tensor::from_blob(h.data(), {seq, cache_dim},
                                                     infinicore::DataType::BF16, infinicore::Device::cpu());
            out->copy_from(cpu);
        } else if (dtype == infinicore::DataType::F16) {
            std::vector<uint16_t> h(f.size());
            for (size_t i = 0; i < f.size(); ++i) {
                h[i] = f32_to_f16(f[i]);
            }
            auto cpu = infinicore::Tensor::from_blob(h.data(), {seq, cache_dim},
                                                     infinicore::DataType::F16, infinicore::Device::cpu());
            out->copy_from(cpu);
        } else {
            throw std::runtime_error("build_mrope_: unsupported dtype for rope tables");
        }
        return out;
    };

    auto sin_tbl = to_table(sin_f);
    auto cos_tbl = to_table(cos_f);

    // Position index = arange(seq): row i of the table holds token i's rotation.
    std::vector<int64_t> idx(seq);
    for (size_t i = 0; i < seq; ++i) {
        idx[i] = static_cast<int64_t>(i);
    }
    auto idx_cpu = infinicore::Tensor::from_blob(idx.data(), {seq}, infinicore::DataType::I64,
                                                 infinicore::Device::cpu());
    auto pos_index = idx_cpu->to(device);

    return std::make_tuple(sin_tbl, cos_tbl, pos_index);
}

infinicore::Tensor Ernie4_5_VLMoeAttention::forward(const infinicore::Tensor &positions,
                                                    const infinicore::Tensor &hidden_states) const {
    if (::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
        return forward_static_(positions, hidden_states);
    }
    return forward_paged_(positions, hidden_states);
}

infinicore::Tensor Ernie4_5_VLMoeAttention::forward_static_(const infinicore::Tensor &position_ids,
                                                            const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    auto [q, k, v] = qkv_proj_->forward_split(hidden_states_mutable);

    auto q_reshaped = q->view({batch_size, seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

    // RoPE. position_ids is [3, seq] for 3D mrope (time,height,width), or 1D /
    // [1, seq] for text. q_rope is a fresh contiguous buffer in the
    // [batch, heads, seq, dim] layout StaticAttentionImpl expects (applying RoPE
    // in place on the non-contiguous narrowed q_reshaped would leave incompatible
    // strides).
    auto pos_shape = position_ids->shape();
    bool use_mrope = (pos_shape.size() == 2) && (pos_shape[0] == 3) && !mrope_section_.empty();

    auto q_rope = infinicore::Tensor::empty(
        {batch_size, num_attention_heads_, seq_len, head_dim_},
        q_reshaped->dtype(), q_reshaped->device())->permute({0, 2, 1, 3});

    if (use_mrope) {
        auto [sin_tbl, cos_tbl, pos_index] =
            build_mrope_(position_ids, q_reshaped->dtype(), q_reshaped->device());
        infinicore::op::rope_(q_rope, q_reshaped, pos_index, sin_tbl, cos_tbl, rope_algo_);
        infinicore::op::rope_(k_reshaped, k_reshaped, pos_index, sin_tbl, cos_tbl, rope_algo_);
    } else {
        infinicore::Tensor pos_ids_for_rope = position_ids;
        if (pos_shape.size() == 2) {
            auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
            pos_ids_for_rope = pos_narrowed->contiguous()->view({pos_shape[1]});
        } else if (pos_shape.size() == 1) {
            pos_ids_for_rope = position_ids->contiguous();
        }
        rotary_emb_->forward(q_rope, q_reshaped, pos_ids_for_rope);
        rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);
    }

    auto attn_output = attn_->forward(q_rope, k_reshaped, v_reshaped);
    return o_proj_->forward(attn_output);
}

infinicore::Tensor Ernie4_5_VLMoeAttention::forward_paged_(const infinicore::Tensor &position_ids,
                                                           const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];
    ASSERT_EQ(batch_size, 1);

    auto [q, k, v] = qkv_proj_->forward_split(hidden_states_mutable);

    // Make contiguous before view: q/k/v are narrow slices of the fused QKV output
    // and therefore non-contiguous; view on a non-contiguous tensor can fail in paged
    // attention backends that permute+view the result.
    auto q_cont = q->contiguous();
    auto k_cont = k->contiguous();
    auto v_cont = v->contiguous();

    auto q_reshaped = q_cont->view({seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k_cont->view({seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v_cont->view({seq_len, num_key_value_heads_, head_dim_});

    // RoPE (see forward_static_). 3D mrope when position_ids is [3, seq].
    auto pos_shape = position_ids->shape();
    bool use_mrope = (pos_shape.size() == 2) && (pos_shape[0] == 3) && !mrope_section_.empty();
    if (use_mrope) {
        auto [sin_tbl, cos_tbl, pos_index] =
            build_mrope_(position_ids, q_reshaped->dtype(), q_reshaped->device());
        infinicore::op::rope_(q_reshaped, q_reshaped, pos_index, sin_tbl, cos_tbl, rope_algo_);
        infinicore::op::rope_(k_reshaped, k_reshaped, pos_index, sin_tbl, cos_tbl, rope_algo_);
    } else {
        infinicore::Tensor pos_ids_for_rope = position_ids;
        if (pos_shape.size() == 2) {
            auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
            pos_ids_for_rope = pos_narrowed->view({pos_shape[1]});
        } else if (pos_shape.size() == 1) {
            pos_ids_for_rope = position_ids;
        }
        rotary_emb_->forward(q_reshaped, pos_ids_for_rope, true);
        rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);
    }

    auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped);
    return o_proj_->forward(attn_output);
}

} // namespace infinilm::models::ernie4_5_moe_vl
