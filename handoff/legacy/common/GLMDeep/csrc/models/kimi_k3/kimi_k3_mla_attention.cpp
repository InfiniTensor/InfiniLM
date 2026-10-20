#include "execution/submit.hpp"
#include "../../weights/weights.hpp"
#include "kimi_k3_mla_attention.hpp"
#include <exception>
#include "runtime/runtime.hpp"
#include "runtime/utils.hpp"

#include "../../global_state/global_state.hpp"


#include <cmath>
#include <stdexcept>

namespace infinilm::models::kimi_k3 {
namespace {

infinilm::runtime::Tensor build_key(
    infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &kv_b,
    const infinilm::runtime::Tensor &k_rot,
    size_t num_heads,
    size_t qk_nope_head_dim,
    size_t qk_rope_head_dim) {
    auto key_shape = kv_b->shape();
    const size_t head_axis = key_shape.size() - 2;
    const size_t feature_dim = key_shape.size() - 1;
    key_shape[feature_dim] = qk_nope_head_dim + qk_rope_head_dim;

    auto key = infinilm::runtime::Tensor::empty(runtime, key_shape, kv_b->dtype());
    key->narrow({{feature_dim, 0, qk_nope_head_dim}})
        ->copy_from(runtime, kv_b->narrow({{feature_dim, 0, qk_nope_head_dim}}));

    auto k_rot_shape = key_shape;
    k_rot_shape[head_axis] = num_heads;
    k_rot_shape[feature_dim] = qk_rope_head_dim;
    infinilm::runtime::Strides k_rot_strides(key_shape.size());
    for (size_t i = 0; i < head_axis; ++i) {
        k_rot_strides[i] = k_rot->stride(i);
    }
    k_rot_strides[head_axis] = 0;
    k_rot_strides[feature_dim] = k_rot->stride(k_rot->ndim() - 1);
    auto k_rot_heads = k_rot->as_strided(k_rot_shape, k_rot_strides);
    key->narrow({{feature_dim, qk_nope_head_dim, qk_rope_head_dim}})
        ->copy_from(runtime, k_rot_heads);
    return key;
}

} // namespace

KimiK3MLAAttention::KimiK3MLAAttention(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("KimiK3MLAAttention construction requires eager Runtime");
    if (!model_config || rank_info.device != runtime.device() || rank_info.tp_size <= 0
        || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size
        || (rank_info.tp_size > 1 && !rank_info.comm))
        throw std::invalid_argument("Invalid explicit KimiK3MLAAttention construction resources");
    if (attention_backend != backends::AttentionBackend::STATIC_ATTN
        && attention_backend != backends::AttentionBackend::PAGED_ATTN
        && attention_backend != backends::AttentionBackend::FLASH_ATTN)
        throw std::invalid_argument("Invalid KimiK3MLAAttention attention backend");
    if (attention_backend == backends::AttentionBackend::STATIC_ATTN)
        throw std::runtime_error("KimiK3MLAAttention does not support static attention");
    try {
        layer_idx_ = layer_idx;
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    q_lora_rank_ = model_config->get<size_t>("q_lora_rank");
    kv_lora_rank_ = model_config->get<size_t>("kv_lora_rank");
    qk_nope_head_dim_ = model_config->get<size_t>("qk_nope_head_dim");
    qk_rope_head_dim_ = model_config->get<size_t>("qk_rope_head_dim");
    q_head_dim_ = qk_nope_head_dim_ + qk_rope_head_dim_;
    v_head_dim_ = model_config->get<size_t>("v_head_dim");
    const size_t num_heads = model_config->get<size_t>("num_attention_heads");
    if (!model_config->get<bool>("mla_use_nope")) {
        throw std::runtime_error("KimiK3MLAAttention only supports K3's NoPE MLA");
    }
    if (num_heads % static_cast<size_t>(rank_info.tp_size) != 0) {
        throw std::runtime_error("KimiK3MLAAttention: num_attention_heads must be divisible by tp_size");
    }
    local_num_heads_ = num_heads / static_cast<size_t>(rank_info.tp_size);
    const auto &dtype = model_config->get_dtype();
    const double eps = model_config->get<double>("rms_norm_eps");

        q_a_proj_ = std::make_shared<std::remove_reference_t<decltype(*q_a_proj_)>>(
            runtime, hidden_size, q_lora_rank_, false, dtype);
        this->bind_child("q_a_proj", q_a_proj_);
        q_a_layernorm_ = std::make_shared<std::remove_reference_t<decltype(*q_a_layernorm_)>>(
            runtime, q_lora_rank_, eps, dtype);
        this->bind_child("q_a_layernorm", q_a_layernorm_);
        q_b_proj_ = std::make_shared<std::remove_reference_t<decltype(*q_b_proj_)>>(
            runtime, q_lora_rank_, num_heads * q_head_dim_, false, dtype, rank_info.tp_rank, rank_info.tp_size);
        this->bind_child("q_b_proj", q_b_proj_);
        kv_a_proj_with_mqa_ = std::make_shared<std::remove_reference_t<decltype(*kv_a_proj_with_mqa_)>>(
            runtime, hidden_size, kv_lora_rank_ + qk_rope_head_dim_, false, dtype);
        this->bind_child("kv_a_proj_with_mqa", kv_a_proj_with_mqa_);
        kv_a_layernorm_ = std::make_shared<std::remove_reference_t<decltype(*kv_a_layernorm_)>>(
            runtime, kv_lora_rank_, eps, dtype);
        this->bind_child("kv_a_layernorm", kv_a_layernorm_);
        kv_b_proj_ = std::make_shared<std::remove_reference_t<decltype(*kv_b_proj_)>>(
            runtime, kv_lora_rank_, num_heads * (qk_nope_head_dim_ + v_head_dim_), false, dtype, rank_info.tp_rank, rank_info.tp_size);
        this->bind_child("kv_b_proj", kv_b_proj_);
        g_proj_ = std::make_shared<std::remove_reference_t<decltype(*g_proj_)>>(
            runtime, hidden_size, num_heads * v_head_dim_, false, dtype, rank_info.tp_rank, rank_info.tp_size);
        this->bind_child("g_proj", g_proj_);
        o_proj_ = std::make_shared<std::remove_reference_t<decltype(*o_proj_)>>(
            runtime, num_heads * v_head_dim_, hidden_size, false, dtype, rank_info.tp_rank, rank_info.tp_size, rank_info.comm);
        this->bind_child("o_proj", o_proj_);

    softmax_scale_ = 1.0f / std::sqrt(static_cast<float>(q_head_dim_));
        infinilm::layers::attention::init_kv_cache_quant_params(
            runtime,
            [this](const std::string &name, infinilm::weights::Weight parameter) {
                this->bind_weight(name, std::move(parameter));
            },
            kv_cache_k_scale_,
            kv_cache_v_scale_,
            model_config->get_kv_quant_scheme());
    attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(
        local_num_heads_, q_head_dim_, softmax_scale_, local_num_heads_, layer_idx_,
            kv_cache_k_scale_, kv_cache_v_scale_, attention_backend,
            model_config->get_kv_quant_scheme());

    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor KimiK3MLAAttention::forward(
    const infinilm::runtime::Tensor &hidden_states,
    global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    const auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    if (batch_size != 1) {
        throw std::runtime_error("KimiK3MLAAttention: paged attention expects flattened batch size 1");
    }
    auto input = hidden_states;
    auto q_lora = q_a_proj_->forward(runtime, input);
    q_lora = q_a_layernorm_->forward(runtime, q_lora);
    auto q = q_b_proj_->forward(runtime, q_lora);
    auto compressed = kv_a_proj_with_mqa_->forward(runtime, input);
    auto compressed_kv = compressed->narrow({{2, 0, kv_lora_rank_}});
    auto k_rot = compressed->narrow({{2, kv_lora_rank_, qk_rope_head_dim_}});
    auto normalized_kv = kv_a_layernorm_->forward(runtime, compressed_kv);
    auto kv = kv_b_proj_->forward(runtime, normalized_kv);

    q = q->view({seq_len, local_num_heads_, q_head_dim_});
    kv = kv->view({seq_len, local_num_heads_, qk_nope_head_dim_ + v_head_dim_});
    k_rot = k_rot->squeeze(0);

    const size_t feature_dim = kv->ndim() - 1;
    auto key = build_key(
        runtime, kv, k_rot, local_num_heads_, qk_nope_head_dim_, qk_rope_head_dim_);
    auto value = kv->narrow({{feature_dim, qk_nope_head_dim_, v_head_dim_}});
    // Flash/paged attention caches use the Q/K head width. Preserve the
    // model's V width by zero-padding only the unused tail before caching,
    // then take the original V slice after attention below.
    if (v_head_dim_ > q_head_dim_) throw std::invalid_argument("Kimi MLA V head exceeds Q/K head width");
    if (v_head_dim_ < q_head_dim_) {
        auto padded_shape = value->shape();padded_shape.back() = q_head_dim_;
        auto padded_value = infinilm::runtime::Tensor::empty(runtime,padded_shape,value->dtype());
        infinilm::execution::submit<infini::ops::Fill>(runtime,
            infini::ops::Fill::DefaultConfig(runtime.device().type()),padded_value,double{0.0},padded_value);
        padded_value->narrow({{feature_dim,0,v_head_dim_}})->copy_from(runtime,value);
        value = std::move(padded_value);
    }
    auto padded_output = attn_->forward(q, key, value, forward_context)
                             ->view({batch_size, seq_len, local_num_heads_, q_head_dim_});
    auto output = infinilm::runtime::Tensor::empty(
        runtime, {batch_size, seq_len, local_num_heads_, v_head_dim_},
        padded_output->dtype());
    auto gate = g_proj_->forward(runtime, input)
                    ->view({batch_size, seq_len, local_num_heads_, v_head_dim_});
    INFINILM_ASSERT(runtime.device().type() != infinilm::runtime::Device::Type::kCpu
        && runtime.device().type() != infinilm::runtime::Device::Type::kMars
        && runtime.device().type() != infinilm::runtime::Device::Type::kThead);
    infinilm::execution::submit<infini::ops::Sigmoid>(
        runtime, infini::ops::Sigmoid::DefaultConfig(runtime.device().type()), gate, gate);
    INFINILM_TRACE_OP(Sigmoid, gate);
    {
        const auto ops_input = padded_output->narrow({{3, 0, v_head_dim_}});
        INFINILM_ASSERT(runtime.device().type() != infinilm::runtime::Device::Type::kCpu
            && runtime.device().type() != infinilm::runtime::Device::Type::kMars
            && runtime.device().type() != infinilm::runtime::Device::Type::kThead);
        infinilm::execution::submit<infini::ops::Mul>(
            runtime, infini::ops::Mul::DefaultConfig(runtime.device().type()), ops_input, gate, output);
        INFINILM_TRACE_OP(Mul, output);
    }
    output = output->view({batch_size, seq_len, local_num_heads_ * v_head_dim_});
    return o_proj_->forward(runtime, output);
}

} // namespace infinilm::models::kimi_k3
