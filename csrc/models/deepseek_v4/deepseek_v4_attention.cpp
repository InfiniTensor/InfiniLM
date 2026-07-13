#include "deepseek_v4_attention.hpp"

#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"
#include "deepseek_v4_linear.hpp"
#include "deepseek_v4_utils.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/cat.hpp"
#include "infinicore/ops/deepseek_v4_compressed_decode.hpp"
#include "infinicore/ops/deepseek_v4_swa_decode.hpp"
#include "infinicore/ops/deepseek_v4_swa_prefill.hpp"
#include "infinicore/ops/linear.hpp"
#include "infinicore/ops/unweighted_rms_norm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infinilm::models::deepseek_v4 {
namespace {

bool disable_compressed_empty_fastpath() {
    static const bool value = std::getenv("DSV4_DISABLE_COMPRESSED_EMPTY_FASTPATH") != nullptr;
    return value;
}

bool disable_compressed_kv_cache() {
    static const bool value = std::getenv("DSV4_DISABLE_COMPRESSED_KV_CACHE") != nullptr;
    return value;
}

bool disable_incremental_compressor() {
    static const bool value = std::getenv("DSV4_DISABLE_INCREMENTAL_COMPRESSOR") != nullptr;
    return value;
}

bool disable_cached_no_index_sentinel() {
    static const bool value = std::getenv("DSV4_DISABLE_CACHED_NO_INDEX_SENTINEL") != nullptr;
    return value;
}

bool disable_precomputed_block_positions() {
    static const bool value = std::getenv("DSV4_DISABLE_PRECOMPUTED_BLOCK_POSITIONS") != nullptr;
    return value;
}

bool disable_contiguous_window_fastpath() {
    static const bool value = std::getenv("DSV4_DISABLE_CONTIGUOUS_WINDOW_FASTPATH") != nullptr;
    return value;
}

bool disable_swa_prefill_position_reuse() {
    static const bool value = std::getenv("DSV4_DISABLE_SWA_PREFILL_POSITION_REUSE") != nullptr;
    return value;
}

bool disable_decode_position_fastpath() {
    static const bool value = std::getenv("DSV4_DISABLE_DECODE_POSITION_FASTPATH") != nullptr;
    return value;
}

bool disable_reuse_raw_position_tensor() {
    static const bool value = std::getenv("DSV4_DISABLE_REUSE_RAW_POSITION_TENSOR") != nullptr;
    return value;
}

bool disable_device_rope_positions() {
    static const bool value = std::getenv("DSV4_DISABLE_DEVICE_ROPE_POSITIONS") != nullptr;
    return value;
}

infinicore::Tensor position_tensor_for_query(const infinicore::Tensor &raw_positions,
                                             const std::vector<int64_t> &positions,
                                             size_t query_start,
                                             size_t query_len,
                                             const infinicore::Device &device) {
    if (!disable_reuse_raw_position_tensor()
        && raw_positions
        && raw_positions->device() == device
        && (raw_positions->dtype() == infinicore::DataType::I64
            || raw_positions->dtype() == infinicore::DataType::I32)
        && raw_positions->numel() == query_len
        && (query_start == 0 || query_len == 1)) {
        auto query = raw_positions->is_contiguous()
                       ? raw_positions
                       : raw_positions->contiguous();
        return query->view({query_len});
    }
    std::vector<int64_t> query_positions(query_len);
    for (size_t tq = 0; tq < query_len; ++tq) {
        query_positions[tq] = positions[query_start + tq];
    }
    return int64_vector_to_tensor(query_positions, {query_len}, device);
}

bool positions_are_contiguous(const std::vector<int64_t> &positions) {
    for (size_t i = 1; i < positions.size(); ++i) {
        if (positions[i] != positions[i - 1] + 1) {
            return false;
        }
    }
    return true;
}

bool has_no_visible_compressed_blocks(size_t compress_ratio,
                                      const std::vector<int64_t> &positions,
                                      size_t query_start,
                                      size_t query_len) {
    if (compress_ratio == 0) {
        return true;
    }
    if (disable_compressed_empty_fastpath() || query_len == 0
        || positions.size() < query_start + query_len) {
        return false;
    }
    const int64_t ratio = static_cast<int64_t>(compress_ratio);
    for (size_t tq = 0; tq < query_len; ++tq) {
        const int64_t visible_blocks = (positions[query_start + tq] + 1) / ratio;
        if (visible_blocks > 0) {
            return false;
        }
    }
    return true;
}

std::vector<int64_t> normalize_uniform_packed_positions(
    const infinicore::Tensor &positions,
    size_t batch_size,
    size_t seq_len) {
    if (batch_size <= 1) {
        return normalize_positions(positions, seq_len);
    }

    const auto packed_positions = tensor_to_int64_vector(positions);
    if (packed_positions.size() != batch_size * seq_len) {
        throw std::runtime_error(
            "DeepseekV4Attention: packed positions size does not match batch and sequence length");
    }

    std::vector<int64_t> shared_positions(
        packed_positions.begin(), packed_positions.begin() + seq_len);
    for (size_t batch_idx = 1; batch_idx < batch_size; ++batch_idx) {
        const auto begin = packed_positions.begin() + batch_idx * seq_len;
        if (!std::equal(shared_positions.begin(), shared_positions.end(), begin)) {
            throw std::runtime_error(
                "DeepseekV4Attention: paged batch requires equal-length requests with identical positions");
        }
    }
    return shared_positions;
}

} // namespace

DeepseekV4Attention::DeepseekV4Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                         const infinicore::Device &device)
    : DeepseekV4Attention(std::move(model_config), 0, device) {
}

DeepseekV4Attention::DeepseekV4Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                         size_t layer_idx,
                                         const infinicore::Device &device)
    : layer_idx_(layer_idx),
      hidden_size_(model_config->get<size_t>("hidden_size")),
      global_num_attention_heads_(model_config->get<size_t>("num_attention_heads")),
      num_attention_heads_(global_num_attention_heads_),
      num_key_value_heads_(model_config->get_or<size_t>("num_key_value_heads", 1)),
      head_dim_(model_config->get<size_t>("head_dim")),
      q_lora_rank_(model_config->get<size_t>("q_lora_rank")),
      o_lora_rank_(model_config->get<size_t>("o_lora_rank")),
      global_o_groups_(model_config->get<size_t>("o_groups")),
      o_groups_(global_o_groups_),
      o_a_input_size_(global_num_attention_heads_ * head_dim_ / global_o_groups_),
      o_a_output_size_(o_lora_rank_ * global_o_groups_),
      sliding_window_(model_config->get_or<size_t>("sliding_window", 0)),
      rms_norm_eps_(model_config->get<double>("rms_norm_eps")),
      rotary_emb_(model_config, layer_idx, device),
      softmax_scale_(1.0f / std::sqrt(static_cast<float>(head_dim_))) {
    const auto &dtype = model_config->get_dtype();
    const size_t q_output_size = num_attention_heads_ * head_dim_;
    const size_t compress_ratio = rotary_emb_.compress_ratio();
    const size_t max_position_embeddings = model_config->get_or<size_t>("max_position_embeddings", 4096);
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    auto quantization_method = deepseek_v4_linear_quantization(model_config, true);
    auto none_quantization = deepseek_v4_linear_quantization(model_config, false);

    INFINICORE_NN_MODULE_INIT(q_norm, q_lora_rank_, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(wq_a, hidden_size_, q_lora_rank_, quantization_method, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(wq_b, q_lora_rank_, q_output_size, quantization_method, false, dtype, device, rank_info.tp_rank, rank_info.tp_size);

    INFINICORE_NN_MODULE_INIT(kv_norm, head_dim_, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(wkv, hidden_size_, head_dim_, quantization_method, false, dtype, device);

    INFINICORE_NN_MODULE_INIT(wo_a, o_a_input_size_, o_a_output_size_, none_quantization, false, dtype, device, rank_info.tp_rank, rank_info.tp_size);
    INFINICORE_NN_MODULE_INIT(wo_b, o_a_output_size_, hidden_size_, quantization_method, false, dtype, device, rank_info.tp_rank, rank_info.tp_size, rank_info.comm);

    if (compress_ratio == 4) {
        INFINICORE_NN_MODULE_INIT(indexer, model_config, compress_ratio, device);
    }
    if (compress_ratio > 1) {
        INFINICORE_NN_MODULE_INIT(compressor, model_config, compress_ratio, head_dim_, device);
        no_index_sentinel_ = int64_vector_to_tensor({-1}, {1}, device);
        const size_t max_blocks = max_position_embeddings / compress_ratio;
        std::vector<int64_t> block_positions(max_blocks);
        for (size_t block = 0; block < max_blocks; ++block) {
            block_positions[block] = static_cast<int64_t>(block * compress_ratio);
        }
        block_position_table_ = int64_vector_to_tensor(block_positions, {max_blocks}, device);
    }

    const int tp_size = rank_info.tp_size;
    if (num_attention_heads_ % static_cast<size_t>(tp_size) != 0) {
        throw std::runtime_error("DeepseekV4Attention: num_attention_heads must be divisible by tp_size");
    }
    if (global_o_groups_ % static_cast<size_t>(tp_size) != 0) {
        throw std::runtime_error("DeepseekV4Attention: o_groups must be divisible by tp_size");
    }
    num_attention_heads_ /= static_cast<size_t>(tp_size);
    o_groups_ = global_o_groups_ / static_cast<size_t>(tp_size);
    o_a_output_size_ = o_lora_rank_ * o_groups_;
    if (num_key_value_heads_ >= static_cast<size_t>(tp_size)) {
        num_key_value_heads_ /= static_cast<size_t>(tp_size);
    } else {
        num_key_value_heads_ = 1;
    }
    if (num_attention_heads_ % o_groups_ != 0) {
        throw std::runtime_error("DeepseekV4Attention: local num_attention_heads must be divisible by local o_groups");
    }

    INFINICORE_NN_PARAMETER_INIT(attn_sink, ({global_num_attention_heads_}, infinicore::DataType::F32, device,
                                             0, rank_info.tp_rank, rank_info.tp_size));

    auto register_fn = [this](const std::string &n, infinicore::nn::Parameter p) { this->register_parameter(n, std::move(p)); };
    attention_backend_ = infinilm::global_state::get_infinilm_config().attention_backend;

    attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(
        num_attention_heads_, head_dim_, softmax_scale_, num_key_value_heads_, layer_idx_,
        kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_);

    infinilm::layers::attention::init_kv_cache_quant_params(
        register_fn, device, kv_cache_k_scale_, kv_cache_v_scale_);
}

infinicore::Tensor DeepseekV4Attention::forward(const infinicore::Tensor &positions,
                                                const infinicore::Tensor &hidden_states) const {
    if (::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
        return forward_static_(positions, hidden_states);
    }
    return forward_paged_(positions, hidden_states);
}

infinicore::Tensor DeepseekV4Attention::forward_static_(const infinicore::Tensor &positions,
                                                        const infinicore::Tensor &hidden_states) const {
    const auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    auto hidden_states_mutable = hidden_states;
    const auto pos = normalize_positions(positions, seq_len);

    auto q_residual = wq_a_->forward(hidden_states_mutable);
    q_residual = q_norm_->forward(q_residual);

    auto q = wq_b_->forward(q_residual)->view({batch_size, seq_len, num_attention_heads_, head_dim_});

    auto q_normed = infinicore::op::unweighted_rms_norm(q->contiguous(), static_cast<float>(rms_norm_eps_));
    const infinicore::Tensor rope_positions = disable_device_rope_positions()
                                                ? infinicore::Tensor{}
                                                : positions;
    q_normed = rotary_emb_.forward(q_normed, pos, rope_positions);

    auto kv = wkv_->forward(hidden_states_mutable);

    kv = kv_norm_->forward(kv);

    auto key_states = rotary_emb_.forward(
        kv->view({batch_size, seq_len, num_key_value_heads_, head_dim_}), pos, rope_positions);

    auto attn_output = attention_prefill_(positions, q_normed, key_states, hidden_states_mutable, q_residual);

    return apply_grouped_output_projection_(attn_output);
}

infinicore::Tensor DeepseekV4Attention::forward_paged_(const infinicore::Tensor &positions,
                                                       const infinicore::Tensor &hidden_states) const {
    const auto packed_shape = hidden_states->shape();
    size_t batch_size = packed_shape[0];
    const auto &input_offsets = infinilm::global_state::get_forward_context().attn_metadata.input_offsets;
    if (batch_size == 1 && input_offsets && input_offsets.value()->numel() > 2) {
        batch_size = input_offsets.value()->numel() - 1;
    }
    if (batch_size == 0 || packed_shape[1] % batch_size != 0) {
        throw std::runtime_error(
            "DeepseekV4Attention: packed token count is not divisible by batch size");
    }
    const size_t seq_len = packed_shape[1] / batch_size;
    auto hidden_states_mutable = hidden_states->view(
        {batch_size, seq_len, hidden_size_});
    std::vector<int64_t> pos;
    const bool decode_position_fastpath = !disable_decode_position_fastpath() && runtime_state_.seq_len > 0 && seq_len == 1;
    if (decode_position_fastpath) {
        pos = {static_cast<int64_t>(runtime_state_.seq_len)};
    } else {
        pos = normalize_uniform_packed_positions(positions, batch_size, seq_len);
    }
    infinicore::Tensor shared_position_tensor = positions;
    if (batch_size > 1 && positions->numel() == batch_size * seq_len) {
        shared_position_tensor = positions->narrow({{0, 0, seq_len}});
    }

    auto q_residual = q_norm_->forward(wq_a_->forward(hidden_states_mutable));

    auto q = wq_b_->forward(q_residual)->view({batch_size, seq_len, num_attention_heads_, head_dim_});

    auto q_normed = infinicore::op::unweighted_rms_norm(q->contiguous(), static_cast<float>(rms_norm_eps_)); // shape []

    const infinicore::Tensor rope_positions = disable_device_rope_positions()
                                                ? infinicore::Tensor{}
                                                : shared_position_tensor;

    // std::cout << "before rotary_emb_ q_normed:: " << q_normed->info() << std::endl;
    // std::cout << "before rotary_emb_ rope_positions:: " << rope_positions->info() << std::endl;
    // std::cout << "before rotary_emb_ rope_positions:: " << rope_positions << std::endl;
    // for (int i = 0; i < pos.size(); ++i) {
    //     std::cout << "pos[" << i << "] = " << pos[i] << std::endl;
    // }
    /*
    before rotary_emb_ q_normed:: Tensor: shape[ 1 12 64 512 ] strides[ 393216 32768 512 1 ] dtype=bfloat16 device=HYGON:0
    before rotary_emb_ rope_positions:: Tensor: shape[ 12 ] strides[ 1 ] dtype=int64 device=HYGON:0
    before rotary_emb_ rope_positions:: tensor([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11], device='HYGON:0', dtype=infinicore.int64)

    pos[0] = 0
    pos[1] = 1
    pos[2] = 2
    pos[3] = 3
    pos[4] = 4
    pos[5] = 5
    pos[6] = 6
    pos[7] = 7
    pos[8] = 8
    pos[9] = 9
    pos[10] = 10
    pos[11] = 11
    */

    // q_normed的是shaoe是 [ 1 12 64 512 ]，对512的后64个数值做rope

    q_normed = rotary_emb_.forward(q_normed, pos, rope_positions);

    auto kv = kv_norm_->forward(wkv_->forward(hidden_states_mutable));

    auto key_states = rotary_emb_.forward(kv->view({batch_size, seq_len, num_key_value_heads_, head_dim_}), pos, rope_positions);

    DeepseekV4AttentionStep step;
    step.raw_positions = shared_position_tensor;
    step.positions = pos;
    step.query_len = seq_len;
    const bool is_decode = runtime_state_.seq_len > 0 && seq_len == 1 && !pos.empty() && pos[0] >= static_cast<int64_t>(runtime_state_.seq_len);

    step.mode = is_decode ? DeepseekV4AttentionMode::Decode : DeepseekV4AttentionMode::Prefill;
    step.query_start = is_decode ? runtime_state_.seq_len : 0;

    infinicore::Tensor attn_output;
    if (step.is_decode()) {
        runtime_state_.append(hidden_states_mutable, q_residual, key_states, step.positions);
        if (has_no_visible_compressed_blocks(rotary_emb_.compress_ratio(),
                                             runtime_state_.positions,
                                             step.query_start,
                                             step.query_len)
            && q_normed->device().getType() != infinicore::Device::Type::CPU) {
            attn_output = sliding_attention_gpu_(
                q_normed, runtime_state_.key_states, runtime_state_.positions,
                step.query_start, step.raw_positions);
        } else {
            attn_output = compressed_attention_gpu_(
                q_normed, runtime_state_.key_states, runtime_state_.hidden_states,
                runtime_state_.q_residual, runtime_state_.positions,
                step.query_start, step.raw_positions);
        }
    } else {
        runtime_state_.initialize(hidden_states_mutable, q_residual, key_states, step.positions);
        attn_output = attention_prefill_(shared_position_tensor, q_normed, key_states, hidden_states_mutable, q_residual);
        const size_t compress_ratio = rotary_emb_.compress_ratio();
        const bool hidden_history_is_cached = !compressor_ || compress_ratio == 0
                                           || (!disable_compressed_kv_cache()
                                               && !disable_incremental_compressor());
        if (hidden_history_is_cached && sliding_window_ > 0) {
            runtime_state_.retain_recent_hidden(sliding_window_);
        }
    }

    return apply_grouped_output_projection_(attn_output)->view(packed_shape);
}

infinicore::Tensor DeepseekV4Attention::rotate_compressed_blocks_(
    const infinicore::Tensor &blocks,
    const std::vector<int64_t> &positions,
    size_t total_len,
    size_t first_block) const {
    if (!blocks || blocks->shape().size() != 3 || blocks->shape()[1] == 0) {
        return blocks;
    }
    const auto shape = blocks->shape();
    const size_t block_count = shape[1];
    const size_t compress_ratio = rotary_emb_.compress_ratio();
    if (compress_ratio == 0 || positions.size() < total_len
        || first_block + block_count > total_len / compress_ratio) {
        throw std::runtime_error(
            "DeepseekV4Attention: compressed block rotation range is invalid");
    }

    std::vector<int64_t> block_positions(block_count);
    for (size_t local_block = 0; local_block < block_count; ++local_block) {
        const size_t block = first_block + local_block;
        const size_t token = std::min(block * compress_ratio, total_len - 1);
        block_positions[local_block]
            = (positions[token] / static_cast<int64_t>(compress_ratio))
            * static_cast<int64_t>(compress_ratio);
    }

    infinicore::Tensor device_positions;
    const bool standard_positions = !positions.empty()
                                 && positions.front() == 0
                                 && positions.back()
                                        == static_cast<int64_t>(total_len - 1);
    if (standard_positions && block_position_table_
        && block_position_table_->numel() >= first_block + block_count) {
        device_positions = block_position_table_->narrow(
            {{0, first_block, block_count}});
    } else {
        device_positions = int64_vector_to_tensor(
            block_positions, {block_count}, blocks->device());
    }

    rotary_emb_.forward(
        blocks->view({shape[0], block_count, 1, shape[2]}),
        block_positions,
        device_positions);
    return blocks;
}

infinicore::Tensor DeepseekV4Attention::compressed_attention_gpu_(const infinicore::Tensor &q_rope,
                                                                  const infinicore::Tensor &key_states,
                                                                  const infinicore::Tensor &hidden_states,
                                                                  const infinicore::Tensor &q_residual,
                                                                  const std::vector<int64_t> &positions,
                                                                  size_t query_start,
                                                                  const infinicore::Tensor &raw_positions) const {
    const auto q_shape = q_rope->shape();
    const size_t batch_size = q_shape[0];
    const size_t query_len = q_shape[1];
    const size_t num_heads = q_shape[2];
    const size_t head_dim = q_shape[3];
    const size_t total_len = key_states->shape()[1];
    const size_t window = sliding_window_ == 0 ? total_len : sliding_window_;
    const size_t compress_ratio = rotary_emb_.compress_ratio();

    infinicore::Tensor kv_comp_tensor;
    size_t comp_batch = 0;
    size_t nb = 0;
    size_t index_top_k = 0;
    infinicore::Tensor indexed_blocks_tensor;
    if (q_rope->device().getType() == infinicore::Device::Type::CPU) {
        throw std::runtime_error("DeepseekV4Attention: compressed attention requires a GPU device");
    }
    if (!compressor_ || compress_ratio == 0) {
        throw std::runtime_error("DeepseekV4Attention: compressed attention requires a compressor");
    }
    const size_t expected_nb = total_len / compress_ratio;
    if (expected_nb == 0) {
        throw std::runtime_error("DeepseekV4Attention: no compressed block is visible");
    }
    if (!disable_compressed_kv_cache()
        && runtime_state_.kv_comp && runtime_state_.kv_comp_batch == batch_size
        && runtime_state_.kv_comp_blocks == expected_nb) {
        kv_comp_tensor = runtime_state_.kv_comp;
        comp_batch = runtime_state_.kv_comp_batch;
        nb = runtime_state_.kv_comp_blocks;
    } else if (!disable_compressed_kv_cache()
               && !disable_incremental_compressor()
               && query_len == 1
               && runtime_state_.kv_comp
               && runtime_state_.kv_comp_batch == batch_size
               && expected_nb == runtime_state_.kv_comp_blocks + 1) {
        const size_t recent_len = compress_ratio == 4
                                    ? 2 * compress_ratio
                                    : compress_ratio;
        const size_t available_hidden_len = hidden_states->shape()[1];
        if (available_hidden_len >= recent_len) {
            auto recent_hidden = hidden_states->narrow(
                                                  {{1, available_hidden_len - recent_len,
                                                    recent_len}})
                                     ->contiguous();
            size_t recent_batch = 0;
            size_t recent_blocks = 0;
            auto recent_compressed = compressor_->forward_tensor(
                recent_hidden, recent_batch, recent_blocks);
            if (recent_batch == batch_size && recent_blocks > 0) {
                auto newest_block = recent_compressed->narrow(
                                                         {{1, recent_blocks - 1, 1}})
                                        ->contiguous();
                newest_block = rotate_compressed_blocks_(
                    newest_block, positions, total_len, expected_nb - 1);
                runtime_state_.append_compressed_kv(newest_block);
                kv_comp_tensor = runtime_state_.kv_comp;
                comp_batch = runtime_state_.kv_comp_batch;
                nb = runtime_state_.kv_comp_blocks;
            }
        }
        if (!kv_comp_tensor) {
            if (hidden_states->shape()[1] != total_len) {
                throw std::runtime_error(
                    "DeepseekV4Attention: compressed KV cache cannot be rebuilt from bounded hidden history");
            }
            kv_comp_tensor = compressor_->forward_tensor(hidden_states, comp_batch, nb);
            kv_comp_tensor = rotate_compressed_blocks_(
                kv_comp_tensor, positions, total_len, 0);
            runtime_state_.set_compressed_kv(kv_comp_tensor, comp_batch, nb);
        }
    } else {
        if (hidden_states->shape()[1] != total_len) {
            throw std::runtime_error(
                "DeepseekV4Attention: compressed KV cache is unavailable for bounded hidden history");
        }
        kv_comp_tensor = compressor_->forward_tensor(hidden_states, comp_batch, nb);
        kv_comp_tensor = rotate_compressed_blocks_(
            kv_comp_tensor, positions, total_len, 0);
        if (!disable_compressed_kv_cache()) {
            runtime_state_.set_compressed_kv(kv_comp_tensor, comp_batch, nb);
        }
    }
    if (nb > 0 && indexer_) {
        indexed_blocks_tensor = indexer_->forward_tensor(hidden_states, q_residual, positions, index_top_k,
                                                         query_start, query_len, total_len,
                                                         raw_positions);
    }

    const bool contiguous_prefill = query_start == 0 && query_len == total_len
                                 && positions_are_contiguous(positions);
    const bool contiguous_positions = query_len == 1
                                        ? runtime_state_.positions_contiguous
                                        : positions_are_contiguous(positions);
    if (!contiguous_positions || (query_len != 1 && !contiguous_prefill)) {
        throw std::runtime_error("DeepseekV4Attention: compressed attention requires contiguous positions");
    }
    if (!kv_comp_tensor || nb == 0) {
        throw std::runtime_error("DeepseekV4Attention: compressed KV is unavailable");
    }
    const size_t compressed_key_count = index_top_k > 0 ? index_top_k : nb;
    const size_t active_key_count = compressed_key_count
                                  + (sliding_window_ == 0
                                         ? total_len
                                         : std::min(total_len, sliding_window_));
    if (active_key_count > 4096) {
        throw std::runtime_error("DeepseekV4Attention: compressed attention key count exceeds the GPU kernel limit");
    }

    size_t kv_start = 0;
    size_t kv_len = total_len;
    if (sliding_window_ > 0) {
        const bool contiguous_decode = !disable_contiguous_window_fastpath()
                                    && query_len == 1
                                    && query_start + 1 == total_len
                                    && runtime_state_.positions_contiguous;
        if (contiguous_decode) {
            kv_len = std::min(total_len, sliding_window_);
            kv_start = total_len - kv_len;
        } else {
            const int64_t pos_min = positions[query_start]
                                  - static_cast<int64_t>(sliding_window_);
            while (kv_start < total_len && positions[kv_start] <= pos_min) {
                ++kv_start;
            }
            kv_len = total_len - kv_start;
        }
    }

    const auto &rope_params = rotary_emb_.params();
    auto query_positions_tensor = position_tensor_for_query(
        raw_positions, positions, query_start, query_len, q_rope->device());
    infinicore::Tensor block_positions_tensor;
    const bool standard_contiguous_positions = positions.size() == total_len
                                            && !positions.empty()
                                            && positions.front() == 0
                                            && positions.back()
                                                   == static_cast<int64_t>(total_len - 1);
    if (!disable_precomputed_block_positions()
        && standard_contiguous_positions
        && block_position_table_ && block_position_table_->numel() >= nb) {
        block_positions_tensor = block_position_table_->narrow({{0, 0, nb}});
    } else if (runtime_state_.block_positions
               && runtime_state_.block_positions_blocks == nb) {
        block_positions_tensor = runtime_state_.block_positions;
    } else {
        std::vector<int64_t> block_positions(nb);
        for (size_t block = 0; block < nb; ++block) {
            const size_t block_token = std::min(block * compress_ratio, total_len > 0 ? total_len - 1 : 0);
            block_positions[block] = (positions[block_token] / static_cast<int64_t>(compress_ratio))
                                   * static_cast<int64_t>(compress_ratio);
        }
        block_positions_tensor = int64_vector_to_tensor(
            block_positions, {nb}, q_rope->device());
        runtime_state_.block_positions = block_positions_tensor;
        runtime_state_.block_positions_blocks = nb;
    }
    size_t gpu_index_top_k = 0;
    auto gpu_indexed_blocks_tensor = disable_cached_no_index_sentinel()
                                       ? int64_vector_to_tensor(
                                             std::vector<int64_t>{-1}, {1}, q_rope->device())
                                       : no_index_sentinel_;
    if (indexed_blocks_tensor && index_top_k > 0) {
        gpu_index_top_k = index_top_k;
        gpu_indexed_blocks_tensor = indexed_blocks_tensor->view({indexed_blocks_tensor->numel()})->contiguous();
    }
    auto window_key_states = key_states->narrow(
        {{1, kv_start, kv_len}})->contiguous();
    const int64_t window_position_base = positions.empty()
                                           ? 0
                                           : positions[kv_start];
    auto out = infinicore::op::deepseek_v4_compressed_decode(
        q_rope->contiguous(),
        window_key_states,
        kv_comp_tensor,
        attn_sink_->contiguous(),
        query_positions_tensor,
        block_positions_tensor,
        gpu_indexed_blocks_tensor,
        0,
        kv_len,
        contiguous_prefill,
        window,
        window_position_base,
        softmax_scale_,
        compress_ratio,
        gpu_index_top_k,
        rope_params.rope_dim,
        rope_params.rope_theta,
        rope_params.use_yarn,
        rope_params.yarn_factor,
        rope_params.yarn_beta_fast,
        rope_params.yarn_beta_slow,
        rope_params.yarn_original_seq_len,
        rope_params.yarn_extrapolation_factor);
    return out->view({batch_size, query_len, num_heads * head_dim});
}

infinicore::Tensor DeepseekV4Attention::sliding_attention_gpu_(const infinicore::Tensor &q_rope,
                                                               const infinicore::Tensor &key_states,
                                                               const std::vector<int64_t> &pos,
                                                               size_t query_start,
                                                               const infinicore::Tensor &raw_positions) const {
    const auto shape = q_rope->shape();
    const size_t batch_size = shape[0];
    const size_t query_len = shape[1];
    const size_t num_heads = shape[2];
    const size_t head_dim = shape[3];
    const size_t total_len = key_states->shape()[1];
    const size_t num_kv_heads = key_states->shape()[2];
    if (num_heads % num_kv_heads != 0) {
        throw std::runtime_error("DeepseekV4Attention: num_heads must be divisible by num_key_value_heads");
    }
    if (pos.size() < query_start + query_len) {
        throw std::runtime_error("DeepseekV4Attention: position_ids length mismatch");
    }
    if (q_rope->device().getType() == infinicore::Device::Type::CPU) {
        throw std::runtime_error("DeepseekV4Attention: sliding attention requires a GPU device");
    }
    const size_t window = sliding_window_ == 0 ? total_len : sliding_window_;

    const bool use_decode_kernel = query_len == 1 && query_start > 0;
    if (!use_decode_kernel) {
        infinicore::Tensor query_positions_tensor;
        infinicore::Tensor key_positions_tensor;
        if (!disable_swa_prefill_position_reuse()
            && query_start == 0 && query_len == total_len) {
            query_positions_tensor = position_tensor_for_query(
                raw_positions, pos, query_start, query_len, q_rope->device());
            key_positions_tensor = query_positions_tensor;
        } else {
            std::vector<int64_t> query_positions(query_len);
            for (size_t tq = 0; tq < query_len; ++tq) {
                query_positions[tq] = pos[query_start + tq];
            }
            query_positions_tensor = int64_vector_to_tensor(
                query_positions, {query_len}, q_rope->device());
            key_positions_tensor = int64_vector_to_tensor(
                pos, {total_len}, q_rope->device());
        }
        const auto &rope_params = rotary_emb_.params();
        auto out = infinicore::op::deepseek_v4_swa_prefill(
            q_rope->contiguous(),
            key_states->contiguous(),
            attn_sink_->contiguous(),
            query_positions_tensor,
            key_positions_tensor,
            softmax_scale_,
            window,
            rope_params.rope_dim,
            rope_params.rope_theta,
            rope_params.use_yarn,
            rope_params.yarn_factor,
            rope_params.yarn_beta_fast,
            rope_params.yarn_beta_slow,
            rope_params.yarn_original_seq_len,
            rope_params.yarn_extrapolation_factor);
        return out->view({batch_size, query_len, num_heads * head_dim});
    }

    size_t kv_start = 0;
    size_t kv_len = total_len;
    if (sliding_window_ > 0 && query_len == 1) {
        const bool contiguous_decode = !disable_contiguous_window_fastpath()
                                    && query_start + 1 == total_len
                                    && runtime_state_.positions_contiguous;
        if (contiguous_decode) {
            kv_len = std::min(total_len, sliding_window_);
            kv_start = total_len - kv_len;
        } else {
            const int64_t pos_min = pos[query_start]
                                  - static_cast<int64_t>(sliding_window_);
            while (kv_start < total_len && pos[kv_start] <= pos_min) {
                ++kv_start;
            }
            kv_len = total_len - kv_start;
        }
    }

    if (query_len == 1) {
        const auto &rope_params = rotary_emb_.params();
        auto positions_tensor = position_tensor_for_query(
            raw_positions, pos, query_start, query_len, q_rope->device());
        auto window_key_states = key_states->narrow(
            {{1, kv_start, kv_len}})->contiguous();
        auto out = infinicore::op::deepseek_v4_swa_decode(
            q_rope->contiguous(),
            window_key_states,
            attn_sink_->contiguous(),
            positions_tensor,
            0,
            kv_len,
            softmax_scale_,
            rope_params.rope_dim,
            rope_params.rope_theta,
            rope_params.use_yarn,
            rope_params.yarn_factor,
            rope_params.yarn_beta_fast,
            rope_params.yarn_beta_slow,
            rope_params.yarn_original_seq_len,
            rope_params.yarn_extrapolation_factor);
        return out->view({batch_size, query_len, num_heads * head_dim});
    }
    throw std::runtime_error("DeepseekV4Attention: unsupported sliding attention query shape");
}

infinicore::Tensor DeepseekV4Attention::attention_prefill_(const infinicore::Tensor &positions,
                                                           const infinicore::Tensor &q_rope,
                                                           const infinicore::Tensor &key_states,
                                                           const infinicore::Tensor &hidden_states,
                                                           const infinicore::Tensor &q_residual) const {
    const auto shape = q_rope->shape();
    const size_t seq_len = shape[1];
    const size_t compress_ratio = rotary_emb_.compress_ratio();
    auto pos = normalize_positions(positions, seq_len);

    if (has_no_visible_compressed_blocks(compress_ratio, pos, 0, seq_len)
        && q_rope->device().getType() != infinicore::Device::Type::CPU) {
        return sliding_attention_gpu_(q_rope, key_states, pos, 0, positions);
    }

    if (compressor_ && compress_ratio > 0
        && positions_are_contiguous(pos)
        && q_rope->device().getType() != infinicore::Device::Type::CPU) {
        return compressed_attention_gpu_(q_rope, key_states, hidden_states, q_residual, pos, 0, positions);
    }

    throw std::runtime_error(
        "DeepseekV4Attention: prefill requires a GPU device and contiguous positions");
}

infinicore::Tensor DeepseekV4Attention::apply_grouped_output_projection_(const infinicore::Tensor &attn_output) const {
    const auto shape = attn_output->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    auto grouped = attn_output->view({batch_size * seq_len, o_groups_, o_a_input_size_});

    const auto wo_a_weight = wo_a_->weight();
    std::vector<infinicore::Tensor> projected_groups;
    projected_groups.reserve(o_groups_);
    for (size_t group_idx = 0; group_idx < o_groups_; ++group_idx) {
        auto group_input = grouped->narrow({{1, group_idx, 1}})->squeeze(1)->contiguous();
        auto group_weight = wo_a_weight->narrow({{0, group_idx * o_lora_rank_, o_lora_rank_}})->contiguous();
        auto group_output = infinicore::op::linear(group_input, group_weight, std::nullopt);
        projected_groups.push_back(group_output->view({batch_size, seq_len, o_lora_rank_}));
    }

    auto projected = infinicore::op::cat(projected_groups, 2);

    auto final_output = wo_b_->forward(projected);

    return final_output;
}

} // namespace infinilm::models::deepseek_v4
