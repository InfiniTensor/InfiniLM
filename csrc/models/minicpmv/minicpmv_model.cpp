#include "minicpmv_model.hpp"

#include "infinicore/ops.hpp"

#include <algorithm>
#include <stdexcept>

namespace infinilm::models::minicpmv {

MiniCPMVModel::MiniCPMVModel(const MiniCPMVConfig &config,
                             const infinicore::Device &device,
                             engine::distributed::RankInfo rank_info)
    : config_(config), rank_info_(rank_info) {
    INFINICORE_NN_MODULE_INIT(llm, config.llm_config, device, rank_info);

    auto vision_cfg = config.vision_config;
    if (config.drop_vision_last_layer && vision_cfg.num_hidden_layers > 0) {
        vision_cfg.num_hidden_layers -= 1;
    }
    INFINICORE_NN_MODULE_INIT(vpm, vision_cfg, config.dtype, device, false);

    size_t embed_dim = config.llm_config.hidden_size;
    size_t num_heads = embed_dim / 128;
    INFINICORE_NN_MODULE_INIT(resampler, config.query_num, embed_dim, num_heads, vision_cfg.hidden_size, config.dtype, device);
}

infinicore::Tensor MiniCPMVModel::replace_embeddings(const infinicore::Tensor &inputs_embeds,
                                                     const infinicore::Tensor &vision_hidden,
                                                     const infinicore::Tensor &image_bound) const {
    auto out = infinicore::Tensor::empty(inputs_embeds->shape(), inputs_embeds->dtype(), inputs_embeds->device());
    out->copy_from(inputs_embeds);

    auto bounds_cpu = image_bound->to(infinicore::Device::cpu());
    auto batch_size = bounds_cpu->size(0);
    auto num_ranges = bounds_cpu->size(1);
    auto *bound_ptr = reinterpret_cast<const int64_t *>(bounds_cpu->data());

    for (size_t b = 0; b < batch_size; ++b) {
        auto vision_slice = vision_hidden->narrow({{0, b, 1}});
        auto vision_len = vision_slice->size(1);
        size_t offset = 0;
        for (size_t r = 0; r < num_ranges; ++r) {
            auto start = bound_ptr[(b * num_ranges + r) * 2];
            auto end = bound_ptr[(b * num_ranges + r) * 2 + 1];
            if (end <= start) {
                continue;
            }
            size_t len = static_cast<size_t>(end - start);
            if (offset + len > vision_len) {
                throw std::runtime_error("MiniCPMVModel: image_bound length exceeds vision tokens");
            }
            auto src = vision_slice->narrow({{1, offset, len}});
            if (src->shape().size() == 2) {
                src = src->unsqueeze(0);
            }
            auto dst = out->narrow({{0, b, 1}, {1, static_cast<size_t>(start), len}});
            dst->copy_from(src);
            offset += len;
        }
    }

    return out;
}

InfinilmModel::Output MiniCPMVModel::forward(const InfinilmModel::Input &input) const {
    if (!input.input_ids.has_value()) {
        throw std::runtime_error("MiniCPMVModel: input_ids is required");
    }
    auto input_ids = input.input_ids.value();

    if (input.pixel_values.has_value() && input_ids->size(1) > 1) {
        if (!input.image_bound.has_value()) {
            throw std::runtime_error("MiniCPMVModel: image_bound required for multimodal input");
        }
        auto pixel_values = input.pixel_values.value();
        auto vision_embedding = vpm_->forward(pixel_values, input.tgt_sizes);
        auto vision_hidden = resampler_->forward(vision_embedding, input.tgt_sizes);

        auto inputs_embeds = llm_->model().embed_tokens(input_ids);
        auto merged_embeds = replace_embeddings(inputs_embeds, vision_hidden, input.image_bound.value());

        infinicore::Tensor position_ids;
        if (input.position_ids.has_value()) {
            position_ids = input.position_ids.value();
        } else {
            auto batch = merged_embeds->size(0);
            auto seq_len = merged_embeds->size(1);
            auto pos_cpu = infinicore::Tensor::zeros({batch, seq_len}, infinicore::DataType::I64, infinicore::Device::cpu());
            auto *pos_ptr = reinterpret_cast<int64_t *>(pos_cpu->data());
            for (size_t b = 0; b < batch; ++b) {
                for (size_t i = 0; i < seq_len; ++i) {
                    pos_ptr[b * seq_len + i] = static_cast<int64_t>(i);
                }
            }
            position_ids = pos_cpu->to(merged_embeds->device());
        }

        auto hidden_states = llm_->model().forward_embeds(
            merged_embeds,
            position_ids,
            input.past_sequence_lengths,
            input.total_sequence_lengths,
            input.input_offsets,
            input.block_tables,
            input.slot_mapping);

        auto logits = llm_->logits_from_hidden(hidden_states);
        return {logits};
    }

    return llm_->forward(input);
}

void MiniCPMVModel::reset_cache(const cache::CacheConfig *cache_config) {
    llm_->reset_cache(cache_config);
}

uint32_t MiniCPMVModel::compress_kv_cache_inplace(uint32_t seq_len,
                                                  size_t batch_size,
                                                  const cache::KVCompressionConfig &cfg) {
    return llm_->compress_kv_cache_inplace(seq_len, batch_size, cfg);
}

} // namespace infinilm::models::minicpmv
