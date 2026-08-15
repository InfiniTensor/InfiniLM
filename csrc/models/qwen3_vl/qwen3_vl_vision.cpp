#include "qwen3_vl_vision.hpp"

#include "../../config/model_config.hpp"
#include "infinicore/ops.hpp"

#include <cmath>

namespace infinilm::models::qwen3_vl {

using infinilm::config::json_size;

Qwen3VLVisionModel::Qwen3VLVisionModel(const nlohmann::json &config,
                                       const infinicore::DataType &dtype,
                                       const infinicore::Device &device)
    : hidden_size_(json_size(config, "hidden_size")),
      spatial_merge_size_(json_size(config, "spatial_merge_size", 2)),
      num_grid_per_side_(static_cast<size_t>(std::sqrt(static_cast<double>(json_size(config, "num_position_embeddings"))))),
      position_builder_(hidden_size_,
                        spatial_merge_size_,
                        num_grid_per_side_,
                        json_size(config, "num_heads"),
                        dtype,
                        device) {
    INFINICORE_NN_MODULE_INIT(patch_embed, config, dtype, device);
    INFINICORE_NN_MODULE_INIT(pos_embed, json_size(config, "num_position_embeddings"), hidden_size_, std::nullopt, dtype, device);
    INFINICORE_NN_MODULE_VEC_INIT(blocks, json_size(config, "depth"), Qwen3VLVisionBlock, config, dtype, device);
    INFINICORE_NN_MODULE_INIT(merger, config, false, dtype, device);
    size_t deepstack_count = config.contains("deepstack_visual_indexes") ? config["deepstack_visual_indexes"].size() : 0;
    INFINICORE_NN_MODULE_VEC_INIT(deepstack_merger_list, deepstack_count, Qwen3VLPatchMerger, config, true, dtype, device);
}

infinicore::Tensor Qwen3VLVisionModel::forward(const infinicore::Tensor &pixel_values,
                                               const infinicore::Tensor &image_grid_thw) const {
    auto hidden_states = patch_embed_->forward(pixel_values);
    hidden_states = infinicore::op::add(hidden_states, position_builder_.position_embeddings(image_grid_thw, *pos_embed_));
    auto [rotary_pos_ids, sin_table, cos_table] = position_builder_.rotary_embeddings(image_grid_thw);
    for (const auto &block : blocks_) {
        hidden_states = block->forward(hidden_states, rotary_pos_ids, sin_table, cos_table);
    }
    return merger_->forward(hidden_states);
}

} // namespace infinilm::models::qwen3_vl
