#pragma once

#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"
#include "qwen3_vl_position.hpp"
#include "qwen3_vl_vision_layers.hpp"

#include <nlohmann/json.hpp>

namespace infinilm::models::qwen3_vl {

class Qwen3VLVisionModel : public infinicore::nn::Module {
public:
    Qwen3VLVisionModel(const nlohmann::json &config,
                       const infinicore::DataType &dtype,
                       const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &pixel_values,
                               const infinicore::Tensor &image_grid_thw) const;

private:
    size_t hidden_size_;
    size_t spatial_merge_size_;
    size_t num_grid_per_side_;
    Qwen3VLPositionBuilder position_builder_;

    INFINICORE_NN_MODULE(Qwen3VLPatchEmbed, patch_embed);
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, pos_embed);
    INFINICORE_NN_MODULE_VEC(Qwen3VLVisionBlock, blocks);
    INFINICORE_NN_MODULE(Qwen3VLPatchMerger, merger);
    INFINICORE_NN_MODULE_VEC(Qwen3VLPatchMerger, deepstack_merger_list);
};

} // namespace infinilm::models::qwen3_vl
