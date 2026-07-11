#pragma once

#include "ernie4_5_moe_vl_text_model.hpp"
#include "ernie4_5_moe_vl_vision.hpp"

#include "../infinilm_model.hpp"
#include "../../layers/linear/linear.hpp"

namespace infinilm::models::ernie4_5_moe_vl {

// Top-level: Ernie4_5_VLMoeForConditionalGeneration.
// Assembles the text MoE backbone + vision tower + lm_head.
// Module layout matches HF checkpoint prefixes:
//   model.*       - text MoE backbone
//   visual.*      - vision tower (visual.blocks.*, visual.patch_embed.*)
//   visual.merger.* - spatial/temporal adapter (inside vision tower)
//   lm_head.*     - output projection (weights tied to model.embed_tokens)
class Ernie4_5_VLMoeForConditionalGeneration : public InfinilmModel {
public:
    Ernie4_5_VLMoeForConditionalGeneration(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                           const infinicore::Device &device);

    Output forward(const Input &input) const override;

    void reset_cache(const cache::CacheConfig *cache_config) override;

private:
    // Replace im_patch_id placeholder positions in inputs_embeds with vision_embeds.
    infinicore::Tensor merge_vision_embeddings(const infinicore::Tensor &inputs_embeds,
                                               const infinicore::Tensor &vision_embeds,
                                               const infinicore::Tensor &input_ids) const;

    // Derive per-token modality ids (0=text, 1=vision) from input_ids, using the
    // image/video token ids from config. Avoids adding token_type_ids to the
    // framework-level Input struct.
    infinicore::Tensor derive_token_type_ids(const infinicore::Tensor &input_ids) const;

    // The module's compute device. nn::Module::device_ is default-constructed and
    // never set for parent modules (only leaf weights get the device via their
    // constructors), so we stash the ctor's device here to stage pixel_values onto
    // the GPU in forward().
    infinicore::Device compute_device_;

    int64_t im_patch_id_{0};
    int64_t image_start_token_id_{0};
    int64_t image_end_token_id_{0};
    int64_t video_start_token_id_{0};
    int64_t video_end_token_id_{0};

    INFINICORE_NN_MODULE(Ernie4_5_VLMoeModel, model);
    // Registered as "visual" to match HF checkpoint prefix visual.*
    INFINICORE_NN_MODULE(Ernie4_5_VisionTransformer, visual);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);
};

std::shared_ptr<infinilm::config::ModelConfig> create_ernie4_5_moe_vl_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::ernie4_5_moe_vl
