#include "ernie4_5_moe_vl_for_conditional_generation.hpp"

#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"
#include "infinicore/ops.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace infinilm::models::ernie4_5_moe_vl {

Ernie4_5_VLMoeForConditionalGeneration::Ernie4_5_VLMoeForConditionalGeneration(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device) {
    model_config_ = model_config;
    compute_device_ = device;

    // ERNIE-4.5-VL uses GPT-J / interleaved rotary (not the framework-default
    // GPT-NEOX half-split). The shared get_rope() reads the algorithm from the
    // config, so set it once here before the text model (and its RoPE) is built.
    model_config->set_rope_algo(infinicore::nn::RoPE::Algo::GPT_J);

    const auto &dtype{model_config->get_dtype()};
    size_t hidden_size = model_config->get<size_t>("hidden_size");
    size_t vocab_size = model_config->get<size_t>("vocab_size");

    im_patch_id_ = model_config->get_or<int64_t>("im_patch_id", 100295);
    image_start_token_id_ = model_config->get_or<int64_t>("image_start_token_id", 101304);
    image_end_token_id_ = model_config->get_or<int64_t>("image_end_token_id", 101305);
    video_start_token_id_ = model_config->get_or<int64_t>("video_start_token_id", 101306);
    video_end_token_id_ = model_config->get_or<int64_t>("video_end_token_id", 101307);

    INFINICORE_NN_MODULE_INIT(model, model_config, device);

    // Vision tower + adapter in one module registered as "visual" to match HF
    // checkpoint prefix. visual.merger.* holds the resampler weights.
    INFINICORE_NN_MODULE_INIT(visual, model_config, device);

    // tie_word_embeddings = true: weights are shared with model.embed_tokens and
    // copied in by the loader. The module is still created to own the logits proj.
    INFINICORE_NN_MODULE_INIT(lm_head, hidden_size, vocab_size, false, dtype, device);
}

infinicore::Tensor Ernie4_5_VLMoeForConditionalGeneration::derive_token_type_ids(
    const infinicore::Tensor &input_ids) const {
    // 0 = text, 1 = vision. Vision tokens are the image patch placeholder and any
    // token inside an image/video span.
    auto ids_cpu = input_ids->to(infinicore::Device::cpu());
    auto shape = ids_cpu->shape();
    size_t numel = ids_cpu->numel();
    const auto *ids = reinterpret_cast<const int64_t *>(ids_cpu->data());

    // Build on CPU and write EVERY element explicitly. Tensor::zeros(I64) is not
    // reliable here (it does not actually clear the buffer), so a zeros-then-set-1
    // construction leaves the text (0) entries full of garbage, which randomly
    // misclassifies text tokens as vision and routes them through the wrong expert
    // set -> non-deterministic output. Allocate with empty() and assign all slots.
    // Keep the result on CPU: the MoE dispatch is the only consumer and reads it on
    // the host; round-tripping this I64 tensor through MetaX device memory is both
    // wasteful and an additional corruption risk.
    auto token_type = infinicore::Tensor::empty(shape, infinicore::DataType::I64, infinicore::Device::cpu());
    auto *tt = reinterpret_cast<int64_t *>(token_type->data());
    // HF marks the entire contiguous span [START .. END] -- including the
    // IMAGE/VIDEO_START and _END marker tokens, not just the inner placeholders --
    // as the vision modality (token_type=1). Routing the markers through text
    // experts (placeholder-only classification) perturbs the MoE output and diverges
    // from HF. Track the span and mark start/end inclusive.
    bool in_vision = false;
    for (size_t i = 0; i < numel; ++i) {
        if (ids[i] == image_start_token_id_ || ids[i] == video_start_token_id_) {
            in_vision = true;
        }
        tt[i] = (in_vision || ids[i] == im_patch_id_) ? 1 : 0;
        if (ids[i] == image_end_token_id_ || ids[i] == video_end_token_id_) {
            in_vision = false;
        }
    }
    return token_type;
}

infinicore::Tensor Ernie4_5_VLMoeForConditionalGeneration::merge_vision_embeddings(
    const infinicore::Tensor &inputs_embeds,
    const infinicore::Tensor &vision_embeds,
    const infinicore::Tensor &input_ids) const {
    // Scatter vision_embeds[0..V) into inputs_embeds at the V positions where
    // input_ids == im_patch_id, preserving order. Mirrors minicpmv replace_embeddings
    // but keyed on the patch id rather than explicit bounds.
    auto out = infinicore::Tensor::empty(inputs_embeds->shape(), inputs_embeds->dtype(), inputs_embeds->device());
    out->copy_from(inputs_embeds);

    ASSERT_EQ(inputs_embeds->size(0), 1); // batch == 1 for prefill
    auto ids_cpu = input_ids->to(infinicore::Device::cpu());
    const auto *ids = reinterpret_cast<const int64_t *>(ids_cpu->data());
    size_t seq_len = ids_cpu->numel();

    auto out_slice = out->squeeze(0); // [seq, hidden]
    size_t v = 0;
    size_t vision_len = vision_embeds->size(0);
    for (size_t i = 0; i < seq_len && v < vision_len; ++i) {
        if (ids[i] == im_patch_id_) {
            auto patch = vision_embeds->narrow({{0, v, 1}});
            out_slice->narrow({{0, i, 1}})->copy_from(patch);
            ++v;
        }
    }
    return out;
}

InfinilmModel::Output Ernie4_5_VLMoeForConditionalGeneration::forward(const Input &input) const {
    if (!input.input_ids.has_value()) {
        throw std::runtime_error("Ernie4_5_VLMoeForConditionalGeneration: input_ids is required");
    }
    auto input_ids = input.input_ids.value();

    // Multimodal prefill: pixel_values present and processing a full prompt.
    if (input.pixel_values.has_value() && input_ids->size(1) > 1) {
        if (!input.tgt_sizes.has_value()) {
            throw std::runtime_error("Ernie4_5_VLMoeForConditionalGeneration: grid_thw (tgt_sizes) required for multimodal input");
        }
        auto pixel_values = input.pixel_values.value();
        auto grid_thw = input.tgt_sizes.value();

        // pixel_values arrives as a CPU tensor (built via from_numpy). Small images
        // work through the implicit host-mapped access, but a video's ~21.5MB buffer
        // can't be handled that way. Stage it to the module's compute device.
        // NOTE 1: use compute_device_, not nn::Module::device_ (default-constructed
        //   /garbage for this parent module -- moving to it crashes the GEMM).
        // NOTE 2: a single large pageable host->device copy hangs the MetaX runtime,
        //   so copy in row chunks to keep each transfer small.
        if (pixel_values->device() != compute_device_) {
            size_t rows = pixel_values->shape()[0];
            auto dev = infinicore::Tensor::empty(pixel_values->shape(), pixel_values->dtype(),
                                                 compute_device_);
            constexpr size_t H2D_ROWS = 1024;
            for (size_t s = 0; s < rows; s += H2D_ROWS) {
                size_t n = std::min(H2D_ROWS, rows - s);
                dev->narrow({{0, s, n}})->copy_from(pixel_values->narrow({{0, s, n}}));
            }
            pixel_values = dev;
        }

        // visual_ forward: ViT blocks -> merger -> [num_merged_tokens, text_hidden].
        auto vision_embeds = visual_->forward(pixel_values, grid_thw);

        auto inputs_embeds = model_->embed_tokens(input_ids);
        auto merged = merge_vision_embeddings(inputs_embeds, vision_embeds, input_ids);
        auto token_type_ids = derive_token_type_ids(input_ids);

        auto position_ids = input.position_ids.value();
        auto hidden_states = model_->forward_embeds(merged, position_ids, token_type_ids);
        auto logits = lm_head_->forward(hidden_states);
        return {logits};
    }

    // Text-only path (also the decode path for multimodal: vision already in cache).
    auto hidden_states = model_->forward(input);
    auto logits = lm_head_->forward(hidden_states);
    return {logits};
}

void Ernie4_5_VLMoeForConditionalGeneration::reset_cache(const cache::CacheConfig *cache_config) {
    if (nullptr == cache_config) {
        InfinilmModel::reset_cache(nullptr);
        return;
    }
    cache_config_ = cache_config->unique_copy();

    auto &kv_cache_vec = infinilm::global_state::get_forward_context().kv_cache_vec;
    kv_cache_vec.clear();
    const backends::AttentionBackend attention_backend = infinilm::global_state::get_infinilm_config().attention_backend;
    kv_cache_vec = std::move(default_allocate_kv_cache_tensors(cache_config, model_config_, attention_backend));
}

std::shared_ptr<infinilm::config::ModelConfig> create_ernie4_5_moe_vl_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("ernie4_5_moe_vl" != model_type) {
        throw std::runtime_error("create_ernie4_5_moe_vl_model_config: model_type is not ernie4_5_moe_vl");
    }

    nlohmann::json &config_json = model_config->get_config_json();
    if (!config_json.contains("head_dim")) {
        size_t head_dim = model_config->get<size_t>("hidden_size")
                        / model_config->get<size_t>("num_attention_heads");
        config_json["head_dim"] = head_dim; // 2560 / 20 = 128
    }
    return model_config;
}

} // namespace infinilm::models::ernie4_5_moe_vl

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    ernie4_5_moe_vl,
    infinilm::models::ernie4_5_moe_vl::Ernie4_5_VLMoeForConditionalGeneration,
    infinilm::models::ernie4_5_moe_vl::create_ernie4_5_moe_vl_model_config);
} // namespace
