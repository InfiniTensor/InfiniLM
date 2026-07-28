#include "kimi_k25_vision.hpp"

#include <infinicore/ops.hpp>
#include <infinicore/ops/mha.hpp>
#include <infinicore/ops/upsample_bilinear.hpp>

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace infinilm::models::kimi_k25 {
namespace {

std::vector<int64_t> grid_to_cpu(const infinicore::Tensor &grid_thw) {
    auto cpu = grid_thw->to(infinicore::Device::cpu());
    std::vector<int64_t> grid(cpu->numel());
    if (cpu->dtype() == infinicore::DataType::I64) {
        const auto *data = reinterpret_cast<const int64_t *>(cpu->data());
        return {data, data + cpu->numel()};
    }
    if (cpu->dtype() == infinicore::DataType::I32) {
        const auto *data = reinterpret_cast<const int32_t *>(cpu->data());
        for (size_t i = 0; i < cpu->numel(); ++i) {
            grid[i] = data[i];
        }
        return grid;
    }
    throw std::runtime_error("KimiK25VisionTower: grid_thw must be int32 or int64");
}

} // namespace

KimiK25VisionPatchProjection::KimiK25VisionPatchProjection(size_t hidden_size,
                                                           size_t patch_size,
                                                           const infinicore::DataType &dtype,
                                                           const infinicore::Device &device)
    : patch_size_(patch_size) {
    INFINICORE_NN_PARAMETER_INIT(weight, ({hidden_size, 3, patch_size, patch_size}, dtype, device));
    INFINICORE_NN_PARAMETER_INIT(bias, ({hidden_size}, dtype, device));
}

infinicore::Tensor KimiK25VisionPatchProjection::forward(const infinicore::Tensor &pixel_values) const {
    const size_t patch_dim = 3 * patch_size_ * patch_size_;
    auto pixels = pixel_values->view({pixel_values->numel() / patch_dim, patch_dim});
    return infinicore::op::linear(pixels, weight_->view({weight_->size(0), patch_dim}),
                                  std::make_optional<infinicore::Tensor>(bias_));
}

KimiK25VisionPosEmbed::KimiK25VisionPosEmbed(size_t height,
                                             size_t width,
                                             size_t hidden_size,
                                             const infinicore::DataType &dtype,
                                             const infinicore::Device &device)
    : height_(height), width_(width), hidden_size_(hidden_size) {
    INFINICORE_NN_PARAMETER_INIT(weight, ({height_, width_, hidden_size_}, dtype, device));
}

infinicore::Tensor KimiK25VisionPosEmbed::forward(size_t grid_h, size_t grid_w) const {
    if (grid_h == height_ && grid_w == width_) {
        return weight_->view({height_ * width_, hidden_size_});
    }
    // TODO(kimi_k25): Replace this structural fallback with bicubic interpolation
    // once InfiniCore provides it; the official MoonViT path uses bicubic resize.
    auto nchw = weight_->permute({2, 0, 1})->unsqueeze(0);
    return infinicore::op::upsample_bilinear(
               nchw,
               {static_cast<int64_t>(grid_h), static_cast<int64_t>(grid_w)},
               false)
        ->squeeze(0)
        ->permute({1, 2, 0})
        ->contiguous()
        ->view({grid_h * grid_w, hidden_size_});
}

KimiK25VisionPatchEmbed::KimiK25VisionPatchEmbed(const nlohmann::json &config,
                                                 const infinicore::DataType &dtype,
                                                 const infinicore::Device &device) {
    const size_t hidden_size = config.at("vt_hidden_size").get<size_t>();
    INFINICORE_NN_MODULE_INIT(proj, hidden_size, config.at("patch_size").get<size_t>(), dtype, device);
    INFINICORE_NN_MODULE_INIT(pos_emb,
                              config.at("init_pos_emb_height").get<size_t>(),
                              config.at("init_pos_emb_width").get<size_t>(),
                              hidden_size,
                              dtype,
                              device);
}

infinicore::Tensor KimiK25VisionPatchEmbed::forward(const infinicore::Tensor &pixel_values,
                                                    const infinicore::Tensor &grid_thw) const {
    const auto grid = grid_to_cpu(grid_thw);
    if (grid.size() != 3 || grid[0] != 1) {
        // TODO(kimi_k25): Add the fixed temporal positional embedding for video inputs.
        throw std::runtime_error("KimiK25VisionPatchEmbed: only single-frame image input is supported");
    }
    auto hidden_states = proj_->forward(pixel_values);
    return infinicore::op::add(hidden_states,
                               pos_emb_->forward(static_cast<size_t>(grid[1]), static_cast<size_t>(grid[2])));
}

KimiK25VisionMLP::KimiK25VisionMLP(size_t hidden_size,
                                   size_t intermediate_size,
                                   const infinicore::DataType &dtype,
                                   const infinicore::Device &device) {
    INFINICORE_NN_MODULE_INIT(fc0, hidden_size, intermediate_size, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(fc1, intermediate_size, hidden_size, true, dtype, device);
}

infinicore::Tensor KimiK25VisionMLP::forward(const infinicore::Tensor &hidden_states) const {
    auto mutable_hidden = hidden_states;
    auto intermediate = infinicore::op::gelu_tanh(fc0_->forward(mutable_hidden));
    return fc1_->forward(intermediate);
}

KimiK25VisionBlock::KimiK25VisionBlock(const nlohmann::json &config,
                                       std::shared_ptr<infinicore::nn::RoPE> rotary_emb,
                                       const infinicore::DataType &dtype,
                                       const infinicore::Device &device)
    : hidden_size_(config.at("vt_hidden_size").get<size_t>()),
      num_heads_(config.at("vt_num_attention_heads").get<size_t>()),
      head_dim_(hidden_size_ / num_heads_),
      scale_(1.0f / std::sqrt(static_cast<float>(head_dim_))),
      rotary_emb_(std::move(rotary_emb)) {
    INFINICORE_NN_MODULE_INIT(norm0, hidden_size_, 1e-5, dtype, device);
    INFINICORE_NN_MODULE_INIT(norm1, hidden_size_, 1e-5, dtype, device);
    INFINICORE_NN_MODULE_INIT(wqkv, hidden_size_, hidden_size_ * 3, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(wo, hidden_size_, hidden_size_, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(mlp, hidden_size_, config.at("vt_intermediate_size").get<size_t>(), dtype, device);
}

infinicore::Tensor KimiK25VisionBlock::forward(const infinicore::Tensor &hidden_states,
                                               const infinicore::Tensor &row_positions,
                                               const infinicore::Tensor &col_positions) const {
    const size_t seq_len = hidden_states->size(0);
    auto normalized = norm0_->forward(hidden_states);
    auto qkv = wqkv_->forward(normalized)->view({seq_len, 3, num_heads_, head_dim_});
    auto q = qkv->narrow({{1, 0, 1}})->squeeze(1)->contiguous();
    auto k = qkv->narrow({{1, 1, 1}})->squeeze(1)->contiguous();
    auto v = qkv->narrow({{1, 2, 1}})->squeeze(1)->contiguous();

    const size_t axis_dim = head_dim_ / 2;
    const size_t axis_pairs = axis_dim / 2;
    auto apply_2d_rope = [&](const infinicore::Tensor &x) {
        auto col = infinicore::Tensor::empty({seq_len, num_heads_, axis_dim}, x->dtype(), x->device());
        auto row = infinicore::Tensor::empty({seq_len, num_heads_, axis_dim}, x->dtype(), x->device());
        for (size_t pair = 0; pair < axis_pairs; ++pair) {
            col->narrow({{2, pair * 2, 2}})->copy_from(x->narrow({{2, pair * 4, 2}}));
            row->narrow({{2, pair * 2, 2}})->copy_from(x->narrow({{2, pair * 4 + 2, 2}}));
        }
        rotary_emb_->forward(col, col_positions, true);
        rotary_emb_->forward(row, row_positions, true);
        for (size_t pair = 0; pair < axis_pairs; ++pair) {
            x->narrow({{2, pair * 4, 2}})->copy_from(col->narrow({{2, pair * 2, 2}}));
            x->narrow({{2, pair * 4 + 2, 2}})->copy_from(row->narrow({{2, pair * 2, 2}}));
        }
    };
    apply_2d_rope(q);
    apply_2d_rope(k);

    auto attn_output = infinicore::op::mha(q->unsqueeze(0), k->unsqueeze(0), v->unsqueeze(0),
                                           std::nullopt, scale_, false)
                           ->view({seq_len, hidden_size_});
    auto attention_residual = infinicore::op::add(hidden_states, wo_->forward(attn_output));
    auto mlp_input = norm1_->forward(attention_residual);
    return infinicore::op::add(attention_residual, mlp_->forward(mlp_input));
}

KimiK25VisionEncoder::KimiK25VisionEncoder(const nlohmann::json &config,
                                           const infinicore::DataType &dtype,
                                           const infinicore::Device &device) {
    const size_t hidden_size = config.at("vt_hidden_size").get<size_t>();
    const size_t num_heads = config.at("vt_num_attention_heads").get<size_t>();
    const size_t head_dim = hidden_size / num_heads;
    auto rotary_emb = std::make_shared<infinicore::nn::RoPE>(
        head_dim / 2, head_dim / 2, 512, 10000.0,
        infinicore::nn::RoPE::Algo::GPT_J, dtype, device);
    const size_t num_layers = config.at("vt_num_hidden_layers").get<size_t>();
    blocks_.reserve(num_layers);
    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        blocks_.push_back(this->register_module<KimiK25VisionBlock>(
            "blocks." + std::to_string(layer_idx), config, rotary_emb, dtype, device));
    }
    INFINICORE_NN_MODULE_INIT(final_layernorm, hidden_size, 1e-5, dtype, device);
}

infinicore::Tensor KimiK25VisionEncoder::forward(const infinicore::Tensor &hidden_states,
                                                 const infinicore::Tensor &grid_thw) const {
    const auto grid = grid_to_cpu(grid_thw);
    if (grid.size() != 3 || grid[0] != 1) {
        throw std::runtime_error("KimiK25VisionEncoder: only one image grid is supported per call");
    }
    const size_t grid_h = static_cast<size_t>(grid[1]);
    const size_t grid_w = static_cast<size_t>(grid[2]);
    auto positions_cpu = infinicore::Tensor::empty({2, grid_h * grid_w}, infinicore::DataType::I64, infinicore::Device::cpu());
    auto *positions = reinterpret_cast<int64_t *>(positions_cpu->data());
    for (size_t row = 0; row < grid_h; ++row) {
        for (size_t col = 0; col < grid_w; ++col) {
            const size_t token = row * grid_w + col;
            positions[token] = static_cast<int64_t>(row);
            positions[grid_h * grid_w + token] = static_cast<int64_t>(col);
        }
    }
    auto positions_device = positions_cpu->to(hidden_states->device());
    auto row_positions = positions_device->narrow({{0, 0, 1}})->view({grid_h * grid_w});
    auto col_positions = positions_device->narrow({{0, 1, 1}})->view({grid_h * grid_w});

    auto output = hidden_states;
    for (const auto &block : blocks_) {
        output = block->forward(output, row_positions, col_positions);
    }
    return final_layernorm_->forward(output);
}

KimiK25VisionTower::KimiK25VisionTower(const nlohmann::json &config,
                                       const infinicore::DataType &dtype,
                                       const infinicore::Device &device)
    : hidden_size_(config.at("vt_hidden_size").get<size_t>()) {
    const auto &kernel = config.at("merge_kernel_size");
    merge_height_ = kernel.at(0).get<size_t>();
    merge_width_ = kernel.at(1).get<size_t>();
    INFINICORE_NN_MODULE_INIT(patch_embed, config, dtype, device);
    INFINICORE_NN_MODULE_INIT(encoder, config, dtype, device);
}

infinicore::Tensor KimiK25VisionTower::forward(const infinicore::Tensor &pixel_values,
                                               const infinicore::Tensor &grid_thw) const {
    const auto grid = grid_to_cpu(grid_thw);
    const size_t grid_h = static_cast<size_t>(grid.at(1));
    const size_t grid_w = static_cast<size_t>(grid.at(2));
    if (grid.at(0) != 1 || grid_h % merge_height_ != 0 || grid_w % merge_width_ != 0) {
        // TODO(kimi_k25): Implement temporal pooling for video grids with t > 1.
        throw std::runtime_error("KimiK25VisionTower: unsupported image/video merge grid");
    }
    auto hidden_states = encoder_->forward(patch_embed_->forward(pixel_values, grid_thw), grid_thw);
    return hidden_states
        ->view({grid_h / merge_height_, merge_height_, grid_w / merge_width_, merge_width_, hidden_size_})
        ->permute({0, 2, 1, 3, 4})
        ->contiguous()
        ->view({grid_h * grid_w / (merge_height_ * merge_width_), merge_height_ * merge_width_, hidden_size_});
}

KimiK25Projector::KimiK25Projector(const nlohmann::json &config,
                                   const infinicore::DataType &dtype,
                                   const infinicore::Device &device) {
    const size_t vision_hidden_size = config.at("mm_hidden_size").get<size_t>();
    const auto &kernel = config.at("merge_kernel_size");
    merged_hidden_size_ = vision_hidden_size * kernel.at(0).get<size_t>() * kernel.at(1).get<size_t>();
    const size_t text_hidden_size = config.at("text_hidden_size").get<size_t>();
    const double eps = config.at("projector_ln_eps").get<double>();
    INFINICORE_NN_MODULE_INIT(pre_norm, vision_hidden_size, eps, dtype, device);
    linear_0_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>(
        "proj.0", merged_hidden_size_, merged_hidden_size_, true, dtype, device);
    linear_2_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>(
        "proj.2", merged_hidden_size_, text_hidden_size, true, dtype, device);
}

infinicore::Tensor KimiK25Projector::forward(const infinicore::Tensor &vision_features) const {
    auto normalized = pre_norm_->forward(vision_features)
                          ->view({vision_features->size(0), merged_hidden_size_});
    auto hidden = infinicore::op::gelu(linear_0_->forward(normalized));
    return linear_2_->forward(hidden);
}

} // namespace infinilm::models::kimi_k25
