#include "ernie4_5_vision.hpp"

#include "../../utils.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/cat.hpp"
#include "infinicore/ops/gelu.hpp"
#include "infinicore/ops/gelutanh.hpp"
#include "infinicore/ops/matmul.hpp"
#include "infinicore/ops/mha.hpp"
#include "infinicore/ops/quickgelu.hpp"
#include "infinicore/ops/relu.hpp"
#include "infinicore/ops/softmax.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace infinilm::models::ernie4_5_moe_vl {
namespace {

constexpr float kImageMean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
constexpr float kImageStd[3] = {0.26862954f, 0.26130258f, 0.27577711f};
constexpr float kRescaleFactor = 0.00392156862745098f;
constexpr float kRopeTheta = 10000.0f;

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

void write_float(void *dst, size_t index, infinicore::DataType dtype, float value) {
    if (dtype == infinicore::DataType::F32) {
        reinterpret_cast<float *>(dst)[index] = value;
        return;
    }
    if (dtype == infinicore::DataType::BF16) {
        reinterpret_cast<uint16_t *>(dst)[index] = fp32_to_bf16(value);
        return;
    }
    throw std::runtime_error("ERNIE vision: only float32 and bfloat16 host conversion are supported");
}

float read_float(const void *src, size_t index, infinicore::DataType dtype) {
    if (dtype == infinicore::DataType::F32) {
        return reinterpret_cast<const float *>(src)[index];
    }
    if (dtype == infinicore::DataType::BF16) {
        return bf16_to_fp32(reinterpret_cast<const uint16_t *>(src)[index]);
    }
    throw std::runtime_error("ERNIE vision: only float32 and bfloat16 host conversion are supported");
}

std::vector<int64_t> read_grid(const infinicore::Tensor &grid_thw) {
    auto grid_cpu = grid_thw->to(infinicore::Device::cpu());
    ASSERT(grid_cpu->ndim() == 2);
    ASSERT(grid_cpu->shape()[1] == 3);
    const auto *grid_ptr = reinterpret_cast<const int64_t *>(grid_cpu->data());
    return std::vector<int64_t>(grid_ptr, grid_ptr + grid_cpu->numel());
}

std::vector<std::pair<int64_t, int64_t>> build_vision_pos_ids(const std::vector<int64_t> &grid,
                                                              size_t spatial_merge_size) {
    std::vector<std::pair<int64_t, int64_t>> pos_ids;
    for (size_t row = 0; row < grid.size(); row += 3) {
        const int64_t t = grid[row];
        const int64_t h = grid[row + 1];
        const int64_t w = grid[row + 2];
        ASSERT(h % static_cast<int64_t>(spatial_merge_size) == 0);
        ASSERT(w % static_cast<int64_t>(spatial_merge_size) == 0);

        for (int64_t ti = 0; ti < t; ++ti) {
            for (int64_t hb = 0; hb < h; hb += static_cast<int64_t>(spatial_merge_size)) {
                for (int64_t wb = 0; wb < w; wb += static_cast<int64_t>(spatial_merge_size)) {
                    for (int64_t ih = 0; ih < static_cast<int64_t>(spatial_merge_size); ++ih) {
                        for (int64_t iw = 0; iw < static_cast<int64_t>(spatial_merge_size); ++iw) {
                            pos_ids.emplace_back(hb + ih, wb + iw);
                        }
                    }
                }
            }
        }
    }
    return pos_ids;
}

std::vector<size_t> segment_lengths(const std::vector<int64_t> &grid) {
    std::vector<size_t> lengths;
    for (size_t row = 0; row < grid.size(); row += 3) {
        const int64_t t = grid[row];
        const int64_t h = grid[row + 1];
        const int64_t w = grid[row + 2];
        for (int64_t ti = 0; ti < t; ++ti) {
            lengths.push_back(static_cast<size_t>(h * w));
        }
    }
    return lengths;
}

} // namespace

Ernie4_5VisionPatchEmbed::Ernie4_5VisionPatchEmbed(const nlohmann::json &vision_config,
                                                   const infinicore::DataType &dtype,
                                                   const infinicore::Device &device) {
    patch_size_ = vision_config.value("patch_size", 14);
    in_channels_ = vision_config.value("in_channels", vision_config.value("in_chans", 3));
    embed_dim_ = vision_config.value("embed_dim", vision_config.value("hidden_size", 1280));

    INFINICORE_NN_MODULE_INIT(proj, in_channels_ * patch_size_ * patch_size_, embed_dim_, false, dtype, device);
}

infinicore::Tensor Ernie4_5VisionPatchEmbed::forward(const infinicore::Tensor &hidden_states) const {
    auto x = hidden_states;
    return proj_->forward(x);
}

Ernie4_5VisionAttention::Ernie4_5VisionAttention(const nlohmann::json &vision_config,
                                                 const infinicore::DataType &dtype,
                                                 const infinicore::Device &device) {
    embed_dim_ = vision_config.value("embed_dim", vision_config.value("hidden_size", 1280));
    num_heads_ = vision_config.value("num_heads", 16);
    spatial_merge_size_ = vision_config.value("spatial_merge_size", 2);
    if (embed_dim_ % num_heads_ != 0) {
        throw std::runtime_error("Ernie4_5VisionAttention: embed_dim must be divisible by num_heads");
    }
    head_dim_ = embed_dim_ / num_heads_;
    scale_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    INFINICORE_NN_MODULE_INIT(qkv, embed_dim_, embed_dim_ * 3, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(proj, embed_dim_, embed_dim_, true, dtype, device);
}

infinicore::Tensor Ernie4_5VisionAttention::apply_rotary_pos_emb_(const infinicore::Tensor &tensor,
                                                                  const infinicore::Tensor &grid_thw) const {
    ASSERT(tensor->ndim() == 3);
    ASSERT_EQ(tensor->shape()[1], num_heads_);
    ASSERT_EQ(tensor->shape()[2], head_dim_);
    ASSERT(head_dim_ % 4 == 0);

    const auto grid = read_grid(grid_thw);
    const auto pos_ids = build_vision_pos_ids(grid, spatial_merge_size_);
    ASSERT_EQ(pos_ids.size(), tensor->shape()[0]);

    auto src_cpu = tensor->to(infinicore::Device::cpu());
    auto out_cpu = infinicore::Tensor::empty(tensor->shape(), tensor->dtype(), infinicore::Device::cpu());
    const void *src = src_cpu->data();
    void *dst = out_cpu->data();

    const size_t seq_len = tensor->shape()[0];
    const size_t half_dim = head_dim_ / 2;
    const size_t freq_dim = head_dim_ / 4;
    std::vector<float> inv_freq(freq_dim);
    for (size_t i = 0; i < freq_dim; ++i) {
        inv_freq[i] = 1.0f / std::pow(kRopeTheta, static_cast<float>(i * 2) / static_cast<float>(half_dim));
    }

    for (size_t s = 0; s < seq_len; ++s) {
        for (size_t h = 0; h < num_heads_; ++h) {
            const size_t base = (s * num_heads_ + h) * head_dim_;
            for (size_t j = 0; j < half_dim; ++j) {
                const size_t freq_idx = j % freq_dim;
                const float pos = j < freq_dim
                                    ? static_cast<float>(pos_ids[s].first)
                                    : static_cast<float>(pos_ids[s].second);
                const float angle = pos * inv_freq[freq_idx];
                const float c = std::cos(angle);
                const float sn = std::sin(angle);
                const float x1 = read_float(src, base + j, tensor->dtype());
                const float x2 = read_float(src, base + half_dim + j, tensor->dtype());
                write_float(dst, base + j, tensor->dtype(), x1 * c - x2 * sn);
                write_float(dst, base + half_dim + j, tensor->dtype(), x2 * c + x1 * sn);
            }
        }
    }

    return out_cpu->to(tensor->device());
}

infinicore::Tensor Ernie4_5VisionAttention::segmented_attention_(const infinicore::Tensor &q,
                                                                 const infinicore::Tensor &k,
                                                                 const infinicore::Tensor &v,
                                                                 const infinicore::Tensor &grid_thw) const {
    const auto grid = read_grid(grid_thw);
    const auto lengths = segment_lengths(grid);
    std::vector<infinicore::Tensor> outputs;
    outputs.reserve(lengths.size());

    size_t offset = 0;
    for (size_t len : lengths) {
        auto q_flat = q->narrow({{0, offset, len}})
                          ->permute({1, 0, 2})
                          ->contiguous()
                          ->view({num_heads_, len, head_dim_});
        auto k_flat = k->narrow({{0, offset, len}})
                          ->permute({1, 0, 2})
                          ->contiguous()
                          ->view({num_heads_, len, head_dim_});
        auto v_flat = v->narrow({{0, offset, len}})
                          ->permute({1, 0, 2})
                          ->contiguous()
                          ->view({num_heads_, len, head_dim_});

        auto attn_weights = infinicore::op::matmul(q_flat, k_flat->permute({0, 2, 1}), scale_);
        infinicore::op::softmax_(attn_weights, attn_weights, -1);
        auto out = infinicore::op::matmul(attn_weights, v_flat)
                       ->view({num_heads_, len, head_dim_})
                       ->permute({1, 0, 2})
                       ->contiguous();
        outputs.push_back(out);
        offset += len;
    }
    ASSERT_EQ(offset, q->shape()[0]);

    if (outputs.size() == 1) {
        return outputs[0];
    }
    return infinicore::op::cat(outputs, 0);
}

infinicore::Tensor Ernie4_5VisionAttention::forward(const infinicore::Tensor &hidden_states,
                                                    const infinicore::Tensor &grid_thw) const {
    auto x = hidden_states;
    auto qkv_states = qkv_->forward(x)->view({hidden_states->shape()[0], 3, num_heads_, head_dim_});
    auto q = qkv_states->narrow({{1, 0, 1}})->squeeze(1)->contiguous();
    auto k = qkv_states->narrow({{1, 1, 1}})->squeeze(1)->contiguous();
    auto v = qkv_states->narrow({{1, 2, 1}})->squeeze(1)->contiguous();

    q = apply_rotary_pos_emb_(q, grid_thw);
    k = apply_rotary_pos_emb_(k, grid_thw);

    auto attn_output = segmented_attention_(q, k, v, grid_thw)
                           ->contiguous()
                           ->view({hidden_states->shape()[0], embed_dim_});
    return proj_->forward(attn_output);
}

Ernie4_5VisionMLP::Ernie4_5VisionMLP(const nlohmann::json &vision_config,
                                     const infinicore::DataType &dtype,
                                     const infinicore::Device &device)
    : hidden_act_(vision_config.value("hidden_act", "quick_gelu")) {
    const size_t dim = vision_config.value("embed_dim", vision_config.value("hidden_size", 1280));
    const double mlp_ratio = vision_config.value("mlp_ratio", 4.0);
    const size_t hidden_dim = static_cast<size_t>(static_cast<double>(dim) * mlp_ratio);
    INFINICORE_NN_MODULE_INIT(fc1, dim, hidden_dim, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(fc2, hidden_dim, dim, true, dtype, device);
}

infinicore::Tensor Ernie4_5VisionMLP::forward(const infinicore::Tensor &hidden_states) const {
    auto x_in = hidden_states;
    auto x = fc1_->forward(x_in);
    if (hidden_act_ == "quick_gelu") {
        x = infinicore::op::quick_gelu(x);
    } else if (hidden_act_ == "gelu") {
        x = infinicore::op::gelu(x);
    } else if (hidden_act_ == "gelu_tanh") {
        x = infinicore::op::gelu_tanh(x);
    } else if (hidden_act_ == "relu") {
        x = infinicore::op::relu(x);
    } else {
        throw std::runtime_error("Ernie4_5VisionMLP: unsupported activation " + hidden_act_);
    }
    return fc2_->forward(x);
}

Ernie4_5VisionBlock::Ernie4_5VisionBlock(const nlohmann::json &vision_config,
                                         const infinicore::DataType &dtype,
                                         const infinicore::Device &device) {
    const size_t hidden_size = vision_config.value("embed_dim", vision_config.value("hidden_size", 1280));
    INFINICORE_NN_MODULE_INIT(norm1, hidden_size, 1e-6, dtype, device);
    INFINICORE_NN_MODULE_INIT(attn, vision_config, dtype, device);
    INFINICORE_NN_MODULE_INIT(norm2, hidden_size, 1e-6, dtype, device);
    INFINICORE_NN_MODULE_INIT(mlp, vision_config, dtype, device);
}

infinicore::Tensor Ernie4_5VisionBlock::forward(const infinicore::Tensor &hidden_states,
                                                const infinicore::Tensor &grid_thw) const {
    auto residual = hidden_states;
    auto x = norm1_->forward(hidden_states);
    x = attn_->forward(x, grid_thw);
    x = infinicore::op::add(x, residual);

    residual = x;
    x = norm2_->forward(x);
    x = mlp_->forward(x);
    return infinicore::op::add(x, residual);
}

Ernie4_5VisionModel::Ernie4_5VisionModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                         const infinicore::Device &device) {
    vision_config_ = model_config->get_config_json().value("vision_config", nlohmann::json::object());
    dtype_ = model_config->get_dtype();
    depth_ = vision_config_.value("depth", 32);
    hidden_size_ = vision_config_.value("hidden_size", vision_config_.value("embed_dim", 1280));
    patch_size_ = vision_config_.value("patch_size", 14);

    INFINICORE_NN_MODULE_INIT(patch_embed, vision_config_, dtype_, device);
    blocks_.reserve(depth_);
    for (size_t i = 0; i < depth_; ++i) {
        blocks_.push_back(this->register_module<Ernie4_5VisionBlock>("blocks." + std::to_string(i), vision_config_, dtype_, device));
    }
    INFINICORE_NN_MODULE_INIT(ln, hidden_size_, 1e-6, dtype_, device);
}

infinicore::Tensor Ernie4_5VisionModel::normalize_images_(const infinicore::Tensor &images) const {
    if (images->dtype() != infinicore::DataType::U8 && images->dtype() != infinicore::DataType::BYTE) {
        return images;
    }

    ASSERT(images->ndim() == 2);
    const size_t patch_width = images->shape()[1];
    const size_t channel_patch = patch_size_ * patch_size_;
    ASSERT_EQ(patch_width, channel_patch * 3);

    auto images_cpu = images->to(infinicore::Device::cpu());
    auto out_cpu = infinicore::Tensor::empty(images->shape(), dtype_, infinicore::Device::cpu());
    const auto *src = reinterpret_cast<const uint8_t *>(images_cpu->data());
    void *dst = out_cpu->data();

    for (size_t i = 0; i < images->shape()[0]; ++i) {
        for (size_t j = 0; j < patch_width; ++j) {
            const size_t channel = j / channel_patch;
            const float value = (static_cast<float>(src[i * patch_width + j]) * kRescaleFactor - kImageMean[channel]) / kImageStd[channel];
            write_float(dst, i * patch_width + j, dtype_, value);
        }
    }
    return out_cpu->to(images->device());
}

infinicore::Tensor Ernie4_5VisionModel::vision_grid_thw_(const infinicore::Tensor &grid_thw) const {
    const auto grid = read_grid(grid_thw);
    size_t rows = 0;
    for (size_t i = 0; i < grid.size(); i += 3) {
        rows += static_cast<size_t>(grid[i]);
    }

    auto out_cpu = infinicore::Tensor::empty({rows, 3}, infinicore::DataType::I64, infinicore::Device::cpu());
    auto *out = reinterpret_cast<int64_t *>(out_cpu->data());
    size_t row = 0;
    for (size_t i = 0; i < grid.size(); i += 3) {
        const int64_t t = grid[i];
        const int64_t h = grid[i + 1];
        const int64_t w = grid[i + 2];
        for (int64_t ti = 0; ti < t; ++ti) {
            out[row * 3] = 1;
            out[row * 3 + 1] = h;
            out[row * 3 + 2] = w;
            ++row;
        }
    }
    return out_cpu->to(grid_thw->device());
}

infinicore::Tensor Ernie4_5VisionModel::forward(const infinicore::Tensor &images,
                                                const infinicore::Tensor &grid_thw) const {
    auto vision_grid = vision_grid_thw_(grid_thw);
    auto hidden_states = normalize_images_(images);
    hidden_states = patch_embed_->forward(hidden_states);

    for (const auto &block : blocks_) {
        hidden_states = block->forward(hidden_states, vision_grid);
    }
    return ln_->forward(hidden_states);
}

} // namespace infinilm::models::ernie4_5_moe_vl
