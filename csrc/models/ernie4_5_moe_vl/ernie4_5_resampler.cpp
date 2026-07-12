#include "ernie4_5_resampler.hpp"

#include "../../utils.hpp"
#include "infinicore/ops.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace infinilm::models::ernie4_5_moe_vl {

Ernie4_5Resampler::Ernie4_5Resampler(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                     const infinicore::Device &device) {
    const auto &dtype{model_config->get_dtype()};
    in_dim_ = model_config->get_or<size_t>("pixel_hidden_size", 1280);
    out_dim_ = model_config->get<size_t>("hidden_size");
    spatial_conv_size_ = model_config->get_or<size_t>("spatial_conv_size", 2);
    temporal_conv_size_ = model_config->get_or<size_t>("temporal_conv_size", 2);
    use_temporal_conv_ = model_config->get_or<bool>("use_temporal_conv", true);

    spatial_dim_ = in_dim_ * spatial_conv_size_ * spatial_conv_size_;
    temporal_dim_ = spatial_dim_ * temporal_conv_size_;

    spatial_linear_0_ = this->register_module<infinilm::nn::Linear>("spatial_linear.0", spatial_dim_, spatial_dim_, true, dtype, device);
    spatial_linear_2_ = this->register_module<infinilm::nn::Linear>("spatial_linear.2", spatial_dim_, spatial_dim_, true, dtype, device);
    spatial_linear_3_ = this->register_module<infinicore::nn::LayerNorm>("spatial_linear.3", spatial_dim_, 1e-6, dtype, device);
    temporal_linear_0_ = this->register_module<infinilm::nn::Linear>("temporal_linear.0", temporal_dim_, spatial_dim_, true, dtype, device);
    temporal_linear_2_ = this->register_module<infinilm::nn::Linear>("temporal_linear.2", spatial_dim_, spatial_dim_, true, dtype, device);
    temporal_linear_3_ = this->register_module<infinicore::nn::LayerNorm>("temporal_linear.3", spatial_dim_, 1e-6, dtype, device);
    mlp_ = this->register_module<infinilm::nn::Linear>("mlp", spatial_dim_, out_dim_, true, dtype, device);
    after_norm_ = this->register_module<infinicore::nn::RMSNorm>("after_norm", out_dim_, model_config->get<double>("rms_norm_eps"), dtype, device);
}

infinicore::Tensor Ernie4_5Resampler::temporal_placeholder_(const infinicore::Tensor &x,
                                                            const infinicore::Tensor &grid_thw) const {
    ASSERT(x->ndim() == 2);
    ASSERT(x->shape()[1] == spatial_dim_);
    ASSERT(temporal_conv_size_ == 2);

    auto grid_cpu = grid_thw->to(infinicore::Device::cpu());
    ASSERT(grid_cpu->ndim() == 2);
    ASSERT(grid_cpu->shape()[1] == 3);
    auto *grid_ptr = reinterpret_cast<const int64_t *>(grid_cpu->data());
    const size_t n_grid = grid_cpu->shape()[0];

    size_t total_rows = 0;
    for (size_t i = 0; i < n_grid; ++i) {
        const size_t t = static_cast<size_t>(grid_ptr[i * 3]);
        const size_t h = static_cast<size_t>(grid_ptr[i * 3 + 1]);
        const size_t w = static_cast<size_t>(grid_ptr[i * 3 + 2]);
        ASSERT(h % spatial_conv_size_ == 0);
        ASSERT(w % spatial_conv_size_ == 0);
        const size_t spatial_size = (h * w) / (spatial_conv_size_ * spatial_conv_size_);
        total_rows += ((t + 1) / 2) * spatial_size;
    }

    auto out = infinicore::Tensor::empty({total_rows, temporal_dim_}, x->dtype(), x->device());

    size_t input_base = 0;
    size_t out_row = 0;
    for (size_t i = 0; i < n_grid; ++i) {
        const size_t t = static_cast<size_t>(grid_ptr[i * 3]);
        const size_t h = static_cast<size_t>(grid_ptr[i * 3 + 1]);
        const size_t w = static_cast<size_t>(grid_ptr[i * 3 + 2]);
        const size_t spatial_size = (h * w) / (spatial_conv_size_ * spatial_conv_size_);

        for (size_t temp_offset = 0; temp_offset < t; temp_offset += 2) {
            const size_t temp_offset2 = (t == 1) ? 0 : std::min(temp_offset + 1, t - 1);
            for (size_t row = 0; row < spatial_size; ++row) {
                const size_t src1 = input_base + temp_offset * spatial_size + row;
                const size_t src2 = input_base + temp_offset2 * spatial_size + row;
                auto dst_first = out->narrow({{0, out_row, 1}, {1, 0, spatial_dim_}});
                auto dst_second = out->narrow({{0, out_row, 1}, {1, spatial_dim_, spatial_dim_}});
                dst_first->copy_from(x->narrow({{0, src1, 1}}));
                dst_second->copy_from(x->narrow({{0, src2, 1}}));
                ++out_row;
            }
        }
        input_base += t * spatial_size;
    }

    ASSERT_EQ(out_row, total_rows);
    ASSERT_EQ(input_base, x->shape()[0]);
    return out;
}

infinicore::Tensor Ernie4_5Resampler::forward(const infinicore::Tensor &vision_features,
                                              const infinicore::Tensor &grid_thw) const {
    ASSERT(vision_features->ndim() == 2);
    ASSERT(vision_features->shape()[1] == in_dim_);
    const size_t merge = spatial_conv_size_ * spatial_conv_size_;
    ASSERT(vision_features->shape()[0] % merge == 0);

    auto x = vision_features->view({vision_features->shape()[0] / merge, spatial_dim_});
    x = spatial_linear_0_->forward(x);
    x = infinicore::op::gelu(x);
    x = spatial_linear_2_->forward(x);
    x = spatial_linear_3_->forward(x);

    if (use_temporal_conv_) {
        x = temporal_placeholder_(x, grid_thw);
        x = temporal_linear_0_->forward(x);
        x = infinicore::op::gelu(x);
        x = temporal_linear_2_->forward(x);
        x = temporal_linear_3_->forward(x);
    }

    x = mlp_->forward(x);
    x = after_norm_->forward(x);
    return x;
}

} // namespace infinilm::models::ernie4_5_moe_vl
