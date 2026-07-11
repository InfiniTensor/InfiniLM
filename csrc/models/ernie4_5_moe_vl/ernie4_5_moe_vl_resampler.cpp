#include "ernie4_5_moe_vl_resampler.hpp"

#include "../../utils.hpp"
#include "infinicore/ops.hpp"

#include <cstdint>
#include <vector>

namespace infinilm::models::ernie4_5_moe_vl {

Ernie4_5_VLResampler::Ernie4_5_VLResampler(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                           const infinicore::Device &device) {
    const auto &dtype{model_config->get_dtype()};

    pixel_hidden_size_ = model_config->get<size_t>("pixel_hidden_size");      // 1280
    text_hidden_size_ = model_config->get<size_t>("hidden_size");             // 2560
    spatial_conv_size_ = model_config->get_or<size_t>("spatial_conv_size", 2);
    temporal_conv_size_ = model_config->get_or<size_t>("temporal_conv_size", 2);

    size_t spatial_dim = pixel_hidden_size_ * spatial_conv_size_ * spatial_conv_size_;  // 5120
    size_t temporal_dim = spatial_dim * temporal_conv_size_;                            // 10240
    double layer_norm_eps = 1e-6;   // spatial/temporal Sequential LayerNorms
    double after_norm_eps = 1e-5;   // HF after_norm RMSNorm uses eps=1e-5

    // spatial_linear: Linear(5120,5120) -> act -> Linear(5120,5120) -> LayerNorm(5120)
    spatial_linear_0_ = this->register_module<infinicore::nn::Linear>(
        "spatial_linear.0", spatial_dim, spatial_dim, true, dtype, device);
    spatial_linear_2_ = this->register_module<infinicore::nn::Linear>(
        "spatial_linear.2", spatial_dim, spatial_dim, true, dtype, device);
    spatial_linear_3_ = this->register_module<infinicore::nn::LayerNorm>(
        "spatial_linear.3", spatial_dim, layer_norm_eps, dtype, device);

    // temporal_linear: Linear(10240,5120) -> act -> Linear(5120,5120) -> LayerNorm(5120)
    temporal_linear_0_ = this->register_module<infinicore::nn::Linear>(
        "temporal_linear.0", temporal_dim, spatial_dim, true, dtype, device);
    temporal_linear_2_ = this->register_module<infinicore::nn::Linear>(
        "temporal_linear.2", spatial_dim, spatial_dim, true, dtype, device);
    temporal_linear_3_ = this->register_module<infinicore::nn::LayerNorm>(
        "temporal_linear.3", spatial_dim, layer_norm_eps, dtype, device);

    INFINICORE_NN_MODULE_INIT(mlp, spatial_dim, text_hidden_size_, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(after_norm, text_hidden_size_, after_norm_eps, dtype, device);
}

infinicore::Tensor Ernie4_5_VLResampler::forward(const infinicore::Tensor &x,
                                                 const infinicore::Tensor &grid_thw) const {
    // Input:  x [num_patches, pixel_hidden_size]
    // Output: [num_merged_tokens, text_hidden_size]
    ASSERT(x->ndim() == 2);
    size_t num_patches = x->shape()[0];
    size_t spatial_block = spatial_conv_size_ * spatial_conv_size_;
    size_t spatial_dim = pixel_hidden_size_ * spatial_block;

    ASSERT_EQ(num_patches % spatial_block, 0);
    size_t num_spatial_merged = num_patches / spatial_block;

    auto merged_spatial = x->view({num_spatial_merged, spatial_dim});

    // spatial_linear Sequential: Linear -> act -> Linear -> LayerNorm
    // TODO(ernie-vl): activation type (gelu vs quick_gelu) is a guess; verify.
    auto h = spatial_linear_0_->forward(merged_spatial);
    h = infinicore::op::gelu(h);
    h = spatial_linear_2_->forward(h);
    h = spatial_linear_3_->forward(h);
    auto x_spatial = h;  // [num_spatial_merged, spatial_dim]

    // Temporal merge. ERNIE applies temporal_linear for BOTH images and video
    // (use_temporal_conv=true) -- HF fwd_placeholder gathers two interleaved
    // "timesteps" of spatial tokens and concatenates them on the feature axis to
    // [n_out, 2*spatial_dim], then temporal_linear projects back to spatial_dim.
    // timestep_1 = frames 0,2,4,...; timestep_2 = frames 1,3,5,... (for t==1 both
    // are the single frame, i.e. the spatial features are duplicated). Indices are
    // computed per media from grid_thw on CPU.
    auto thw_cpu = grid_thw->to(infinicore::Device::cpu())->contiguous();
    const auto *g = reinterpret_cast<const int64_t *>(thw_cpu->data());
    size_t num_media = grid_thw->shape()[0];

    std::vector<size_t> idx1;
    std::vector<size_t> idx2;
    size_t base = 0;
    for (size_t i = 0; i < num_media; ++i) {
        int64_t t = g[i * 3 + 0];
        int64_t hh = g[i * 3 + 1];
        int64_t ww = g[i * 3 + 2];
        size_t ss = static_cast<size_t>(hh * ww) / spatial_block;  // spatial tokens / frame
        for (int64_t to = 0; to < t; to += 2) {
            for (size_t s = 0; s < ss; ++s) {
                idx1.push_back(base + static_cast<size_t>(to) * ss + s);
            }
        }
        for (int64_t to = (t > 1 ? 1 : 0); to < t; to += 2) {
            for (size_t s = 0; s < ss; ++s) {
                idx2.push_back(base + static_cast<size_t>(to) * ss + s);
            }
        }
        base += static_cast<size_t>(t) * ss;
    }
    ASSERT_EQ(idx1.size(), idx2.size());
    size_t n_out = idx1.size();
    size_t temporal_dim = spatial_dim * temporal_conv_size_;  // 10240

    auto x_temporal_in = infinicore::Tensor::empty(
        {n_out, temporal_dim}, x_spatial->dtype(), x_spatial->device());
    for (size_t r = 0; r < n_out; ++r) {
        auto dst = x_temporal_in->narrow({{0, r, 1}});  // [1, temporal_dim]
        dst->narrow({{1, 0, spatial_dim}})->copy_from(x_spatial->narrow({{0, idx1[r], 1}}));
        dst->narrow({{1, spatial_dim, spatial_dim}})->copy_from(x_spatial->narrow({{0, idx2[r], 1}}));
    }

    // temporal_linear Sequential: Linear -> gelu -> Linear -> LayerNorm
    auto tt = temporal_linear_0_->forward(x_temporal_in);
    tt = infinicore::op::gelu(tt);
    tt = temporal_linear_2_->forward(tt);
    tt = temporal_linear_3_->forward(tt);

    auto projected = mlp_->forward(tt);  // -> [num_merged_tokens, text_hidden_size]
    return after_norm_->forward(projected);  // RMSNorm (HF uses RMSNorm here)
}

} // namespace infinilm::models::ernie4_5_moe_vl
