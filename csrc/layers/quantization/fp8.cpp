#include "fp8.hpp"
#include "none_quantization.hpp"

#include <infinicore/ops/add.hpp>
#include <infinicore/ops/block_fp8_linear.hpp>
#include <infinicore/ops/linear.hpp>

#include <spdlog/spdlog.h>
#include <stdexcept>

namespace infinilm::quantization {

std::vector<ParamDescriptor> FP8Quantization::get_param_layout(
    size_t in_features, size_t out_features,
    int split_dim, int tp_rank, int tp_size,
    int /*tp_num_heads*/,
    const infinicore::DataType &dtype,
    bool bias) const {

    std::vector<ParamDescriptor> descs;

    // Weight: FP8 (E4M3) format - keep as F8, do NOT convert to BF16
    descs.push_back({"weight", {out_features, in_features},
                     infinicore::DataType::F8, split_dim, tp_rank, tp_size});

    // Per-block weight scale (inverse): BF16, shape = [ceil(N/128), ceil(K/128)]
    size_t num_out_blocks = (out_features + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t num_in_blocks = (in_features + BLOCK_SIZE - 1) / BLOCK_SIZE;
    descs.push_back({"weight_scale_inv", {num_out_blocks, num_in_blocks},
                     infinicore::DataType::F32, split_dim, tp_rank, tp_size});

    if (bias) {
        descs.push_back({"bias", {out_features}, dtype,
                         split_dim >= 0 ? 0 : -1,
                         split_dim >= 0 ? tp_rank : 0,
                         split_dim >= 0 ? tp_size : 1});
    }
    return descs;
}

infinicore::Tensor FP8Quantization::forward(
    const ParamsMap &params,
    const infinicore::Tensor &input,
    bool has_bias,
    float /*alpha*/) const {

    auto weight_it = params.find("weight");
    auto scale_it = params.find("weight_scale_inv");
    auto bias_it = params.find("bias");

    if (weight_it == params.end()) {
        throw std::runtime_error("FP8Quantization::forward: weight not found");
    }
    if (scale_it == params.end()) {
        throw std::runtime_error("FP8Quantization::forward: weight_scale_inv not found");
    }

    auto weight = weight_it->second;
    auto scale = scale_it->second;

    // Ensure input, weight, and scale are contiguous
    // (split_params creates narrow views that may not be contiguous)
    auto x = input->is_contiguous() ? input : input->contiguous();
    auto w = weight->is_contiguous() ? weight : weight->contiguous();
    auto s = scale->is_contiguous() ? scale : scale->contiguous();

    // Get dimensions
    auto x_shape = x->shape();
    size_t ndim = x_shape.size();
    size_t K = x_shape[ndim - 1];  // last dim is always feature dim
    // M = product of all leading dims
    size_t M = 1;
    for (size_t i = 0; i < ndim - 1; i++) {
        M *= x_shape[i];
    }
    auto w_shape = w->shape();
    size_t N = w_shape[0];

    // Flatten input to 2D [M, K] and ensure contiguous
    auto flat = x->view({M, K});
    flat = flat->is_contiguous() ? flat : flat->contiguous();

    // Allocate output [M, N]
    auto out = infinicore::Tensor::empty(
        {M, N}, input->dtype(), input->device());

    // Call block-FP8 linear: BF16 input x F8 weight + block scale -> BF16 output
    infinicore::op::block_fp8_linear_(
        out, flat, w, s);

    if (has_bias && bias_it != params.end()) {
        auto bias = bias_it->second;
        auto bias_broadcast = bias->view({1, N});
        infinicore::op::add_(out, out, bias_broadcast);
    }

    // Reshape output to match input's leading dims with N
    std::vector<size_t> out_shape(x_shape.begin(), x_shape.end() - 1);
    out_shape.push_back(N);
    return out->view(out_shape);
}

std::vector<SplitParam> FP8Quantization::split_params(
    const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
    const std::vector<SplitInfo> &splits,
    int narrow_dim,
    int tp_rank, int tp_size, int /*tp_num_heads*/) const {

    std::vector<SplitParam> result;
    auto weight_it = params.find("weight");
    auto scale_it = params.find("weight_scale_inv");
    auto bias_it = params.find("bias");

    for (const auto &s : splits) {
        result.push_back({s.prefix + ".weight",
                          infinicore::nn::Parameter(
                              weight_it->second->narrow({{static_cast<size_t>(narrow_dim), s.start, s.size}}),
                              narrow_dim, tp_rank, tp_size, s.num_shards)});

        if (scale_it != params.end()) {
            size_t scale_start = s.start / BLOCK_SIZE;
            size_t scale_size = (s.size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            result.push_back({s.prefix + ".weight_scale_inv",
                              infinicore::nn::Parameter(
                                  scale_it->second->narrow({{static_cast<size_t>(narrow_dim), scale_start, scale_size}}),
                                  narrow_dim, tp_rank, tp_size, s.num_shards)});
        }

        if (bias_it != params.end()) {
            result.push_back({s.prefix + ".bias",
                              infinicore::nn::Parameter(
                                  bias_it->second->narrow({{0, s.start, s.size}}),
                                  0, tp_rank, tp_size, s.num_shards)});
        }
    }
    return result;
}

std::shared_ptr<BaseQuantization> FP8Quantization::process_weights_after_loading(
    ParamsMap &params,
    const infinicore::Device &device,
    int /*split_dim*/) const {

    auto weight_it = params.find("weight");
    auto scale_it = params.find("weight_scale_inv");

    if (weight_it == params.end()) {
        return nullptr;
    }
    if (scale_it == params.end()) {
        spdlog::debug("FP8: no weight_scale_inv found, skipping");
        return nullptr;
    }

    auto weight = weight_it->second;
    auto scale = scale_it->second;

    size_t out_features = weight->shape()[0];
    size_t in_features = weight->shape()[1];

    size_t num_out_blocks = (out_features + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t num_in_blocks = (in_features + BLOCK_SIZE - 1) / BLOCK_SIZE;

    auto scale_shape = scale->shape();
    if (scale_shape.size() != 2 ||
        scale_shape[0] != num_out_blocks ||
        scale_shape[1] != num_in_blocks) {
        throw std::runtime_error("FP8Quantization: weight_scale_inv shape mismatch");
    }

    // Keep weight as FP8 (1 byte/element), just ensure contiguous
    params["weight"] = weight->contiguous();

    // Scale is already FP32 (converted on Python side during loading)
    params["weight_scale_inv"] = scale->contiguous();

    spdlog::debug("FP8: kept weight as F8, scale cast to F32, shape=[{}, {}]",
                  out_features, in_features);

    // Return nullptr to continue using FP8Quantization (not NoneQuantization)
    return nullptr;
}

} // namespace infinilm::quantization
