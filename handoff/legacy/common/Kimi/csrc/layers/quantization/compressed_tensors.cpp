#include "compressed_tensors.hpp"
#include "runtime/runtime.hpp"
#include "execution/submit.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <optional>

namespace infinilm::quantization {

std::vector<ParamDescriptor> CompressedTensors::get_param_layout(
    size_t in_features, size_t out_features,
    int split_dim, int tp_rank, int tp_size,
    int /*tp_num_heads*/,
    const infinilm::runtime::DataType &dtype,
    bool bias) const {

    std::vector<ParamDescriptor> descs;
    descs.push_back({"weight", {out_features, in_features}, infinilm::runtime::DataType::kInt8, split_dim, tp_rank, tp_size});
    // weight_scale is per-output-channel [out_features, 1]; always split on
    // dim0 (output dimension) for ColumnParallel, and don't split for RowParallel.
    int scale_split_dim = (split_dim == 0) ? 0 : -1;
    int scale_tp_size = (split_dim == 0) ? tp_size : 1;
    int scale_tp_rank = (split_dim == 0) ? tp_rank : 0;
    descs.push_back({"weight_scale", {out_features, 1}, infinilm::runtime::DataType::kFloat32, scale_split_dim, scale_tp_rank, scale_tp_size});
    if (bias) {
        descs.push_back({"bias", {out_features}, dtype, -1, 0, 1});
    }
    return descs;
}

infinilm::runtime::Tensor CompressedTensors::forward(
    infinilm::runtime::Runtime &runtime,
    const ParamsMap &params,
    const infinilm::runtime::Tensor &input,
    bool has_bias,
    float alpha) const {
    runtime.assertOwnerThread();
    auto weight = params.at("weight");
    auto weight_scale = params.at("weight_scale");

    std::optional<infinilm::runtime::Tensor> bias_opt;
    if (has_bias) {
        bias_opt = params.at("bias");
    }

    for (const auto &tensor : {input, weight, weight_scale}) {
        if (!tensor || tensor->device() != runtime.device())
            throw std::invalid_argument("W8A8 tensors must belong to the supplied Runtime");
    }
    if (bias_opt && (!*bias_opt || (*bias_opt)->device() != runtime.device()))
        throw std::invalid_argument("W8A8 bias must belong to the supplied Runtime");
    infinilm::runtime::Tensor input_contiguous;
    auto effective_weight_scale = weight_scale;
    infinilm::runtime::Tensor output, input_packed, input_scale;
    try {
        input_contiguous = input->contiguous(runtime);
    if (std::fabs(alpha - 1.0f) > 1e-7f) {
            effective_weight_scale = infinilm::runtime::Tensor::empty(runtime, weight_scale->shape(), weight_scale->dtype());
#ifdef INFINICORE_MUL_SCALAR_OPS_UNAVAILABLE
            throw std::runtime_error("MulScalar requires an InfiniOps MulScalar provider");
#else
            infinilm::execution::submit<infini::ops::MulScalar>(
                runtime, infinilm::execution::RecordingPolicy::AllowHostReplay,
                infini::ops::MulScalar::DefaultConfig(runtime.device().type()),
                weight_scale, static_cast<double>(alpha), effective_weight_scale);
#endif
            INFINILM_TRACE_OP(MulScalar, effective_weight_scale);
    }
        if (input_contiguous->ndim() == 0 || weight->ndim() != 2 ||
            input_contiguous->size(input_contiguous->ndim() - 1) != weight->size(1) ||
            weight->size(1) == 0)
            throw std::invalid_argument("W8A8 input and weight geometry differ");
        const auto out_features = weight->size(0);
        const auto in_features = weight->size(1);
        auto output_shape = input_contiguous->shape();
        output_shape.back() = out_features;
        output = infinilm::runtime::Tensor::empty(runtime, output_shape, input_contiguous->dtype());

        const auto &shape = input_contiguous->shape();
        const auto last = shape.end() - 1;
        infinilm::runtime::Size batch = 1;
        if (std::find(shape.begin(), last, infinilm::runtime::Size{0}) != last) {
            batch = 0;
        } else {
            for (auto dim = shape.begin(); dim != last; ++dim) {
                if (batch > std::numeric_limits<infinilm::runtime::Size>::max() / *dim)
                    throw std::length_error("W8A8 batch size overflows");
                batch *= *dim;
            }
        }
        input_packed = infinilm::runtime::Tensor::empty(runtime,
            {batch, in_features}, infinilm::runtime::DataType::kInt8);
        input_scale = infinilm::runtime::Tensor::empty(runtime,
            {batch, 1}, infinilm::runtime::DataType::kFloat32);
        auto input_view = input_contiguous->view({batch, in_features});
        auto output_view = output->view({batch, out_features});
        auto weight_view = weight->permute({1, 0});
        if (bias_opt)
            bias_opt = std::make_optional(bias_opt.value()->as_strided({batch, out_features}, {0, 1}));
        try {
            infinilm::execution::submit<infini::ops::PerChannelQuantI8>(
                runtime, infinilm::execution::RecordingPolicy::AllowHostReplay,
                infini::ops::PerChannelQuantI8::DefaultConfig(runtime.device().type()),
                input_view, input_packed, input_scale);
            INFINILM_TRACE_OP(PerChannelQuantI8, input_view);
            infinilm::execution::submit<infini::ops::I8Gemm>(
                runtime, infinilm::execution::RecordingPolicy::AllowHostReplay,
                infini::ops::I8Gemm::DefaultConfig(runtime.device().type()),
                output_view, input_packed, input_scale, weight_view, effective_weight_scale, bias_opt);
            INFINILM_TRACE_OP(I8Gemm, output_view);
        } catch (...) {
            // Keep submitted operands alive through the original failure-only drain.
            const auto failure = std::current_exception();
            try { runtime.syncStream(); } catch (...) { std::terminate(); }
            std::rethrow_exception(failure);
        }
        return output;
    } catch (...) {
        // Glue temporaries remain owned while previously submitted work drains.
        const auto failure = std::current_exception();
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        std::rethrow_exception(failure);
    }
}

std::vector<SplitParam> CompressedTensors::split_params(
    const std::unordered_map<std::string, infinilm::weights::Weight> &params,
    const std::vector<SplitInfo> &splits,
    int narrow_dim,
    int tp_rank, int tp_size, int /*tp_num_heads*/) const {

    std::vector<SplitParam> result;
    auto weight_it = params.find("weight");
    auto scale_it = params.find("weight_scale");
    auto bias_it = params.find("bias");

    for (const auto &s : splits) {
        result.push_back({s.prefix + ".weight",
                          infinilm::weights::Weight(
                              weight_it->second->narrow({{static_cast<size_t>(narrow_dim), s.start, s.size}}),
                              narrow_dim, tp_rank, tp_size, s.num_shards)});
        result.push_back({s.prefix + ".weight_scale",
                          infinilm::weights::Weight(
                              scale_it->second->narrow({{static_cast<size_t>(narrow_dim), s.start, s.size}}),
                              narrow_dim, tp_rank, tp_size, s.num_shards)});
        if (bias_it != params.end()) {
            result.push_back({s.prefix + ".bias",
                              infinilm::weights::Weight(
                                  bias_it->second->narrow({{0, s.start, s.size}}),
                                  0, tp_rank, tp_size, s.num_shards)});
        }
    }
    return result;
}

} // namespace infinilm::quantization
