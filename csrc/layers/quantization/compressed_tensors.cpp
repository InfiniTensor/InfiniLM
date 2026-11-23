#include "compressed_tensors.hpp"
#include "execution/submit.hpp"
#include "runtime/runtime.hpp"

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
        const bool column_parallel = split_dim == 0;
        descs.push_back({"bias", {out_features}, dtype, 0, column_parallel ? tp_rank : 0, column_parallel ? tp_size : 1});
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
        if (!tensor || tensor->device() != runtime.device()) {
            throw std::invalid_argument("W8A8 tensors must belong to the supplied Runtime");
        }
    }
    if (bias_opt && (!*bias_opt || (*bias_opt)->device() != runtime.device())) {
        throw std::invalid_argument("W8A8 bias must belong to the supplied Runtime");
    }
    infinilm::runtime::Tensor input_contiguous;
    auto effective_weight_scale = weight_scale;
    infinilm::runtime::Tensor output, input_packed, input_scale;
    try {
        input_contiguous = input->contiguous(runtime);
        if (std::fabs(alpha - 1.0f) > 1e-7f) {
            effective_weight_scale = infinilm::runtime::Tensor::empty(runtime, weight_scale->shape(), weight_scale->dtype());
            infinilm::execution::submit<infini::ops::Mul>(
                runtime, infinilm::execution::RecordingPolicy::AllowHostReplay,
                infini::ops::Mul::DefaultConfig(runtime.device().type()),
                weight_scale, static_cast<double>(alpha), effective_weight_scale);
            INFINILM_TRACE_OP(Mul, effective_weight_scale);
        }
        const bool nz_weight = weight_prepacked_ && weight->ndim() == 4;
        const auto in_features = weight_prepacked_
                                       ? (nz_weight ? weight->size(1) * 16
                                                   : weight->size(0))
                                      : weight->size(1);
        const auto out_features = weight_prepacked_
                                       ? (nz_weight ? weight->size(0) * 32
                                                   : weight->size(1))
                                       : weight->size(0);
        if (input_contiguous->ndim() == 0 ||
            (weight->ndim() != 2 && !nz_weight) ||
            input_contiguous->size(input_contiguous->ndim() - 1) != in_features ||
            in_features == 0) {
            throw std::invalid_argument("W8A8 input and weight geometry differ");
        }
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
                if (batch > std::numeric_limits<infinilm::runtime::Size>::max() / *dim) {
                    throw std::length_error("W8A8 batch size overflows");
                }
                batch *= *dim;
            }
        }
        input_packed = infinilm::runtime::Tensor::empty(runtime,
                                                        {batch, in_features}, infinilm::runtime::DataType::kInt8);
        input_scale = infinilm::runtime::Tensor::empty(runtime,
                                                       {batch, 1}, infinilm::runtime::DataType::kFloat32);
        auto input_view = input_contiguous->view({batch, in_features});
        auto output_view = output->view({batch, out_features});
        auto weight_view = weight_prepacked_
                                  ? weight
                                  : weight->permute({1, 0});
        if (bias_opt) {
            bias_opt = std::make_optional(bias_opt.value()->as_strided({batch, out_features}, {0, 1}));
        }
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
            try {
                runtime.syncStream();
            } catch (...) { std::terminate(); }
            std::rethrow_exception(failure);
        }
        return output;
    } catch (...) {
        // Glue temporaries remain owned while previously submitted work drains.
        const auto failure = std::current_exception();
        try {
            runtime.syncStream();
        } catch (...) { std::terminate(); }
        std::rethrow_exception(failure);
    }
}

std::pair<infinilm::runtime::Tensor, infinilm::runtime::Tensor>
CompressedTensors::forward_gate_up(
    infinilm::runtime::Runtime &runtime,
    const ParamsMap &gate_params,
    const ParamsMap &up_params,
    const infinilm::runtime::Tensor &input,
    bool gate_has_bias,
    bool up_has_bias,
    float gate_alpha,
    float up_alpha) const {
    runtime.assertOwnerThread();
    const auto gate_weight = gate_params.at("weight");
    const auto gate_weight_scale = gate_params.at("weight_scale");
    const auto up_weight = up_params.at("weight");
    const auto up_weight_scale = up_params.at("weight_scale");
    std::optional<infinilm::runtime::Tensor> gate_bias_opt;
    std::optional<infinilm::runtime::Tensor> up_bias_opt;
    if (gate_has_bias) {
        gate_bias_opt = gate_params.at("bias");
    }
    if (up_has_bias) {
        up_bias_opt = up_params.at("bias");
    }
    for (const auto &tensor : {input, gate_weight, gate_weight_scale,
                                up_weight, up_weight_scale}) {
        if (!tensor || tensor->device() != runtime.device()) {
            throw std::invalid_argument("W8A8 tensors must belong to the supplied Runtime");
        }
    }
    if ((gate_bias_opt && (!*gate_bias_opt || (*gate_bias_opt)->device() != runtime.device())) ||
        (up_bias_opt && (!*up_bias_opt || (*up_bias_opt)->device() != runtime.device()))) {
        throw std::invalid_argument("W8A8 bias must belong to the supplied Runtime");
    }

    infinilm::runtime::Tensor input_contiguous;
    infinilm::runtime::Tensor gate_output, up_output, input_packed, input_scale;
    auto effective_gate_scale = gate_weight_scale;
    auto effective_up_scale = up_weight_scale;
    try {
        input_contiguous = input->contiguous(runtime);
        if (std::fabs(gate_alpha - 1.0f) > 1e-7f) {
            effective_gate_scale = infinilm::runtime::Tensor::empty(
                runtime, gate_weight_scale->shape(), gate_weight_scale->dtype());
            infinilm::execution::submit<infini::ops::Mul>(
                runtime, infinilm::execution::RecordingPolicy::AllowHostReplay,
                infini::ops::Mul::DefaultConfig(runtime.device().type()),
                gate_weight_scale, static_cast<double>(gate_alpha),
                effective_gate_scale);
            INFINILM_TRACE_OP(Mul, effective_gate_scale);
        }
        if (std::fabs(up_alpha - 1.0f) > 1e-7f) {
            effective_up_scale = infinilm::runtime::Tensor::empty(
                runtime, up_weight_scale->shape(), up_weight_scale->dtype());
            infinilm::execution::submit<infini::ops::Mul>(
                runtime, infinilm::execution::RecordingPolicy::AllowHostReplay,
                infini::ops::Mul::DefaultConfig(runtime.device().type()),
                up_weight_scale, static_cast<double>(up_alpha),
                effective_up_scale);
            INFINILM_TRACE_OP(Mul, effective_up_scale);
        }

        const auto geometry = [&](const infinilm::runtime::Tensor &weight) {
            const bool nz_weight = weight_prepacked_ && weight->ndim() == 4;
            if (!weight_prepacked_) {
                return std::pair(weight->shape()[1], weight->shape()[0]);
            }
            if (nz_weight) {
                return std::pair(weight->shape()[1] * 16,
                                 weight->shape()[0] * 32);
            }
            return std::pair(weight->shape()[0], weight->shape()[1]);
        };
        const auto [gate_in_features, gate_out_features] = geometry(gate_weight);
        const auto [up_in_features, up_out_features] = geometry(up_weight);
        if (input_contiguous->ndim() == 0 ||
            (gate_weight->ndim() != 2 &&
             !(weight_prepacked_ && gate_weight->ndim() == 4)) ||
            (up_weight->ndim() != 2 &&
             !(weight_prepacked_ && up_weight->ndim() == 4)) ||
            input_contiguous->size(input_contiguous->ndim() - 1) != gate_in_features ||
            gate_in_features != up_in_features || gate_in_features == 0) {
            throw std::invalid_argument("W8A8 gate/up input and weight geometries differ");
        }
        const auto in_features = gate_in_features;

        auto gate_output_shape = input_contiguous->shape();
        gate_output_shape.back() = gate_out_features;
        gate_output = infinilm::runtime::Tensor::empty(
            runtime, gate_output_shape, input_contiguous->dtype());
        auto up_output_shape = input_contiguous->shape();
        up_output_shape.back() = up_out_features;
        up_output = infinilm::runtime::Tensor::empty(
            runtime, up_output_shape, input_contiguous->dtype());

        const auto &shape = input_contiguous->shape();
        const auto last = shape.end() - 1;
        infinilm::runtime::Size batch = 1;
        if (std::find(shape.begin(), last, infinilm::runtime::Size{0}) != last) {
            batch = 0;
        } else {
            for (auto dim = shape.begin(); dim != last; ++dim) {
                if (batch > std::numeric_limits<infinilm::runtime::Size>::max() / *dim) {
                    throw std::length_error("W8A8 batch size overflows");
                }
                batch *= *dim;
            }
        }
        input_packed = infinilm::runtime::Tensor::empty(
            runtime, {batch, in_features}, infinilm::runtime::DataType::kInt8);
        input_scale = infinilm::runtime::Tensor::empty(
            runtime, {batch, 1}, infinilm::runtime::DataType::kFloat32);
        auto up_input_packed = infinilm::runtime::Tensor::empty(
            runtime, input_packed->shape(), input_packed->dtype());
        auto up_input_scale = infinilm::runtime::Tensor::empty(
            runtime, input_scale->shape(), input_scale->dtype());
        auto input_view = input_contiguous->view({batch, in_features});
        auto gate_output_view = gate_output->view({batch, gate_out_features});
        auto up_output_view = up_output->view({batch, up_out_features});
        auto gate_weight_view = weight_prepacked_
                                      ? gate_weight
                                      : gate_weight->permute({1, 0});
        auto up_weight_view = weight_prepacked_
                                    ? up_weight
                                    : up_weight->permute({1, 0});
        if (gate_bias_opt) {
            gate_bias_opt = std::make_optional(gate_bias_opt.value()->as_strided(
                {batch, gate_out_features}, {0, 1}));
        }
        if (up_bias_opt) {
            up_bias_opt = std::make_optional(up_bias_opt.value()->as_strided(
                {batch, up_out_features}, {0, 1}));
        }

        infinilm::execution::submit<infini::ops::PerChannelQuantI8>(
            runtime, infinilm::execution::RecordingPolicy::AllowHostReplay,
            infini::ops::PerChannelQuantI8::DefaultConfig(runtime.device().type()),
            input_view, input_packed, input_scale);
        INFINILM_TRACE_OP(PerChannelQuantI8, input_view);
        infinilm::execution::submit<infini::ops::Copy>(
            runtime, infinilm::execution::RecordingPolicy::AllowHostReplay,
            infini::ops::Copy::DefaultConfig(runtime.device().type()),
            input_packed, false, up_input_packed);
        infinilm::execution::submit<infini::ops::Copy>(
            runtime, infinilm::execution::RecordingPolicy::AllowHostReplay,
            infini::ops::Copy::DefaultConfig(runtime.device().type()),
            input_scale, false, up_input_scale);
        infinilm::execution::submit<infini::ops::I8Gemm>(
            runtime, infinilm::execution::RecordingPolicy::AllowHostReplay,
            infini::ops::I8Gemm::DefaultConfig(runtime.device().type()),
            gate_output_view, input_packed, input_scale,
            gate_weight_view, effective_gate_scale, gate_bias_opt);
        INFINILM_TRACE_OP(I8Gemm, gate_output_view);
        infinilm::execution::submit<infini::ops::I8Gemm>(
            runtime, infinilm::execution::RecordingPolicy::AllowHostReplay,
            infini::ops::I8Gemm::DefaultConfig(runtime.device().type()),
            up_output_view, up_input_packed, up_input_scale,
            up_weight_view, effective_up_scale, up_bias_opt);
        INFINILM_TRACE_OP(I8Gemm, up_output_view);
    } catch (...) {
        const auto failure = std::current_exception();
        try {
            runtime.syncStream();
        } catch (...) {
            std::terminate();
        }
        std::rethrow_exception(failure);
    }
    return {gate_output, up_output};
}

infinilm::runtime::Tensor CompressedTensors::forward_quantized(
    infinilm::runtime::Runtime &runtime,
    const ParamsMap &params,
    const infinilm::runtime::Tensor &packed,
    const infinilm::runtime::Tensor &scale,
    bool has_bias,
    float alpha,
    const infinilm::runtime::DataType &output_dtype) const {
    runtime.assertOwnerThread();
    const auto weight = params.at("weight");
    const auto weight_scale = params.at("weight_scale");
    std::optional<infinilm::runtime::Tensor> bias_opt;
    if (has_bias) {
        bias_opt = params.at("bias");
    }
    const bool nz_weight = weight_prepacked_ && weight->ndim() == 4;
    const auto expected_input_features = weight_prepacked_
                                                ? (nz_weight
                                                       ? weight->shape()[1] * 16
                                                       : weight->shape()[0])
                                               : weight->shape()[1];
    if (!packed || !scale || !weight || !weight_scale ||
        packed->ndim() != 2 ||
        scale->shape() != infinilm::runtime::Shape{packed->shape()[0], 1} ||
        packed->dtype() != infinilm::runtime::DataType::kInt8 ||
        scale->dtype() != infinilm::runtime::DataType::kFloat32 ||
        (weight->ndim() != 2 && !nz_weight) ||
        packed->shape()[1] != expected_input_features) {
        throw std::invalid_argument(
            "prequantized W8A8 operands and weight geometry differ");
    }

    auto effective_weight_scale = weight_scale;
    if (std::fabs(alpha - 1.0f) > 1e-7f) {
        effective_weight_scale = infinilm::runtime::Tensor::empty(
            runtime, weight_scale->shape(), weight_scale->dtype());
        infinilm::execution::submit<infini::ops::Mul>(
            runtime, infinilm::execution::RecordingPolicy::AllowHostReplay,
            infini::ops::Mul::DefaultConfig(runtime.device().type()),
            weight_scale, static_cast<double>(alpha), effective_weight_scale);
    }

    const auto rows = packed->shape()[0];
    const auto out_features = weight_prepacked_
                                    ? (nz_weight
                                           ? weight->shape()[0] * 32
                                           : weight->shape()[1])
                                    : weight->shape()[0];
    auto output = infinilm::runtime::Tensor::empty(
        runtime, {rows, out_features}, output_dtype);
    if (bias_opt) {
        bias_opt = std::make_optional(
            bias_opt.value()->as_strided({rows, out_features}, {0, 1}));
    }
    infinilm::execution::submit<infini::ops::I8Gemm>(
        runtime, infinilm::execution::RecordingPolicy::AllowHostReplay,
        infini::ops::I8Gemm::DefaultConfig(runtime.device().type()),
        output, packed, scale,
        weight_prepacked_ ? weight : weight->permute({1, 0}),
        effective_weight_scale, bias_opt);
    INFINILM_TRACE_OP(I8Gemm, output);
    return output;
}

std::shared_ptr<BaseQuantization> CompressedTensors::process_weights_after_loading(
    infinilm::runtime::Runtime &runtime, ParamsMap &params,
    bool pre_transpose, int) const {
    if (!pre_transpose) {
        return nullptr;
    }
    runtime.assertOwnerThread();
    if (runtime.graphCaptureState() !=
        infinilm::execution::graph::GraphManager::CaptureState::kInactive) {
        throw std::logic_error("W8A8 pre_transpose requires inactive graph state");
    }

    const auto weight_it = params.find("weight");
    if (weight_it == params.end()) {
        throw std::invalid_argument("W8A8 pre_transpose requires weight");
    }
    auto source = weight_it->second;
    if (!source || source->device() != runtime.device() || source->ndim() != 2) {
        throw std::invalid_argument("W8A8 pre_transpose weight geometry is invalid");
    }
    try {
        const auto output_channels = source->shape()[0];
        const auto input_channels = source->shape()[1];
        const bool use_nz = false;
        auto permuted = source->permute({1, 0});
        infinilm::runtime::Tensor pending;
        if (use_nz) {
            pending = permuted
                          ->view({input_channels / 16, 16,
                                  output_channels / 32, 32})
                          ->permute({2, 0, 1, 3})
                          ->contiguous(runtime);
        } else {
            pending = infinilm::runtime::Tensor::empty(
                runtime, permuted->shape(), permuted->dtype());
            pending->copy_from(runtime, permuted);
        }
        weight_it->second = pending;
        weight_prepacked_ = true;
    } catch (...) {
        const auto failure = std::current_exception();
        try {
            runtime.syncStream();
        } catch (...) {
            std::terminate();
        }
        std::rethrow_exception(failure);
    }
    return std::const_pointer_cast<BaseQuantization>(shared_from_this());
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
