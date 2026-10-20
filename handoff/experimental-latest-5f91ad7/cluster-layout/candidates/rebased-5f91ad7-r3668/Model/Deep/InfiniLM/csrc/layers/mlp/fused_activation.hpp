#pragma once

#include "../../execution/submit.hpp"
#include <limits>
#include <stdexcept>

namespace infinilm::layers::mlp {

// GateUp output is packed [gate, up]. Keep the existing row-aligned launch
// limit in the LM layer; the caller owns the destination and Ops owns math.
inline void packed_activation_into(infinilm::runtime::Runtime &runtime,
                                   const infinilm::runtime::Tensor &out, const infinilm::runtime::Tensor &input) {
    using infinilm::runtime::Size;
    runtime.assertOwnerThread();
    if (!out || !input || out->ndim() == 0 || input->ndim() != out->ndim() || input->dtype() != out->dtype()) {
        throw std::invalid_argument("SiluAndMul rank or dtype is unsupported");
    }
    if (out->shape().back() > std::numeric_limits<Size>::max() / 2 || input->shape().back() != 2 * out->shape().back()) {
        throw std::invalid_argument("SiluAndMul last dimensions do not match");
    }
    for (size_t dim = 0; dim + 1 < out->ndim(); ++dim) {
        if (input->size(dim) != out->size(dim)) {
            throw std::invalid_argument("SiluAndMul leading dimensions do not match");
        }
    }
    const auto config = infini::ops::SiluAndMul::DefaultConfig(runtime.device().type());
    constexpr Size limit = Size{1} << 30;
    if (out->numel() <= limit || !out->is_contiguous() || !input->is_contiguous()) {
        execution::submit<infini::ops::SiluAndMul>(runtime, config, input, out);
        INFINILM_TRACE_OP(SiluAndMul, out);
        return;
    }
    const Size width = out->shape().back();
    if (width == 0 || width > limit || input->numel() != out->numel() * 2) {
        throw std::invalid_argument("SiluAndMul chunk geometry is unsupported");
    }
    const Size rows = out->numel() / width;
    const Size max_rows = limit / width;
    const auto output_rows = out->view({rows, width});
    const auto input_rows = input->view({rows, width * 2});
    for (Size start = 0; start < rows; start += max_rows) {
        const Size remaining = rows - start;
        const Size count = remaining < max_rows ? remaining : max_rows;
        const auto output_chunk = output_rows->narrow({{0, start, count}});
        execution::submit<infini::ops::SiluAndMul>(runtime, config,
                                                   input_rows->narrow({{0, start, count}}),
                                                   output_chunk);
        INFINILM_TRACE_OP(SiluAndMul, output_chunk);
    }
}
// Separate branches keep the same packed [gate, up] algorithm. Three real
// Ops calls replace the former combined prepared-call object.
inline void branch_activation_into(infinilm::runtime::Runtime &runtime,
                                   const infinilm::runtime::Tensor &out, const infinilm::runtime::Tensor &up_input,
                                   const infinilm::runtime::Tensor &gate_input) {
    using infinilm::runtime::Size;
    runtime.assertOwnerThread();
    if (!out || !up_input || !gate_input || out->device() != runtime.device() || up_input->device() != runtime.device() || gate_input->device() != runtime.device()) {
        throw std::invalid_argument("swiglu runtime and tensor devices differ");
    }
    if (out->shape() != up_input->shape() || up_input->shape() != gate_input->shape() || out->dtype() != up_input->dtype() || up_input->dtype() != gate_input->dtype()) {
        throw std::invalid_argument("swiglu input/output shape or dtype differs");
    }
    if (out->shape().empty()) {
        throw std::invalid_argument("swiglu requires a non-scalar tensor");
    }
    constexpr Size limit = Size{1} << 30;
    const auto copy_config = infini::ops::Copy::DefaultConfig(runtime.device().type());
    const auto silu_config = infini::ops::SiluAndMul::DefaultConfig(runtime.device().type());
    const bool non_blocking = false;
    const auto run = [&](const infinilm::runtime::Tensor &destination,
                         const infinilm::runtime::Tensor &up_source,
                         const infinilm::runtime::Tensor &gate_source) {
        auto shape = up_source->shape();
        const Size width = shape.back();
        if (width > std::numeric_limits<Size>::max() / 2) {
            throw std::overflow_error("swiglu packed width overflows");
        }
        shape.back() *= 2;
        auto packed = infinilm::runtime::Tensor::empty(runtime, shape, up_source->dtype());
        const auto gate = packed->narrow({{packed->ndim() - 1, 0, width}});
        const auto up = packed->narrow({{packed->ndim() - 1, width, width}});
        execution::submit<infini::ops::Copy>(runtime, execution::RecordingPolicy::DeviceOnly,
                                             copy_config, gate_source, non_blocking, gate);
        execution::submit<infini::ops::Copy>(runtime, execution::RecordingPolicy::DeviceOnly,
                                             copy_config, up_source, non_blocking, up);
        execution::submit<infini::ops::SiluAndMul>(runtime, execution::RecordingPolicy::DeviceOnly,
                                                   silu_config, packed, destination);
        INFINILM_TRACE_OP(SwiGLU, destination);
    };
    if (out->numel() <= limit) {
        run(out, up_input, gate_input);
        return;
    }
    const Size width = out->shape().back();
    if (width == 0 || width > limit) {
        throw std::invalid_argument("swiglu chunk geometry is unsupported");
    }
    const Size rows = out->numel() / width;
    const Size max_rows = limit / width;
    const auto output_rows = out->view({rows, width});
    const auto up_rows = up_input->view({rows, width});
    const auto gate_rows = gate_input->view({rows, width});
    for (Size start = 0; start < rows; start += max_rows) {
        const Size remaining = rows - start;
        const Size count = remaining < max_rows ? remaining : max_rows;
        run(output_rows->narrow({{0, start, count}}),
            up_rows->narrow({{0, start, count}}),
            gate_rows->narrow({{0, start, count}}));
    }
}
} // namespace infinilm::layers::mlp
