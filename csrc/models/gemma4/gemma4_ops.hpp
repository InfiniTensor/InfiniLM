#pragma once

#include "../../execution/submit.hpp"
#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include <base/add.h>
#include <base/cast.h>
#include <base/copy.h>
#include <base/fill.h>
#include <base/gelu.h>
#include <base/mul.h>
#include <base/rms_norm.h>
#include <base/sigmoid.h>
#include <string>

namespace infinilm::models::gemma4::ops {

inline infinilm::runtime::Tensor make_filled_tensor(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Shape &shape,
    infinilm::runtime::DataType dtype,
    double value) {
    auto t = infinilm::runtime::Tensor::empty(runtime, shape, dtype);
    infinilm::execution::submit<infini::ops::Fill>(
        runtime,
        infini::ops::Fill::DefaultConfig(runtime.device().type()),
        t, value, t);
    return t;
}

inline infinilm::runtime::Tensor make_filled_param(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Shape &shape,
    infinilm::runtime::DataType dtype,
    double value) {
    auto t = infinilm::runtime::Tensor::empty_parameter(runtime, shape, dtype);
    infinilm::execution::submit<infini::ops::Fill>(
        runtime,
        infini::ops::Fill::DefaultConfig(runtime.device().type()),
        t, value, t);
    return t;
}

inline infinilm::runtime::Tensor add(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &a,
    const infinilm::runtime::Tensor &b) {
    auto out = infinilm::runtime::Tensor::empty(runtime, a->shape(), a->dtype());
    infinilm::execution::submit<infini::ops::Add>(
        runtime,
        infini::ops::Add::DefaultConfig(runtime.device().type()),
        a, b, out);
    return out;
}

inline void add_inplace(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &a,
    const infinilm::runtime::Tensor &b) {
    infinilm::execution::submit<infini::ops::Add>(
        runtime,
        infini::ops::Add::DefaultConfig(runtime.device().type()),
        a, b, a);
}

inline infinilm::runtime::Tensor mul(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &a,
    const infinilm::runtime::Tensor &b) {
    auto out = infinilm::runtime::Tensor::empty(runtime, a->shape(), a->dtype());
    infinilm::execution::submit<infini::ops::Mul>(
        runtime,
        infini::ops::Mul::DefaultConfig(runtime.device().type()),
        a, b, out);
    return out;
}

inline void mul_inplace(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &a,
    const infinilm::runtime::Tensor &b) {
    infinilm::execution::submit<infini::ops::Mul>(
        runtime,
        infini::ops::Mul::DefaultConfig(runtime.device().type()),
        a, b, a);
}

inline infinilm::runtime::Tensor scale_out(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &x,
    double alpha) {
    auto s = make_filled_tensor(runtime, {1}, x->dtype(), alpha);
    return mul(runtime, x, s);
}

inline void scale_inplace(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &x,
    double alpha) {
    auto s = make_filled_tensor(runtime, {1}, x->dtype(), alpha);
    mul_inplace(runtime, x, s);
}

inline infinilm::runtime::Tensor gelu_tanh(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &x) {
    auto out = infinilm::runtime::Tensor::empty(runtime, x->shape(), x->dtype());
    infinilm::execution::submit<infini::ops::Gelu>(
        runtime,
        infini::ops::Gelu::DefaultConfig(runtime.device().type()),
        x, std::string{"tanh"}, out);
    return out;
}

inline infinilm::runtime::Tensor rms_norm(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &x,
    const infinilm::runtime::Tensor &weight,
    double eps) {
    auto out = infinilm::runtime::Tensor::empty(runtime, x->shape(), x->dtype());
    infinilm::execution::submit<infini::ops::RmsNorm>(
        runtime,
        infini::ops::RmsNorm::DefaultConfig(runtime.device().type()),
        x, weight, static_cast<float>(eps), out);
    return out;
}

inline infinilm::runtime::Tensor cast(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &x,
    infinilm::runtime::DataType dst_dtype) {
    if (x->dtype() == dst_dtype) {
        return x;
    }
    auto out = infinilm::runtime::Tensor::empty(runtime, x->shape(), dst_dtype);
    infinilm::execution::submit<infini::ops::Cast>(
        runtime,
        infini::ops::Cast::DefaultConfig(runtime.device().type()),
        x, out);
    return out;
}

// Exact softcap: softcap * tanh(x / softcap) == 2 * softcap * sigmoid(x * (2 / softcap)) - softcap.
// Uses native GPU Mul + Sigmoid + Add (zero CPU fallback on CUDA and MetaX C550).
inline void apply_softcap_inplace(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &x,
    double softcap) {
    if (softcap <= 0.0) {
        return;
    }
    const auto dtype = x->dtype();
    auto inv_half_cap = make_filled_tensor(runtime, {1}, dtype, 2.0 / softcap);
    auto two_cap = make_filled_tensor(runtime, {1}, dtype, 2.0 * softcap);
    auto neg_cap = make_filled_tensor(runtime, {1}, dtype, -softcap);
    mul_inplace(runtime, x, inv_half_cap);
    infinilm::execution::submit<infini::ops::Sigmoid>(
        runtime,
        infini::ops::Sigmoid::DefaultConfig(runtime.device().type()),
        x, x);
    mul_inplace(runtime, x, two_cap);
    add_inplace(runtime, x, neg_cap);
}

} // namespace infinilm::models::gemma4::ops
