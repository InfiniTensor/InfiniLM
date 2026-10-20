#pragma once

#include "runtime/runtime.hpp"
#include <stdexcept>

namespace infinilm::layers {

// These checks precede the first write in the residual layer. Ops owns math.
inline void check_add_rms_norm(infinilm::runtime::Runtime &runtime,
                               const infinilm::runtime::Tensor &out, const infinilm::runtime::Tensor &residual,
                               const infinilm::runtime::Tensor &a, const infinilm::runtime::Tensor &b,
                               const infinilm::runtime::Tensor &weight) {
    runtime.assertOwnerThread();
    if (!out || !residual || !a || !b || !weight || out->device() != runtime.device() || residual->device() != runtime.device() || a->device() != runtime.device() || b->device() != runtime.device() || weight->device() != runtime.device()) {
        throw std::invalid_argument("FusedAddRmsNorm runtime and tensor devices differ");
    }
    if (out->shape() != a->shape() || residual->shape() != b->shape() || out->dtype() != a->dtype() || residual->dtype() != b->dtype()) {
        throw std::invalid_argument("fused add+RMSNorm output and input views differ");
    }
}

inline void check_layer_norm(infinilm::runtime::Runtime &runtime,
                             const infinilm::runtime::Tensor &x, const infinilm::runtime::Tensor &weight,
                             const infinilm::runtime::Tensor &bias) {
    runtime.assertOwnerThread();
    if (!x || !weight || x->ndim() == 0 || weight->ndim() != 1 || x->shape().back() == 0 || weight->size(0) != x->shape().back() || weight->device() != x->device() || weight->dtype() != x->dtype() || (bias && (bias->ndim() != 1 || bias->size(0) != x->shape().back() || bias->device() != x->device() || bias->dtype() != x->dtype()))) {
        throw std::invalid_argument("LayerNorm requires matching input, last-dimension weight and optional bias");
    }
    if (runtime.device() != x->device()) {
        throw std::invalid_argument("LayerNorm runtime and tensor devices differ");
    }
}
} // namespace infinilm::layers
