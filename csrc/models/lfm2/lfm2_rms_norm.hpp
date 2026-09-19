#pragma once

#include <infinicore/nn/rmsnorm.hpp>
#include <infinicore/ops/add.hpp>
#include <infinicore/ops/mul.hpp>
#include <infinicore/ops/rms_norm.hpp>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace infinilm::models::lfm2 {

inline infinicore::Tensor lfm2_unit_weight(
    size_t hidden, const infinicore::DataType &dtype,
    const infinicore::Device &device) {
    if (device.getType() == infinicore::Device::Type::ASCEND) {
        // Avoid queueing aclnnInplaceOne during model construction.  With a
        // torch_npu-owned ACL context those tiny kernels can remain pending
        // until the first RankWorker synchronization.  Build the constant in
        // device-pinned host memory and use InfiniCore's synchronous Ascend
        // H2D path instead.
        auto host = infinicore::Tensor::empty(
            {hidden},
            dtype,
            infinicore::Device(infinicore::Device::Type::CPU, 0),
            true);
        if (dtype == infinicore::DataType::F32) {
            std::fill_n(reinterpret_cast<float *>(host->data()), hidden, 1.0f);
        } else if (dtype == infinicore::DataType::F16) {
            std::fill_n(
                reinterpret_cast<std::uint16_t *>(host->data()),
                hidden,
                static_cast<std::uint16_t>(0x3c00));
        } else if (dtype == infinicore::DataType::BF16) {
            std::fill_n(
                reinterpret_cast<std::uint16_t *>(host->data()),
                hidden,
                static_cast<std::uint16_t>(0x3f80));
        } else {
            throw std::runtime_error(
                "LFM2 RMSNorm unit weight requires F32, F16, or BF16");
        }
        return host->to(device);
    }
    return infinicore::Tensor::ones({hidden}, dtype, device);
}

// Match LFM2's reference boundary: normalize in F32, cast to activation
// dtype, then multiply by the learned weight in that dtype. The unit weight
// is runtime data, not a checkpoint parameter; no ATen cast is required.
inline infinicore::Tensor lfm2_rms_norm(
    const infinicore::Tensor &input, const infinicore::Tensor &weight,
    float eps, const infinicore::Tensor &unit_weight) {
    if (input->dtype() == infinicore::DataType::F32
        || input->device().getType() == infinicore::Device::Type::ASCEND) {
        // Ascend aclnnMul does not reliably complete when the learned weight
        // is exposed as a zero-stride broadcast view.  Its RMSNorm backend
        // already accepts the real weight and returns the requested activation
        // dtype, so use that equivalent fused path on this platform.
        return infinicore::op::rms_norm(input, weight, eps);
    }
    auto normalized = infinicore::op::rms_norm(input, unit_weight, eps);
    // The final axis is the norm's feature axis.
    auto strides = input->strides();
    for (auto &stride : strides) { stride = 0; }
    strides.back() = 1;
    auto expanded = weight->as_strided(input->shape(), strides);
    return infinicore::op::mul(normalized, expanded);
}

class Lfm2RMSNorm : public infinicore::nn::RMSNorm {
public:
    Lfm2RMSNorm(size_t hidden, double eps, const infinicore::DataType &dtype,
               const infinicore::Device &device)
        : infinicore::nn::RMSNorm(hidden, eps, dtype, device),
          unit_weight_(lfm2_unit_weight(hidden, dtype, device)) {}

    infinicore::Tensor forward(const infinicore::Tensor &input) const {
        return lfm2_rms_norm(input, weight(), static_cast<float>(eps()), unit_weight_);
    }

    void forward_inplace(infinicore::Tensor &input, infinicore::Tensor &residual) const {
        residual = residual ? infinicore::op::add(input, residual) : input;
        input = forward(residual);
    }

private:
    infinicore::Tensor unit_weight_;
};
} // namespace infinilm::models::lfm2
