#include "infinicore/ops/mha.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/flash_attn_varlen_func.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace infinicore::op::mha_impl::infiniops {
namespace helpers = ::infinicore::op::infiniops;
namespace {

bool is_supported(const Tensor &out, const Tensor &q, const Tensor &k,
                  const Tensor &v, const std::optional<Tensor> &alibi_slopes) {
    if (!out || !q || !k || !v || alibi_slopes.has_value()) {
        return false;
    }

    const auto device_type = q->device().type();
    if (!helpers::isSupportedDevice(device_type) || q->ndim() != 4 || k->ndim() != 4 || v->ndim() != 4 || out->shape() != q->shape() || q->size(0) != k->size(0) || q->size(0) != v->size(0) || q->size(1) != k->size(1) || q->size(1) == 0 || q->size(2) % k->size(2) != 0 || q->size(3) != k->size(3) || k->size(3) != v->size(3) || !q->is_contiguous() || !k->is_contiguous() || !v->is_contiguous() || !out->is_contiguous()) {
        return false;
    }

    const auto dtype = q->dtype();
    return (dtype == DataType::kFloat16 || dtype == DataType::kBFloat16) && out->dtype() == dtype && k->dtype() == dtype && v->dtype() == dtype;
}

Tensor flatten_batch_and_sequence(const Tensor &tensor) {
    const auto &shape = tensor->shape();
    return tensor->view({shape[0] * shape[1], shape[2], shape[3]});
}

Tensor make_cumulative_sequence_lengths(std::size_t batch_size,
                                        std::size_t sequence_length,
                                        const Device &device) {
    std::vector<std::int32_t> lengths(batch_size + 1);
    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        lengths[batch + 1] = static_cast<std::int32_t>((batch + 1) * sequence_length);
    }
    auto host_tensor = Tensor::from_blob(lengths.data(), Shape{batch_size + 1}, DataType::kInt32,
                                         Device{Device::Type::kCpu});
    return host_tensor->to(device);
}

std::size_t
implementation_index_for_device(infini::ops::Device::Type device_type) {
    if (device_type == infini::ops::Device::Type::kIluvatar || device_type == infini::ops::Device::Type::kAscend) {
        return 0;
    }
    if (device_type == infini::ops::Device::Type::kMoore) {
        return 8;
    }
    return 16;
}

struct PlannedMeta {
    Tensor q_flat, k_flat, v_flat, cu_seqlens;
    helpers::TensorMeta out, q, k, v, cu_seqlens_meta;
    graph::GraphTensor out_tensor, q_tensor, k_tensor, v_tensor,
        cu_seqlens_tensor;
    std::int64_t max_seqlen_q, max_seqlen_k;
    float scale;
    bool is_causal;
};

} // namespace

void *plan(Tensor out, const Tensor &q, const Tensor &k, const Tensor &v,
           std::optional<Tensor> alibi_slopes, float scale, bool is_causal) {
    INFINICORE_ASSERT(is_supported(out, q, k, v, alibi_slopes));
    const auto q_flat = flatten_batch_and_sequence(q);
    const auto k_flat = flatten_batch_and_sequence(k);
    const auto v_flat = flatten_batch_and_sequence(v);
    const auto cu_seqlens = make_cumulative_sequence_lengths(q->size(0), q->size(1), q->device());
    return new PlannedMeta{q_flat,
                           k_flat,
                           v_flat,
                           cu_seqlens,
                           helpers::TensorMeta(out),
                           helpers::TensorMeta(q_flat),
                           helpers::TensorMeta(k_flat),
                           helpers::TensorMeta(v_flat),
                           helpers::TensorMeta(cu_seqlens),
                           graph::GraphTensor(out),
                           graph::GraphTensor(q_flat),
                           graph::GraphTensor(k_flat),
                           graph::GraphTensor(v_flat),
                           graph::GraphTensor(cu_seqlens),
                           static_cast<std::int64_t>(q->size(1)),
                           static_cast<std::int64_t>(k->size(1)),
                           scale,
                           is_causal};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    infini::ops::Handle handle;
    handle.set_stream(context::getStream());

    const auto device_type = planned->q.device.type();
    const auto config = helpers::configForImplementation<infini::ops::FlashAttnVarlenFunc>(
        device_type, implementation_index_for_device(device_type));
    const std::optional<infini::ops::Tensor> no_tensor;

    infini::ops::FlashAttnVarlenFunc::Call(
        handle, config, planned->q.tensor(planned->q_tensor),
        planned->k.tensor(planned->k_tensor),
        planned->v.tensor(planned->v_tensor),
        planned->cu_seqlens_meta.tensor(planned->cu_seqlens_tensor),
        planned->cu_seqlens_meta.tensor(planned->cu_seqlens_tensor), no_tensor,
        no_tensor, planned->max_seqlen_q, planned->max_seqlen_k, 0.0,
        std::optional<double>{planned->scale}, planned->is_causal,
        std::vector<std::int64_t>{-1, -1}, 0.0, false, false,
        planned->out.tensor(planned->out_tensor), no_tensor, no_tensor);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    auto &plan_dispatcher = MultiheadAttention::plan_dispatcher();
    auto &run_dispatcher = MultiheadAttention::run_dispatcher();
    auto &cleanup_dispatcher = MultiheadAttention::cleanup_dispatcher();
    helpers::registerSupportedDevices(plan_dispatcher, &plan);
    helpers::registerSupportedDevices(run_dispatcher, &run);
    helpers::registerSupportedDevices(cleanup_dispatcher, &cleanup);
    return true;
}();

} // namespace infinicore::op::mha_impl::infiniops
#endif
