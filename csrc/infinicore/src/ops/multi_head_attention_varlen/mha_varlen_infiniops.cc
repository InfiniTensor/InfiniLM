#include "infinicore/ops/mha_varlen.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/flash_attn_varlen_func.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace infinicore::op::mha_varlen_impl::infiniops {
namespace {

using TensorMeta = ::infinicore::op::infiniops::TensorMeta;

// TODO: Remove backend-specific implementation indices from InfiniLM once
// InfiniOps provides device-aware default selection for these operators.
std::size_t implementation_index_for_device(
    infini::ops::Device::Type device_type) {
    if (device_type == infini::ops::Device::Type::kIluvatar) {
        return 0;
    }
    if (device_type == infini::ops::Device::Type::kMoore) {
        return 8;
    }
    return 16;
}

bool is_supported(const Tensor &out,
                  const Tensor &q,
                  const Tensor &k,
                  const Tensor &v,
                  const Tensor &cum_seqlens_q,
                  const Tensor &cum_seqlens_k,
                  const std::optional<Tensor> &block_table,
                  int max_seqlen_q,
                  int max_seqlen_k,
                  const std::optional<Tensor> &alibi_slopes) {
    const bool paged = block_table.has_value();
    const auto dtype = q->dtype();
    const auto device_type = out->device().type();
    if ((device_type != Device::Type::kNvidia
         && device_type != Device::Type::kMetax
         && device_type != Device::Type::kMoore
         && device_type != Device::Type::kCambricon
         && device_type != Device::Type::kIluvatar)
        || q->ndim() != 3
        || out->ndim() != 3
        || ((paged && (k->ndim() != 4 || v->ndim() != 4))
            || (!paged && (k->ndim() != 3 || v->ndim() != 3)))
        || k->shape() != v->shape()
        || out->shape() != q->shape()
        || (dtype != DataType::kFloat16 && dtype != DataType::kBFloat16)
        || out->dtype() != dtype
        || k->dtype() != dtype
        || v->dtype() != dtype
        || q->size(1) == 0
        || k->size(k->ndim() - 2) == 0
        || q->size(1) % k->size(k->ndim() - 2) != 0
        || q->size(2) == 0
        || q->size(2) > 256
        || q->size(2) % 8 != 0
        || (device_type == Device::Type::kIluvatar && !paged)
        || ((device_type == Device::Type::kMoore
             || device_type == Device::Type::kIluvatar)
            && q->size(2) != 64
            && q->size(2) != 128)
        || q->size(2) != k->size(k->ndim() - 1)
        || q->stride(2) != 1
        || out->stride(2) != 1
        || k->stride(k->ndim() - 1) != 1
        || v->stride(v->ndim() - 1) != 1
        || cum_seqlens_q->ndim() != 1
        || cum_seqlens_k->ndim() != 1
        || cum_seqlens_q->shape() != cum_seqlens_k->shape()
        || cum_seqlens_q->numel() < 2
        || cum_seqlens_q->dtype() != DataType::kInt32
        || cum_seqlens_k->dtype() != DataType::kInt32
        || !cum_seqlens_q->is_contiguous()
        || !cum_seqlens_k->is_contiguous()
        || max_seqlen_q <= 0
        || max_seqlen_k <= 0) {
        return false;
    }

    if (block_table
        && (block_table.value()->ndim() != 2
            || block_table.value()->size(0) + 1 != cum_seqlens_q->size(0)
            || block_table.value()->dtype() != DataType::kInt32
            || !block_table.value()->is_contiguous()
            || k->size(1) % 256 != 0
            || (device_type == Device::Type::kMoore
                && static_cast<std::size_t>(max_seqlen_k)
                       > block_table.value()->size(1) * k->size(1)))) {
        return false;
    }

    if (alibi_slopes
        && ((alibi_slopes.value()->ndim() != 1
             && alibi_slopes.value()->ndim() != 2)
            || (device_type == Device::Type::kMoore
                && (!paged || alibi_slopes.value()->ndim() != 1))
            || (device_type == Device::Type::kIluvatar
                && alibi_slopes.value()->ndim() != 1)
            || alibi_slopes.value()->dtype() != DataType::kFloat32
            || !alibi_slopes.value()->is_contiguous()
            || alibi_slopes.value()->device() != out->device()
            || (alibi_slopes.value()->ndim() == 1
                && alibi_slopes.value()->size(0) != q->size(1))
            || (alibi_slopes.value()->ndim() == 2
                && (alibi_slopes.value()->size(0) + 1
                        != cum_seqlens_q->size(0)
                    || alibi_slopes.value()->size(1) != q->size(1))))) {
        return false;
    }

    return true;
}

struct PlannedMeta {
    TensorMeta out, q, k, v, cum_seqlens_q, cum_seqlens_k;
    std::optional<TensorMeta> block_table, alibi_slopes;
    graph::GraphTensor out_tensor, q_tensor, k_tensor, v_tensor,
        cum_seqlens_q_tensor, cum_seqlens_k_tensor;
    std::optional<graph::GraphTensor> block_table_tensor, alibi_slopes_tensor;
    int max_seqlen_q, max_seqlen_k;
    float scale;
};

} // namespace

void *plan(Tensor out,
           const Tensor &q,
           const Tensor &k,
           const Tensor &v,
           const Tensor &cum_seqlens_q,
           const Tensor &cum_seqlens_k,
           std::optional<Tensor> block_table,
           int max_seqlen_q,
           int max_seqlen_k,
           std::optional<Tensor> alibi_slopes,
           float scale) {
    INFINICORE_ASSERT(is_supported(
        out, q, k, v, cum_seqlens_q, cum_seqlens_k, block_table,
        max_seqlen_q, max_seqlen_k, alibi_slopes));
    return new PlannedMeta{
        TensorMeta(out),
        TensorMeta(q),
        TensorMeta(k),
        TensorMeta(v),
        TensorMeta(cum_seqlens_q),
        TensorMeta(cum_seqlens_k),
        block_table ? std::optional<TensorMeta>{TensorMeta(*block_table)}
                    : std::nullopt,
        alibi_slopes ? std::optional<TensorMeta>{TensorMeta(*alibi_slopes)}
                     : std::nullopt,
        graph::GraphTensor(out),
        graph::GraphTensor(q),
        graph::GraphTensor(k),
        graph::GraphTensor(v),
        graph::GraphTensor(cum_seqlens_q),
        graph::GraphTensor(cum_seqlens_k),
        block_table
            ? std::optional<graph::GraphTensor>{graph::GraphTensor(*block_table)}
            : std::nullopt,
        alibi_slopes
            ? std::optional<graph::GraphTensor>{graph::GraphTensor(*alibi_slopes)}
            : std::nullopt,
        max_seqlen_q,
        max_seqlen_k,
        scale};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    const auto device_type = planned->q.device.type();
    const auto implementation_index = implementation_index_for_device(device_type);
    auto config = ::infinicore::op::infiniops::configForImplementation<
        infini::ops::FlashAttnVarlenFunc>(device_type, implementation_index);

    const std::optional<infini::ops::Tensor> no_tensor;
    const std::optional<infini::ops::Tensor> block_table = planned->block_table
                                                             ? std::optional<infini::ops::Tensor>{
                                                                   planned->block_table->tensor(*planned->block_table_tensor)}
                                                             : std::nullopt;
    const std::optional<infini::ops::Tensor> alibi_slopes = planned->alibi_slopes
                                                              ? std::optional<infini::ops::Tensor>{
                                                                    planned->alibi_slopes->tensor(*planned->alibi_slopes_tensor)}
                                                              : std::nullopt;

    infini::ops::FlashAttnVarlenFunc::Call(
        handle,
        config,
        planned->q.tensor(planned->q_tensor),
        planned->k.tensor(planned->k_tensor),
        planned->v.tensor(planned->v_tensor),
        planned->cum_seqlens_q.tensor(planned->cum_seqlens_q_tensor),
        planned->cum_seqlens_k.tensor(planned->cum_seqlens_k_tensor),
        alibi_slopes,
        block_table,
        static_cast<std::int64_t>(planned->max_seqlen_q),
        static_cast<std::int64_t>(planned->max_seqlen_k),
        0.0,
        std::optional<double>{planned->scale},
        true,
        std::vector<std::int64_t>{-1, -1},
        0.0,
        false,
        false,
        planned->out.tensor(planned->out_tensor),
        no_tensor,
        no_tensor);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    MultiheadAttentionVarlen::plan_dispatcher().registerDevice(
        Device::Type::kNvidia, &plan);
    MultiheadAttentionVarlen::run_dispatcher().registerDevice(
        Device::Type::kNvidia, &run);
    MultiheadAttentionVarlen::cleanup_dispatcher().registerDevice(
        Device::Type::kNvidia, &cleanup);
    MultiheadAttentionVarlen::plan_dispatcher().registerDevice(
        Device::Type::kMetax, &plan);
    MultiheadAttentionVarlen::run_dispatcher().registerDevice(
        Device::Type::kMetax, &run);
    MultiheadAttentionVarlen::cleanup_dispatcher().registerDevice(
        Device::Type::kMetax, &cleanup);
    MultiheadAttentionVarlen::plan_dispatcher().registerDevice(
        Device::Type::kMoore, &plan);
    MultiheadAttentionVarlen::run_dispatcher().registerDevice(
        Device::Type::kMoore, &run);
    MultiheadAttentionVarlen::cleanup_dispatcher().registerDevice(
        Device::Type::kMoore, &cleanup);
    MultiheadAttentionVarlen::plan_dispatcher().registerDevice(
        Device::Type::kCambricon, &plan);
    MultiheadAttentionVarlen::run_dispatcher().registerDevice(
        Device::Type::kCambricon, &run);
    MultiheadAttentionVarlen::cleanup_dispatcher().registerDevice(
        Device::Type::kCambricon, &cleanup);
    MultiheadAttentionVarlen::plan_dispatcher().registerDevice(
        Device::Type::kIluvatar, &plan);
    MultiheadAttentionVarlen::run_dispatcher().registerDevice(
        Device::Type::kIluvatar, &run);
    MultiheadAttentionVarlen::cleanup_dispatcher().registerDevice(
        Device::Type::kIluvatar, &cleanup);
    return true;
}();

} // namespace infinicore::op::mha_varlen_impl::infiniops
#endif
