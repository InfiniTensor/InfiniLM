#include "infinicore/ops/rotary_embedding.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/rotary_embedding.h"

namespace infinicore::op::rotary_embedding_impl::infiniops {
namespace {

using TensorMeta = ::infinicore::op::infiniops::TensorMeta;

struct PlannedMeta {
    TensorMeta positions, query, cos_sin_cache;
    std::optional<TensorMeta> key;
    graph::GraphTensor positions_tensor, query_tensor, cos_sin_cache_tensor;
    std::optional<graph::GraphTensor> key_tensor;
    int64_t head_size;
    bool is_neox;
    int64_t rope_dim_offset;
    bool inverse;
};

} // namespace

void *plan(const Tensor &positions,
           Tensor query,
           std::optional<Tensor> key,
           const Tensor &cos_sin_cache,
           int64_t head_size,
           bool is_neox,
           int64_t rope_dim_offset,
           bool inverse) {
    const auto device_type = query->device().type();
    INFINICORE_ASSERT(device_type == Device::Type::kNvidia
                      || device_type == Device::Type::kMetax
                      || device_type == Device::Type::kIluvatar
                      || device_type == Device::Type::kCambricon
                      || device_type == Device::Type::kAscend);
    return new PlannedMeta{
        TensorMeta(positions),
        TensorMeta(query),
        TensorMeta(cos_sin_cache),
        key ? std::optional<TensorMeta>{TensorMeta(*key)} : std::nullopt,
        graph::GraphTensor(positions),
        graph::GraphTensor(query),
        graph::GraphTensor(cos_sin_cache),
        key ? std::optional<graph::GraphTensor>{graph::GraphTensor(*key)} : std::nullopt,
        head_size,
        is_neox,
        rope_dim_offset,
        inverse};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    auto config = ::infinicore::op::infiniops::defaultConfigForDevice<
        infini::ops::RotaryEmbedding>(
        planned->query.device.type());
    const std::optional<infini::ops::Tensor> key = planned->key
                                                     ? std::optional<infini::ops::Tensor>{planned->key->tensor(*planned->key_tensor)}
                                                     : std::nullopt;
    infini::ops::RotaryEmbedding::Call(
        handle,
        config,
        planned->positions.tensor(planned->positions_tensor),
        planned->query.tensor(planned->query_tensor),
        key,
        planned->cos_sin_cache.tensor(planned->cos_sin_cache_tensor),
        planned->head_size,
        planned->is_neox,
        planned->rope_dim_offset,
        planned->inverse);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    RotaryEmbedding::plan_dispatcher().registerDevice(Device::Type::kNvidia, &plan);
    RotaryEmbedding::run_dispatcher().registerDevice(Device::Type::kNvidia, &run);
    RotaryEmbedding::cleanup_dispatcher().registerDevice(Device::Type::kNvidia, &cleanup);
    RotaryEmbedding::plan_dispatcher().registerDevice(Device::Type::kMetax, &plan);
    RotaryEmbedding::run_dispatcher().registerDevice(Device::Type::kMetax, &run);
    RotaryEmbedding::cleanup_dispatcher().registerDevice(Device::Type::kMetax, &cleanup);
    RotaryEmbedding::plan_dispatcher().registerDevice(Device::Type::kIluvatar, &plan);
    RotaryEmbedding::run_dispatcher().registerDevice(Device::Type::kIluvatar, &run);
    RotaryEmbedding::cleanup_dispatcher().registerDevice(Device::Type::kIluvatar, &cleanup);
    RotaryEmbedding::plan_dispatcher().registerDevice(Device::Type::kCambricon, &plan);
    RotaryEmbedding::run_dispatcher().registerDevice(Device::Type::kCambricon, &run);
    RotaryEmbedding::cleanup_dispatcher().registerDevice(Device::Type::kCambricon, &cleanup);
    RotaryEmbedding::plan_dispatcher().registerDevice(Device::Type::kAscend, &plan);
    RotaryEmbedding::run_dispatcher().registerDevice(Device::Type::kAscend, &run);
    RotaryEmbedding::cleanup_dispatcher().registerDevice(Device::Type::kAscend, &cleanup);
    return true;
}();

} // namespace infinicore::op::rotary_embedding_impl::infiniops
#endif
