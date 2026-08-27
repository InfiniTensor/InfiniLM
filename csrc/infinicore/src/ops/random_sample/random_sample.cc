#include "infinicore/ops/random_sample.hpp"

#include "../../utils.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/argmax.h"
#endif

namespace infinicore::op {
namespace {

#ifdef ENABLE_INFINIOPS_API
bool tryGreedyWithInfiniOps(
    Tensor indices, Tensor logits,
    float random_value, float top_p, int top_k, float temperature) {
    const auto dtype = logits->dtype();
    const auto device_type = logits->device().type();
    if ((device_type != Device::Type::kNvidia
         && device_type != Device::Type::kMetax
         && device_type != Device::Type::kIluvatar
         && device_type != Device::Type::kMoore
         && device_type != Device::Type::kCambricon
         && device_type != Device::Type::kAscend)
        || (random_value != 0.0f
            && top_p != 0.0f
            && top_k != 1
            && temperature != 0.0f)
        || logits->ndim() != 1
        || logits->numel() == 0
        || !logits->is_contiguous()
        || (dtype != DataType::kFloat16 && dtype != DataType::kBFloat16 && dtype != DataType::kFloat32)
        || indices->numel() != 1
        || indices->dtype() != DataType::kInt64
        || !indices->is_contiguous()) {
        return false;
    }

    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    const infiniops::TensorMeta logits_meta(logits);
    const infiniops::TensorMeta indices_meta(indices);
    auto argmax_config = infiniops::defaultConfigForDevice<infini::ops::Argmax>(
        logits_meta.device.type());
    const std::optional<int64_t> no_dim;
    infini::ops::Argmax::Call(
        handle,
        argmax_config,
        logits_meta.tensor(logits),
        no_dim,
        false,
        indices_meta.tensor(indices));
    return true;
}
#endif

} // namespace

common::OpDispatcher<RandomSample::schema> &RandomSample::dispatcher() {
    static common::OpDispatcher<RandomSample::schema> dispatcher_;
    return dispatcher_;
};

void RandomSample::execute(
    Tensor indices, Tensor logits,
    float random_val, float topp, int topk, float temperature) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(indices, logits);
    infinicore::context::setDevice(logits->device());
#ifdef ENABLE_INFINIOPS_API
    if (tryGreedyWithInfiniOps(
            indices, logits, random_val, topp, topk, temperature)) {
        return;
    }
#endif

    dispatcher().lookup(logits->device().type())(
        indices, logits, random_val, topp, topk, temperature);
}

Tensor random_sample(
    Tensor logits,
    float random_val,
    float topp,
    int topk,
    float temperature) {
    auto indices = Tensor::empty({}, DataType::kInt32, logits->device());
    random_sample_(indices, logits, random_val, topp, topk, temperature);
    return indices;
}

void random_sample_(
    Tensor indices,
    Tensor logits,
    float random_val,
    float topp,
    int topk,
    float temperature) {
    RandomSample::execute(indices, logits, random_val, topp, topk, temperature);
}

} // namespace infinicore::op
