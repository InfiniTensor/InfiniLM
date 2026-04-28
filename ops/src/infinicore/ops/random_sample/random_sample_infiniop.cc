#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/random_sample.hpp"
#include <infiniop.h>

namespace infinicore::op::random_sample_impl::infiniop_backend {

thread_local common::OpCache<size_t, infiniopRandomSampleDescriptor_t> caches(
    100, // capacity
    [](infiniopRandomSampleDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyRandomSampleDescriptor(desc));
            desc = nullptr;
        }
    });

static void calculate(
    Tensor indices,
    Tensor logits,
    float random_val,
    float topp,
    int topk,
    float temperature) {
    // cache per (result desc + logits desc) on device
    size_t seed = hash_combine(indices, logits);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopRandomSampleDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateRandomSampleDescriptor(
            context::getInfiniopHandle(device), &desc,
            indices->desc(), logits->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetRandomSampleWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopRandomSample(
        desc,
        workspace->data(), workspace_size,
        indices->data(), logits->data(),
        random_val, topp, topk, temperature,
        context::getStream()));
}

} // namespace infinicore::op::random_sample_impl::infiniop_backend

namespace infinicore::op {

static bool registered = []() {
    RandomSample::dispatcher().registerAll(&random_sample_impl::infiniop_backend::calculate, false);
    return true;
}();

} // namespace infinicore::op
