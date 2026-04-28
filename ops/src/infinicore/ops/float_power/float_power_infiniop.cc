#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/float_power.hpp"
#include <infiniop.h>

namespace infinicore::op::float_power_impl::infiniop {

// =======================================================================
// Descriptor Cache
// =======================================================================

thread_local common::OpCache<size_t, infiniopFloatPowerDescriptor_t> caches(
    100,
    [](infiniopFloatPowerDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(
                infiniopDestroyFloatPowerDescriptor(desc));
            desc = nullptr;
        }
    });

// =======================================================================
// 1. Scalar Exponent
// =======================================================================

void calculate_scalar(Tensor output,
                      Tensor input,
                      double exponent) {
    // Hash: output / input meta + double exponent
    size_t seed = hash_combine(output, input, exponent);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();
    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopFloatPowerDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(
            infiniopCreateFloatPowerDescriptor(
                context::getInfiniopHandle(output->device()),
                &desc,
                output->desc(),
                input->desc(),
                nullptr, // exponent tensor descriptor = null
                static_cast<float>(exponent)));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(
        infiniopGetFloatPowerWorkspaceSize(desc, &workspace_size));

    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(
        infiniopFloatPower(
            desc,
            workspace->data(),
            workspace_size,
            output->data(),
            input->data(),
            nullptr, // exponent data pointer = null
            context::getStream()));
}

// =======================================================================
// 2. Tensor Exponent
// =======================================================================

void calculate_tensor(Tensor output,
                      Tensor input,
                      Tensor exponent) {
    size_t seed = hash_combine(output, input, exponent);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();
    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopFloatPowerDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(
            infiniopCreateFloatPowerDescriptor(
                context::getInfiniopHandle(output->device()),
                &desc,
                output->desc(),
                input->desc(),
                exponent->desc(), // tensor exponent
                0.0f              // scalar ignored
                ));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(
        infiniopGetFloatPowerWorkspaceSize(desc, &workspace_size));

    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(
        infiniopFloatPower(
            desc,
            workspace->data(),
            workspace_size,
            output->data(),
            input->data(),
            exponent->data(),
            context::getStream()));
}

// =======================================================================
// 3. Dispatcher Registration
// =======================================================================

static bool registered = []() {
    FloatPower::dispatcher_scalar().registerAll(&calculate_scalar, false);
    FloatPower::dispatcher_tensor().registerAll(&calculate_tensor, false);
    return true;
}();

} // namespace infinicore::op::float_power_impl::infiniop
