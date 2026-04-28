#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/ldexp.hpp"
#include <infiniop.h>

namespace infinicore::op::ldexp_impl::infiniop {

// 定义描述符缓存
thread_local common::OpCache<size_t, infiniopLdexpDescriptor_t> caches(
    100, // capacity
    [](infiniopLdexpDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyLdexpDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input, Tensor other) {
    // 1. 计算哈希值
    // 注意：必须手动哈希 strides 以区分 Broadcasting 和 Inplace 情况
    size_t seed = 0;
    auto combine_tensor_meta = [&](Tensor t) {
        infinicore::hash_combine(seed, static_cast<size_t>(t->dtype()));
        for (auto s : t->shape()) {
            infinicore::hash_combine(seed, s);
        }
        for (auto str : t->strides()) {
            infinicore::hash_combine(seed, str);
        }
    };

    combine_tensor_meta(output);
    combine_tensor_meta(input);
    combine_tensor_meta(other);

    // 2. 获取缓存对象
    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();
    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopLdexpDescriptor_t desc = nullptr;

    if (!desc_opt) {
        // 3. 创建描述符
        // 这里后端 (CPU/GPU) 会根据 input/other 类型决定是否需要 workspace
        INFINICORE_CHECK_ERROR(infiniopCreateLdexpDescriptor(
            context::getInfiniopHandle(input->device()),
            &desc,
            output->desc(),
            input->desc(),
            other->desc()));

        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    // 4. 获取 Workspace 并执行
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetLdexpWorkspaceSize(desc, &workspace_size));

    // 如果后端检测到需要转换 Int32 -> Float，workspace_size 会 > 0
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopLdexp(
        desc,
        workspace->data(),
        workspace_size,
        output->data(),
        input->data(),
        other->data(),
        context::getStream()));
}

// 5. 注册算子
static bool registered = []() {
    // 注册为普通算子 (dispatcher 会自动处理 inplace 逻辑)
    Ldexp::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::ldexp_impl::infiniop
