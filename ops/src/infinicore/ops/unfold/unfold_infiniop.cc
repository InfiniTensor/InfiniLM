#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/unfold.hpp"
#include <algorithm>
#include <infiniop.h>
#include <vector>

namespace infinicore::op::unfold_impl::infiniop {

// 定义描述符缓存
thread_local common::OpCache<size_t, infiniopUnfoldDescriptor_t> caches(
    100, // capacity
    [](infiniopUnfoldDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyUnfoldDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input,
               const std::vector<int64_t> &kernel_sizes,
               const std::vector<int64_t> &dilations,
               const std::vector<int64_t> &paddings,
               const std::vector<int64_t> &strides) {

    // 1. 计算 Hash Key (修复点：手动拆解，避开 hash.hpp 的递归 bug 和 vector 不支持问题)
    size_t seed = 0;

    // 基础 Tensor 支持 (hash.hpp 中有 Tensor 重载)
    hash_combine(seed, output);
    hash_combine(seed, input);

    // Vector 类型必须手动遍历 (hash.hpp 不支持 vector 直接 hash)
    for (auto v : kernel_sizes) {
        hash_combine(seed, v);
    }
    for (auto v : dilations) {
        hash_combine(seed, v);
    }
    for (auto v : paddings) {
        hash_combine(seed, v);
    }
    for (auto v : strides) {
        hash_combine(seed, v);
    }

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopUnfoldDescriptor_t desc = nullptr;

    if (!desc_opt) {
        // 2. 创建描述符

        // 辅助函数：将 int64_t vector 转换为 int vector 以匹配 C API 的 int* 签名
        auto to_int_vec = [](const std::vector<int64_t> &src) {
            std::vector<int> dst(src.size());
            std::transform(src.begin(), src.end(), dst.begin(),
                           [](int64_t val) { return static_cast<int>(val); });
            return dst;
        };

        std::vector<int> k_int = to_int_vec(kernel_sizes);
        std::vector<int> s_int = to_int_vec(strides);
        std::vector<int> p_int = to_int_vec(paddings);
        std::vector<int> d_int = to_int_vec(dilations);

        INFINICORE_CHECK_ERROR(infiniopCreateUnfoldDescriptor(
            context::getInfiniopHandle(output->device()),
            &desc,
            output->desc(),
            input->desc(),
            k_int.data(),
            s_int.data(),
            p_int.data(),
            d_int.data()));

        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    // 3. 获取 Workspace 并执行
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetUnfoldWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopUnfold(
        desc,
        workspace->data(),
        workspace_size,
        output->data(),
        input->data(),
        context::getStream()));
}

// 4. 注册算子实现
static bool registered = []() {
    Unfold::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::unfold_impl::infiniop
