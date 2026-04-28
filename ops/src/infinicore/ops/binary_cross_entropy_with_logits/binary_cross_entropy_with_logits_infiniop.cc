#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/binary_cross_entropy_with_logits.hpp"
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>

namespace infinicore::op::bce_logits_impl::infiniop {

// 定义线程局部的 BCEWithLogits 算子描述符缓存
thread_local common::OpCache<size_t, infiniopBCEWithLogitsDescriptor_t> caches(
    100,
    [](infiniopBCEWithLogitsDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyBCEWithLogitsDescriptor(desc));
            desc = nullptr;
        }
    });

/**
 * @brief 执行 BCEWithLogits 计算
 * @param out 输出 Tensor (根据 reduction 可能是标量或与 logits 同形状)
 * @param logits 预测值 Tensor
 * @param target 标签 Tensor
 * @param weight 样本权重 Tensor (可选)
 * @param pos_weight 正类权重 Tensor (可选)
 * @param reduction_str 归约方式 ("none", "mean", "sum")
 */
void calculate(Tensor out, Tensor logits, Tensor target, Tensor weight, Tensor pos_weight, std::string reduction_str) {
    // 1. 将字符串归约参数转换为底层 API 使用的枚举值
    infiniopReduction_t reduction;
    if (reduction_str == "none") {
        reduction = INFINIOP_REDUCTION_NONE;
    } else if (reduction_str == "mean") {
        reduction = INFINIOP_REDUCTION_MEAN;
    } else if (reduction_str == "sum") {
        reduction = INFINIOP_REDUCTION_SUM;
    } else {
        throw std::runtime_error("Unknown reduction mode: " + reduction_str);
    }

    // 2. 生成唯一 Hash Seed 用于缓存查找
    // 包含所有输入 Tensor 的状态和 reduction 参数，确保缓存键的唯一性
    size_t seed = hash_combine(out, logits, target, weight, pos_weight, static_cast<int>(reduction));

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopBCEWithLogitsDescriptor_t desc = nullptr;

    // 3. 如果缓存未命中，创建新的描述符并存入缓存
    if (!desc_opt) {
        // 获取可选 Tensor 的描述符，若未定义则传 nullptr
        auto weight_desc = weight ? weight->desc() : nullptr;
        auto pos_weight_desc = pos_weight ? pos_weight->desc() : nullptr;

        INFINICORE_CHECK_ERROR(infiniopCreateBCEWithLogitsDescriptor(
            context::getInfiniopHandle(device),
            &desc,
            out->desc(),
            logits->desc(),
            target->desc(),
            weight_desc,
            pos_weight_desc,
            reduction));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    // 4. 动态获取并分配 Workspace 临时内存
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetBCEWithLogitsWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    // 5. 获取数据指针，处理可选 Tensor 的空指针逻辑
    const void *weight_ptr = weight ? weight->data() : nullptr;
    const void *pos_weight_ptr = pos_weight ? pos_weight->data() : nullptr;

    // 6. 执行底层算子
    INFINICORE_CHECK_ERROR(infiniopBCEWithLogits(
        desc,
        workspace->data(),
        workspace_size,
        out->data(),
        logits->data(),
        target->data(),
        weight_ptr,
        pos_weight_ptr,
        context::getStream()));
}

// 7. 自动注册到调度器 (Dispatcher)
static bool registered = []() {
    BinaryCrossEntropyWithLogits::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::bce_logits_impl::infiniop
