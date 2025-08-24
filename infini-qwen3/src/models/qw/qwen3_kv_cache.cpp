/*
 * KV 缓存管理实现
 * 
 * 此文件实现了用于自回归 transformer 推理的高效键值缓存操作。
 * KV 缓存存储过去的注意力键和值以避免在顺序 token 生成期间重新计算。
 * 
 * 主要特性：
 * - 多设备分布式缓存存储
 * - 高效的内存分配和复制
 * - 用于 beam search 的缓存共享和复制
 * - 正确的资源清理和内存管理
 * 
 * 缓存组织：
 * - 为张量并行在设备间分区
 * - 每个 transformer 层的单独存储
 * - 高效访问模式的连续内存布局
 */

#include "qwen3_impl.hpp"
/*
 * 为新推理请求创建 KV 缓存
 * 
 * 为新推理请求分配新鲜的键值缓存存储。
 * 为每个设备和 transformer 层创建单独的缓存张量
 * 以支持具有张量并行的分布式推理。
 * 
 * 缓存布局：
 * - cache->k[idev][layer]：设备 idev，层 layer 的键缓存
 * - cache->v[idev][layer]：设备 idev，层 layer 的值缓存  
 * - 每个张量形状：[max_len, nkvh/ndev, dh]
 *   - max_len：最大序列长度（模型上下文限制）
 *   - nkvh/ndev：每设备的键值头（在设备间分布）
 *   - dh：头维度
 * 
 * 内存分配：
 * - 使用设备特定的内存分配以获得最佳性能
 * - 预先分配最大容量以避免在生成期间重新分配
 * - 零初始化内存确保干净的缓存状态
 * 
 * 参数：
 * - model：包含设备和架构信息的 JiugeModel 实例
 * 
 * 返回：指向新分配的 KVCache 结构的指针
 */
__C struct Qwen3KVCache *createQwen3KVCache(const Qwen3Model *model) {
    // 创建新的 KVCache 实例
    Qwen3KVCache *cache = new Qwen3KVCache();
    
    // 提取模型维度以确定缓存大小
    auto ndev = model->dev_resources.size();          // 设备数
    auto nkvh = model->meta.nkvh / ndev;             // 每设备的 KV 头数  
    auto max_len = model->meta.dctx;                 // 最大上下文长度
    auto dh = model->meta.dh;                        // 头维度
    
    // 定义缓存张量形状：[max_len, nkvh_per_device, dh]
    auto shape = std::vector<size_t>{max_len, nkvh, dh};
    
    /*
     * 为每个设备分配缓存张量
     * 
     * 对于分布式推理，每个设备存储对应其分配的注意力头的 KV 缓存分区。
     * 这支持跨设备的并行缓存更新和注意力计算。
     */
    for (unsigned int idev = 0; idev < ndev; idev++) {
        // 设置内存分配的设备上下文
        RUN_INFINI(infinirtSetDevice(model->device, model->dev_ids[idev]));
        
        // 为此设备创建每层缓存存储
        auto kcache = std::vector<std::shared_ptr<Tensor>>();  // 每层的键缓存
        auto vcache = std::vector<std::shared_ptr<Tensor>>();  // 每层的值缓存
        
        /*
         * 为每个 Transformer 层分配缓存张量
         * 
         * 每层都需要单独的键和值缓存存储。
         * 缓存张量以最大容量分配，以避免
         * 在自回归生成期间重新分配。
         */
        for (unsigned int layer = 0; layer < model->meta.nlayer; layer++) {
            // 分配并初始化为零
// 在createQwen3KVCache中创建缓存后没有显式清零
            kcache.push_back(std::move(Tensor::buffer(model->meta.dt_logits, shape)));
            vcache.push_back(std::move(Tensor::buffer(model->meta.dt_logits, shape)));
        }
        
        // 存储设备特定的缓存数组
        cache->k.push_back(kcache);
        cache->v.push_back(vcache);
    }

    return cache;
}

/*
 * 复制 KV 缓存用于 Beam Search 或分支
 * 
 * 创建现有 KV 缓存的副本，包含前 seq_len 个 token。
 * 这对于 beam search 很有用，其中多个生成路径共享
 * 一个公共前缀但在某一点上分歧。
 * 
 * 使用场景：
 * - Beam search：在分支点复制缓存
 * - 推测解码：创建候选生成分支
 * - 多轮对话：为新轮次复制基础上下文
 * 
 * 复制过程：
 * 1. 创建与原始缓存相同结构的新缓存
 * 2. 从每个缓存张量复制前 seq_len 个 token
 * 3. 为未来 token 生成保留剩余容量
 * 
 * 内存效率：
 * - 仅复制缓存的活动部分（seq_len 个 token）
 * - 保留缓存结构和设备分布
 * - 使用高效的设备到设备内存复制操作
 * 
 * 参数：
 * - model：用于缓存结构的 JiugeModel 实例
 * - kv_cache：要复制的源缓存
 * - seq_len：要从源缓存复制的 token 数
 * 
 * 返回：指向具有复制数据的新创建缓存的指针
 */
__C struct Qwen3KVCache *duplicateQwen3KVCache(const Qwen3Model *model,
                                     const Qwen3KVCache *kv_cache,
                                     unsigned int seq_len) {
    // 创建与原始缓存相同结构的新缓存
    auto new_kv_cache = createQwen3KVCache(model);

    // 提取模型维度以计算复制大小
    auto ndev = model->dev_resources.size();          // 设备数
    auto nkvh = model->meta.nkvh / ndev;             // 每设备的 KV 头数
    auto dh = model->meta.dh;                        // 头维度  
    auto dt_size = dsize(model->meta.dt_logits);     // 数据类型大小（字节）
    
    /*
     * 跨所有设备和层复制缓存数据
     * 
     * 遍历每个设备和层以复制缓存的活动部分
     * （前 seq_len 个 token）。这保留了分布式
     * 结构，同时仅复制相关数据。
     */
    for (unsigned int idev = 0; idev < ndev; idev++) {
        // 设置内存操作的设备上下文
        RUN_INFINI(infinirtSetDevice(model->device, model->dev_ids[idev]));
        
        /*
         * 复制每层缓存数据
         * 
         * 对于每个 transformer 层，复制键和值缓存。
         * 内存复制大小：seq_len * nkvh * dh * data_type_size
         * 
         * 缓存内存布局：[seq_len, nkvh, dh] 连续存储
         * 复制操作：高效的设备到设备内存传输
         */
        for (unsigned int layer = 0; layer < model->meta.nlayer; layer++) {
            /*
             * 复制键缓存：[seq_len, nkvh, dh] 元素
             * 
             * 源：kv_cache->k[idev][layer]（原始缓存）
             * 目标：new_kv_cache->k[idev][layer]（新缓存）
             * 大小：seq_len * nkvh * dh * sizeof(data_type)
             */
            RUN_INFINI(infinirtMemcpy(new_kv_cache->k[idev][layer]->data(),   // 目标
                                      kv_cache->k[idev][layer]->data(),      // 源
                                      seq_len * nkvh * dh * dt_size,         // 复制大小（字节）
                                      INFINIRT_MEMCPY_D2D));                 // 设备到设备复制
            
            /*
             * 复制值缓存：[seq_len, nkvh, dh] 元素
             * 
             * 与键缓存复制相同的结构和大小。
             */                          
            RUN_INFINI(infinirtMemcpy(new_kv_cache->v[idev][layer]->data(),   // 目标
                                      kv_cache->v[idev][layer]->data(),      // 源  
                                      seq_len * nkvh * dh * dt_size,         // 复制大小（字节）
                                      INFINIRT_MEMCPY_D2D));                 // 设备到设备复制
        }
    }
    
    return new_kv_cache;
}

/*
 * 销毁 KV 缓存并释放内存
 * 
 * 正确地释放 KV 缓存并释放所有关联的设备内存。
 * 此函数确保干净的内存管理并防止内存泄漏
 * 在长时间运行的推理应用程序中。
 * 
 * 清理过程：
 * 1. 为每个设备设置设备上下文
 * 2. 使用 shared_ptr 重置释放张量内存（自动释放）
 * 3. 释放 KVCache 结构本身
 * 
 * 内存安全：
 * - 通过 shared_ptr 使用 RAII 原则进行自动内存管理
 * - 确保在内存操作前正确设置设备上下文
 * - 通过正确的引用计数防止双重释放错误
 * 
 * 参数：
 * - model：用于设备上下文的 JiugeModel 实例
 * - kv_cache：要销毁和释放的缓存
 */
__C void dropQwen3KVCache(Qwen3Model const *model, Qwen3KVCache *kv_cache) {
    auto ndev = model->dev_resources.size();
    
    /*
     * 为每个设备和层释放缓存张量
     * 
     * 遍历所有设备和层以正确释放
     * 使用 shared_ptr 引用计数的张量内存。
     */
    for (unsigned int idev = 0; idev < ndev; idev++) {
        // 设置内存操作的设备上下文
        RUN_INFINI(infinirtSetDevice(model->device, model->dev_ids[idev]));
        
        /*
         * 释放每层缓存张量
         * 
         * 在 shared_ptr 上调用 reset() 以减少引用计数。
         * 当引用计数达到零时，张量内存会自动释放。
         */
        for (unsigned int layer = 0; layer < model->meta.nlayer; layer++) {
            kv_cache->k[idev][layer].reset();  // 释放键缓存张量
            kv_cache->v[idev][layer].reset();  // 释放值缓存张量
        }
    }
    
    // 释放 KVCache 结构本身
    delete kv_cache;
}