#include "Qwen3MoE_impl.hpp"
#include "infinicore_infer.h"

#include "../../tensor.hpp"
#include "../../utils.hpp"
// 注意：Qwen3MoECache 在头文件中声明，但实际使用 Qwen3Cache
// 这里假设它们是同一个类型，或者Qwen3MoECache是Qwen3Cache的typedef

/// @brief 创建KVCache
__C __export struct Qwen3Cache *
createQwen3Cache(const Qwen3MoEAttentionMeta *meta,
                    size_t batch_size, size_t seq_len) {
    Qwen3Cache *cache = new Qwen3Cache();
    
    // 假设只有1层attention（因为只实现attention模块）
    size_t nlayer = 1;
    size_t max_seq_len = meta->max_seq_len;
    size_t num_kv_head = meta->num_kv_head;
    size_t head_dim = meta->head_dim;
    
    // 为每一层创建K和V cache
    // Cache shape: [num_kv_head, max_seq_len, head_dim]
    cache->layers.resize(nlayer);
    
    for (size_t layer = 0; layer < nlayer; layer++) {
        // 创建K cache: [num_kv_head, max_seq_len, head_dim]
        auto k_cache = Tensor::buffer(meta->dtype, {num_kv_head, max_seq_len, head_dim});
        
        // 创建V cache: [num_kv_head, max_seq_len, head_dim]
        auto v_cache = Tensor::buffer(meta->dtype, {num_kv_head, max_seq_len, head_dim});
        
        cache->layers[layer] = std::make_pair(k_cache, v_cache);
    }
    
    return reinterpret_cast<struct Qwen3Cache *>(cache);
}

/// @brief 销毁KVCache（如果需要的话，可以添加这个函数）
// 注意：头文件中没有声明这个函数，如果需要可以添加
// __C void dropQwen3Cache(struct Qwen3Cache *cache) {
//     if (cache) {
//         Qwen3Cache *qwen3_cache = reinterpret_cast<Qwen3Cache *>(cache);
//         for (auto &layer : qwen3_cache->layers) {
//             layer.first.reset();
//             layer.second.reset();
//         }
//         delete qwen3_cache;
//     }
// }

