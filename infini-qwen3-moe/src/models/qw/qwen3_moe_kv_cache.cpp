#include "qwen3_moe_impl.hpp"
#include "qwen3_moe_weight.hpp"
#include "../../tensor.hpp"
#include "../../utils.hpp"

// Helper function declarations
void qwen3_moe_device_worker(
    DeviceQwen3MoeResource &resource,
    Qwen3MoeInferState &state,
    const Qwen3MoeMeta &meta,
    Qwen3MoeInferRequest &req,
    size_t device_id,
    size_t ndev);

// Helper function to get size of data type (if not available elsewhere)
inline size_t sizeof_dtype(infiniDtype_t dtype) {
    switch (dtype) {
        case INFINI_DTYPE_F16:
        case INFINI_DTYPE_BF16:
            return 2;
        case INFINI_DTYPE_F32:
        case INFINI_DTYPE_I32:
            return 4;
        case INFINI_DTYPE_F64:
        case INFINI_DTYPE_I64:
            return 8;
        default:
            return 4; // fallback
    }
}

/*
 * Qwen3-MoE KV 缓存实现
 * 
 * KV缓存在MoE模型中的工作原理与标准Transformer相同，
 * 因为MoE只影响FFN层，注意力机制保持不变。
 */

__C struct Qwen3MoeKVCache *
createQwen3MoeKVCache(const struct Qwen3MoeModel *model) {
    auto cache = new Qwen3MoeKVCache();
    
    const auto &meta = model->meta;
    size_t ndev = model->ndev;
    size_t nlayer = meta.nlayer;
    size_t nkvh = meta.nkvh;
    size_t dh = meta.dh;
    size_t dctx = meta.dctx;
    size_t nkvh_per_dev = nkvh / ndev;
    
    // 为每个设备和每层初始化KV缓存
    cache->k.resize(ndev);
    cache->v.resize(ndev);
    
    for (size_t idev = 0; idev < ndev; ++idev) {
        cache->k[idev].resize(nlayer);
        cache->v[idev].resize(nlayer);
        
        // 设置设备上下文
        RUN_INFINI(infinirtSetDevice(model->dev_resources[idev].device_id));
        
        for (size_t layer = 0; layer < nlayer; ++layer) {
            // Key cache: [dctx, nkvh_per_dev, dh]
            cache->k[idev][layer] = Tensor::create(
                {dctx, nkvh_per_dev, dh},
                meta.dt_logits,
                model->dev_resources[idev].device
            );
            
            // Value cache: [dctx, nkvh_per_dev, dh]
            cache->v[idev][layer] = Tensor::create(
                {dctx, nkvh_per_dev, dh},
                meta.dt_logits,
                model->dev_resources[idev].device
            );
            
            // 初始化为零
            infiniConstant(model->dev_resources[idev].handle,
                          cache->k[idev][layer].get(), 0.0f);
            infiniConstant(model->dev_resources[idev].handle,
                          cache->v[idev][layer].get(), 0.0f);
        }
    }
    
    return cache;
}

__C struct Qwen3MoeKVCache *
duplicateQwen3MoeKVCache(const struct Qwen3MoeModel *model,
                        const struct Qwen3MoeKVCache *src_cache,
                        uint32_t seq_len) {
    auto cache = new Qwen3MoeKVCache();
    
    const auto &meta = model->meta;
    size_t ndev = model->ndev;
    size_t nlayer = meta.nlayer;
    size_t nkvh = meta.nkvh;
    size_t dh = meta.dh;
    size_t dctx = meta.dctx;
    size_t nkvh_per_dev = nkvh / ndev;
    
    // 为每个设备和每层复制KV缓存
    cache->k.resize(ndev);
    cache->v.resize(ndev);
    
    for (size_t idev = 0; idev < ndev; ++idev) {
        cache->k[idev].resize(nlayer);
        cache->v[idev].resize(nlayer);
        
        // 设置设备上下文
        RUN_INFINI(infinirtSetDevice(model->dev_resources[idev].device_id));
        
        for (size_t layer = 0; layer < nlayer; ++layer) {
            // 创建新的缓存张量
            cache->k[idev][layer] = Tensor::create(
                {dctx, nkvh_per_dev, dh},
                meta.dt_logits,
                model->dev_resources[idev].device
            );
            cache->v[idev][layer] = Tensor::create(
                {dctx, nkvh_per_dev, dh},
                meta.dt_logits,
                model->dev_resources[idev].device
            );
            
            if (src_cache && seq_len > 0) {
                // 复制前seq_len个位置的数据
                size_t copy_size = seq_len * nkvh_per_dev * dh * sizeof_dtype(meta.dt_logits);
                
                RUN_INFINI(infinirtMemcpy(
                    cache->k[idev][layer]->data(),
                    src_cache->k[idev][layer]->data(),
                    copy_size,
                    INFINIRT_MEMCPY_D2D
                ));
                
                RUN_INFINI(infinirtMemcpy(
                    cache->v[idev][layer]->data(),
                    src_cache->v[idev][layer]->data(),
                    copy_size,
                    INFINIRT_MEMCPY_D2D
                ));
            } else {
                // 初始化为零
                infiniConstant(model->dev_resources[idev].handle,
                              cache->k[idev][layer].get(), 0.0f);
                infiniConstant(model->dev_resources[idev].handle,
                              cache->v[idev][layer].get(), 0.0f);
            }
        }
    }
    
    return cache;
}

__C void dropQwen3MoeKVCache(const struct Qwen3MoeModel *model,
                            struct Qwen3MoeKVCache *cache) {
    if (!cache) return;
    
    // 清理所有KV缓存张量
    for (auto &dev_cache : cache->k) {
        dev_cache.clear();
    }
    for (auto &dev_cache : cache->v) {
        dev_cache.clear();
    }
    
    delete cache;
}

/*
 * Qwen3-MoE 模型构造函数实现
 * 
 * 初始化分布式MoE模型，包括权重分区和工作线程设置
 */
Qwen3MoeModel::Qwen3MoeModel(const Qwen3MoeMeta *meta,
                           const Qwen3MoeWeights *weights,
                           infiniDevice_t device,
                           std::vector<int> device_ids) 
    : meta(*meta), ndev(device_ids.size()) {
    
    /*
     * 1. 初始化设备资源
     */
    dev_resources.resize(ndev);
    states.resize(ndev);
    threads.resize(ndev);
    
    /*
     * 2. 为每个设备分区权重和创建资源
     */
    for (size_t idev = 0; idev < ndev; ++idev) {
        auto &resource = dev_resources[idev];
        resource.device = device;
        resource.device_id = device_ids[idev];
        
        // 设置设备上下文
        RUN_INFINI(infinirtSetDevice(resource.device_id));
        
        /*
         * 2.1 全局权重（在所有设备上复制）
         */
        resource.w_in_embd = getQwen3MoeInEmbd(meta, weights);
        resource.w_out_norm = getQwen3MoeOutNorm(meta, weights);
        resource.w_out_embd = getQwen3MoeOutEmbd(meta, weights);
        
        // 创建RoPE表
        auto rope_shape = std::vector<size_t>({meta->dctx, meta->dh / 2});
        resource.sin_table = Tensor::create(rope_shape, meta->dt_logits, device);
        resource.cos_table = Tensor::create(rope_shape, meta->dt_logits, device);
        
        // 初始化RoPE表 (简化实现)
        // 实际应该根据theta计算正弦和余弦值
        infiniConstant(resource.handle, resource.sin_table.get(), 0.0f);
        infiniConstant(resource.handle, resource.cos_table.get(), 1.0f);
        
        /*
         * 2.2 逐层权重分区
         */
        resource.w_attn_norm.resize(meta->nlayer);
        resource.w_attn_q_norm.resize(meta->nlayer);
        resource.w_attn_k_norm.resize(meta->nlayer);
        resource.w_attn_q.resize(meta->nlayer);
        resource.w_attn_k.resize(meta->nlayer);
        resource.w_attn_v.resize(meta->nlayer);
        resource.w_attn_o.resize(meta->nlayer);
        resource.w_mlp_norm.resize(meta->nlayer);
        resource.w_mlp_gate.resize(meta->nlayer);
        resource.w_mlp_up.resize(meta->nlayer);
        resource.w_mlp_down.resize(meta->nlayer);
        resource.w_moe_gate.resize(meta->nlayer);
        resource.w_moe_experts_gate.resize(meta->nlayer);
        resource.w_moe_experts_up.resize(meta->nlayer);
        resource.w_moe_experts_down.resize(meta->nlayer);
        resource.is_moe_layer.resize(meta->nlayer);
        resource.expert_usage_count.resize(meta->nlayer);
        
        for (size_t layer = 0; layer < meta->nlayer; ++layer) {
            // 注意力权重
            resource.w_attn_norm[layer] = getQwen3MoeAttnNorm(meta, weights, layer);
            resource.w_attn_q_norm[layer] = getQwen3MoeAttnQNorm(meta, weights, layer);
            resource.w_attn_k_norm[layer] = getQwen3MoeAttnKNorm(meta, weights, layer);
            resource.w_attn_q[layer] = getQwen3MoeAttnQ(meta, weights, layer, idev, ndev);
            resource.w_attn_k[layer] = getQwen3MoeAttnK(meta, weights, layer, idev, ndev);
            resource.w_attn_v[layer] = getQwen3MoeAttnV(meta, weights, layer, idev, ndev);
            resource.w_attn_o[layer] = getQwen3MoeAttnO(meta, weights, layer, idev, ndev);
            
            // MLP归一化权重
            resource.w_mlp_norm[layer] = getQwen3MoeMLPNorm(meta, weights, layer);
            
            // 确定层类型（MoE还是普通MLP）
            resource.is_moe_layer[layer] = isQwen3MoeMoeLayer(meta, layer);
            
            if (resource.is_moe_layer[layer]) {
                // MoE层权重
                resource.w_moe_gate[layer] = getQwen3MoeMoeGate(meta, weights, layer);
                
                // 分配专家到设备
                size_t experts_per_device = (meta->num_experts + ndev - 1) / ndev;
                size_t start_expert = idev * experts_per_device;
                size_t end_expert = std::min(start_expert + experts_per_device, meta->num_experts);
                size_t num_experts_on_device = end_expert - start_expert;
                
                resource.w_moe_experts_gate[layer].resize(num_experts_on_device);
                resource.w_moe_experts_up[layer].resize(num_experts_on_device);
                resource.w_moe_experts_down[layer].resize(num_experts_on_device);
                
                for (size_t local_expert = 0; local_expert < num_experts_on_device; ++local_expert) {
                    size_t global_expert = start_expert + local_expert;
                    resource.w_moe_experts_gate[layer][local_expert] = 
                        getQwen3MoeMoeExpertGate(meta, weights, layer, global_expert, idev, ndev);
                    resource.w_moe_experts_up[layer][local_expert] = 
                        getQwen3MoeMoeExpertUp(meta, weights, layer, global_expert, idev, ndev);
                    resource.w_moe_experts_down[layer][local_expert] = 
                        getQwen3MoeMoeExpertDown(meta, weights, layer, global_expert, idev, ndev);
                }
                
                // 初始化专家使用统计
                resource.expert_usage_count[layer].resize(meta->num_experts, 0);
            } else {
                // 普通MLP层权重
                resource.w_mlp_gate[layer] = getQwen3MoeMLPGate(meta, weights, layer, idev, ndev);
                resource.w_mlp_up[layer] = getQwen3MoeMLPUp(meta, weights, layer, idev, ndev);
                resource.w_mlp_down[layer] = getQwen3MoeMLPDown(meta, weights, layer, idev, ndev);
            }
        }
        
        /*
         * 2.3 计算缓冲区
         */
        size_t max_batch_tokens = 4096;  // 可配置的最大批处理tokens
        
        resource.hidden_states = Tensor::create({max_batch_tokens, meta->d}, meta->dt_logits, device);
        resource.residual_states = Tensor::create({max_batch_tokens, meta->d}, meta->dt_logits, device);
        resource.attn_output = Tensor::create({max_batch_tokens, meta->d}, meta->dt_logits, device);
        resource.mlp_output = Tensor::create({max_batch_tokens, meta->d}, meta->dt_logits, device);
        
        // MoE特定的缓冲区
        if (meta->num_experts > 0) {
            resource.router_logits = Tensor::create({max_batch_tokens, meta->num_experts}, meta->dt_logits, device);
            resource.routing_weights = Tensor::create({max_batch_tokens, meta->num_experts_per_tok}, meta->dt_logits, device);
            resource.selected_experts = Tensor::create({max_batch_tokens, meta->num_experts_per_tok}, INFINI_DTYPE_I32, device);
            resource.expert_outputs = Tensor::create({max_batch_tokens, meta->d}, meta->dt_logits, device);
            
            size_t max_tokens_per_expert = max_batch_tokens;  // 保守估计
            resource.expert_intermediate = Tensor::create({max_tokens_per_expert, meta->moe_intermediate_size / ndev}, meta->dt_logits, device);
        }
    }
    
    /*
     * 3. 启动工作线程
     */
    for (size_t idev = 0; idev < ndev; ++idev) {
        // 初始化同步状态
        states[idev].loaded = false;
        states[idev].proceed = false;
        states[idev].shutdown = false;
        
        // 启动工作线程
        threads[idev] = std::thread([this, idev]() {
            qwen3_moe_device_worker(
                dev_resources[idev],
                states[idev],
                meta,
                req,
                idev,
                ndev
            );
        });
    }
    
    /*
     * 4. 等待所有设备初始化完成
     */
    for (size_t idev = 0; idev < ndev; ++idev) {
        std::unique_lock<std::mutex> lock(states[idev].mtx);
        states[idev].cv_load.wait(lock, [&] { return states[idev].loaded; });
    }
    
    printf("Qwen3-MoE model initialized with %zu devices, %zu experts per layer\n", 
           ndev, meta->num_experts);
}