#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../inference_context.hpp"
#include "../../cache.hpp"
#include "infinicore_infer.h"

#include "llada_impl.hpp"
#include "llada_weight.hpp"

#include <random>
#include <thread>
#include <vector>
#include <algorithm>


// #TODO:这个是草稿版
void createDeviceResource(LLaDADeviceResource *rsrc, const LLaDAMeta * meta,
                          const LLaDAWeights *weights, infiniDevice_t device, int idev,
                          int ndev, int dev_id,
                          infinicclComm_t comm){
    std::cout << "Set Device" << std::endl;
    //Print(meta);
    RUN_INFINI(infinirtSetDevice(device, dev_id));
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle);
    infinirtStream_t stream;
    infinirtStreamCreate(&stream);

    std::cout << "Set Weight" << std::endl; // 逐层获取权重
    std::vector<std::shared_ptr<Tensor>> w_attn_norm, w_attn_qkv, b_attn_qkv, w_attn_q_norm, w_attn_k_norm, w_attn_out,
        w_ffn_norm, w_ffn_gate_up, w_ffn_down;
    for (size_t layer = 0; layer < meta->nlayer; layer++) {
        w_attn_norm.push_back(
            getAttnNorm(meta, weights, layer));
        w_attn_qkv.push_back(
            getAttnQKV(meta, weights, layer, idev, ndev));
        if (weights->attn_qkv_b != nullptr) {
            b_attn_qkv.push_back(
                getAttnQKVBias(meta, weights, layer, idev, ndev));
        }

        if (weights->attn_q_norm != nullptr) {
            w_attn_q_norm.push_back(
                getAttnQNorm(meta, weights, layer));
            w_attn_k_norm.push_back(
                getAttnKNorm(meta, weights, layer));
        }
        w_attn_out.push_back(
            getAttnO(meta, weights, layer, idev, ndev));
        w_ffn_norm.push_back(
            getFFNNorm(meta, weights, layer));
        w_ffn_gate_up.push_back(
            getFFNGateUp(meta, weights, layer, idev, ndev));
        w_ffn_down.push_back(
            getFFNDown(meta, weights, layer, idev, ndev));
    }

    std::cout << "Set Memory Pool" << std::endl;
    auto memory_pool = std::make_shared<MemoryPool>(128 * 1024 * 1024);

    std::cout << "Set LLaDADeviceResource" << std::endl;
    *rsrc = LLaDADeviceResource{
        .device = device,
        .device_id = dev_id,
        .handle = handle,

        .w_in_embd = getInEmbd(meta, weights),
        .w_out_norm = getOutNorm(meta, weights),
        .w_out_embd = getOutEmbd(meta, weights),
        .sin_table = getSinTable(meta),
        .cos_table = getCosTable(meta),

        .w_attn_norm = w_attn_norm,
        .w_attn_qkv = w_attn_qkv,
        .b_attn_qkv = b_attn_qkv,
        .w_attn_q_norm = w_attn_q_norm,
        .w_attn_k_norm = w_attn_k_norm,
        .w_attn_out = w_attn_out,
        .w_ffn_norm = w_ffn_norm,
        .w_ffn_gate_up = w_ffn_gate_up,
        .w_ffn_down = w_ffn_down,

        .stream = stream,
        .comm = comm,
        .memory_pool = memory_pool,
    };
    std::cout << "Over LLaDADeviceResource" << std::endl;
    RUN_INFINI(infinirtDeviceSynchronize());
}

void releaseDeviceResource(LLaDADeviceResource &rsrc){
    std::cout << "Release" << std::endl;
}

__C void
inferBatchLLaDA(struct LLaDAModel *model,
                const uint32_t *tokens, uint32_t ntok,
                const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                struct KVCache **kv_caches,
                const float *temperature, const uint32_t *topk, const float *topp,
                uint32_t *output) {
    std::cout << "[DEBUG] inferBatchLLaDA called with single-threaded mode" << std::endl;

    // Set request data
    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.kv_caches = kv_caches;
    model->req.output = output;
    model->req.logits = nullptr;
    model->req.temperature = temperature;
    model->req.topk = topk;
    model->req.topp = topp;

    // Single-threaded implementation - directly call inference
    if (model->dev_ids.empty()) {
        std::cerr << "[ERROR] No devices available!" << std::endl;
        return;
    }

    // Use the first device (single-threaded)
    int idev = 0;
    int ndev = 1;
    int dev_id = model->dev_ids[idev];

    std::cout << "[DEBUG] Using device " << dev_id << " for inference" << std::endl;

    // Create device resource (temporary for single-threaded call)
    LLaDADeviceResource rsrc;

    // Create communication handle (single device, no comm)
    infinicclComm_t comm = nullptr;

    try {
        // Direct call to inference function using model's device
        launchDevice(model->meta, model->weights, &rsrc, model->states[idev], model->req,
                    model->device, idev, ndev, dev_id, comm);

        std::cout << "[DEBUG] Inference completed successfully" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Inference failed: " << e.what() << std::endl;
    }
}

void inferDeviceBatch(const LLaDAMeta &meta, LLaDADeviceResource &rsrc,
                      uint32_t idev, uint32_t ndev,
                      const uint32_t *tokens, uint32_t ntok,
                      const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                      struct KVCache **kv_caches,
                      const float *temperature, const uint32_t *topk, const float *topp,
                      uint32_t *output, void *last_logits){
    auto nlayer = meta.nlayer;
    auto nkvh = meta.nkvh / ndev;      // 每个设备的KV头数 (GQA分组查询)
    auto nh = meta.nh / ndev;          // 每个设备的注意力头数
    auto ngroup = nh / nkvh;           // 每个设备的组数 (多少个Q头共享一个K/V头)
    auto dctx = meta.dctx;             // 最大上下文长度
    auto dh = meta.dh;                 // 每个头的维度
    auto d = meta.d;                   // 模型隐藏层维度
    auto dt_logits = meta.dt_logits;   // 输出数据类型
    auto di_dense = meta.di_dense / ndev;  // 每个设备的密集FFN中间维度
    auto di_expert = meta.di_expert / ndev; // 每个设备的专家FFN中间维度
    auto dvoc = meta.dvoc;             // 词汇表大小
    auto stream = rsrc.stream;
    bool has_qkv_bias = rsrc.b_attn_qkv.size() > 0;      // 是否有QKV偏置
    bool has_qk_norm = rsrc.w_attn_q_norm.size() > 0 && rsrc.w_attn_k_norm.size() > 0;  // 是否有QK归一化

    // ====================================================================
    // 1. 输入预处理和Token嵌入
    // ====================================================================

    // 1.1 位置编码准备 - 生成序列中每个token的位置ID
    // requests = [
    // "Hello world",      # req_idx=0, 长度=2, 起始位置=0
    // "How are you",      # req_idx=1, 长度=3, 起始位置=2  
    // "Fine thanks",      # req_idx=2, 长度=2, 起始位置=5
    // ]

    // # 输入参数:
    // req_pos = [0, 2, 5]     # 每个请求的起始位置
    // req_lens = [2, 3, 2]    # 每个请求的长度
    // nreq = 3                # 3个请求
    // ntok = 7                # 总共7个token

    // # 执行后结果:
    // batch_pos_ids = [0, 1, 2, 3, 4, 5, 6]
    auto batch_pos_ids = std::vector<uint32_t>(ntok);
    // 根据请求信息填充位置ID (req_pos指示每个请求的起始位置)
    std::cout << "PreFill Id " << "req len is " << nreq << std::endl;
    for (uint32_t req_idx = 0; req_idx < nreq; ++req_idx) {
        uint32_t start_pos = req_pos[req_idx];
        uint32_t req_len = req_lens[req_idx];
        for (uint32_t i = 0; i < req_len; ++i) {
            batch_pos_ids[start_pos + i] = start_pos + i;
        }
    }

    // 1.2 将位置ID复制到GPU (如果是GPU设备)
    std::shared_ptr<Tensor> pos_ids_buf;
    if (rsrc.device == INFINI_DEVICE_CPU) {
        pos_ids_buf = Tensor::weight(batch_pos_ids.data(), INFINI_DTYPE_U32, {ntok});
    } else { // GPU
        pos_ids_buf = Tensor::buffer(INFINI_DTYPE_U32, {ntok}, rsrc.memory_pool);
        RUN_INFINI(infinirtMemcpyAsync(pos_ids_buf->data(), batch_pos_ids.data(), sizeof(uint32_t) * ntok,
                                       INFINIRT_MEMCPY_H2D, stream));
    }

    // 1.3 Token嵌入查找 - 将token ID转换为对应的嵌入向量
    // 输入: tokens[token_count] -> 输出: [token_count, hidden_dim]
    std::shared_ptr<Tensor> hidden_states = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    for (uint32_t i = 0; i < ntok; i++) {
        // 查找词嵌入表: [vocab_size, d] -> 输出token i的嵌入向量
        RUN_INFINI(infinirtMemcpyAsync(hidden_states->data(i * d),
                                       rsrc.w_in_embd->data(tokens[i] * d),
                                       dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
    }

    // ====================================================================
    // 2. 位置编码应用 - RoPE (Rotary Position Embedding)
    // ====================================================================

    // 2.1 从预计算的RoPE表中获取当前位置的正弦余弦值
    // sin_table: [max_seq_len, head_dim/2], cos_table: [max_seq_len, head_dim/2]
    std::shared_ptr<Tensor> sin_cos_for_pos = Tensor::buffer(dt_logits, {ntok, dh}, rsrc.memory_pool);
    // TODO: 根据batch_pos_ids从sin_table和cos_table中提取对应的正弦余弦值

    // ====================================================================
    // 3. Transformer层处理循环
    // ====================================================================

    // 设置推理上下文 - 必须在调用rmsnorm之前

    CacheManager cache_manager(100);
    InferenceContext ctx(rsrc.handle, rsrc.memory_pool, &cache_manager, rsrc.stream);
    setInferenceContext(&ctx);
    std::cout << "[DEBUG] InferenceContext created and set successfully" << std::endl;

    // 保存每层的输入，用于残差连接
    std::shared_ptr<Tensor> layer_input = hidden_states;

    for (uint32_t layer_idx = 0; layer_idx < nlayer; ++layer_idx) {
        std::cout << "Processing layer " << layer_idx << std::endl;

        // ====================================================================
        // 3.1 注意力前归一化 (Input Layernorm)
        // ====================================================================
       std::cout << "Input LayerNorm" << std::endl;
        // RMSNorm: hidden_states = hidden_states * rms_norm_weight / sqrt(mean(hidden_states^2) + eps)
        std::shared_ptr<Tensor> attn_input = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
        // 实现RMSNorm归一化 - 参考jiuge.cpp:218
        std::cout << "Layer Idx " << layer_idx << " Do rmsnorm" << std::endl;

        // 检查权重是否为空
        if (!rsrc.w_attn_norm[layer_idx]) {
            std::cerr << "ERROR: rsrc.w_attn_norm[" << layer_idx << "] is null!" << std::endl;
            return;
        }
        if (!layer_input) {
            std::cerr << "ERROR: layer_input is null!" << std::endl;
            return;
        }
        if (!attn_input) {
            std::cerr << "ERROR: attn_input is null!" << std::endl;
            return;
        }
        if (!rsrc.w_attn_norm[layer_idx]->data()) {
            std::cerr << "ERROR: rsrc.w_attn_norm[" << layer_idx << "] data pointer is null!" << std::endl;
            return;
        }

        std::cout << "All pointers are valid, calling rmsnorm..." << std::endl;
        rmsnorm(attn_input, layer_input, rsrc.w_attn_norm[layer_idx], meta.epsilon);

        // ====================================================================
        // 3.2 QKV线性投影
        // ====================================================================
        // 输入: [ntok, d] -> Q: [ntok, nh, dh], K: [ntok, nkvh, dh], V: [ntok, nkvh, dh]
        std::cout << "Initlilizing qkv buffer " << std::endl;
        std::shared_ptr<Tensor> qkv_buf = Tensor::buffer(dt_logits, {ntok, (nh + nkvh * 2) * dh}, rsrc.memory_pool);

        
        // 执行线性投影: [ntok, d] × [d, (nh + 2*nkvh)*dh] = [ntok, (nh + 2*nkvh)*dh]
        // TODO: 实现QKV矩阵乘法
        // RUN_INFINI(infiniopGemm(..., attn_input, rsrc.w_attn_qkv[layer_idx], qkv_buf, ...));
        std::cout << "Linear project qkv matrix" << std::endl;
        linear(qkv_buf, attn_input, rsrc.w_attn_qkv[layer_idx], 1.0, 0.0, nullptr, nullptr);
        // 如果有QKV偏置，加上偏置
        // if (has_qkv_bias) {
        //     // TODO: 实现偏置加法
        //     // RUN_INFINI(infiniopAdd(..., qkv_buf, rsrc.b_attn_qkv[layer_idx], qkv_buf, ...));
        // }
        // ====================================================================
        // 3.3 QKV重塑和分离
        // ====================================================================
        std::cout << "Initlilizing qkv split  " << std::endl;
        auto qkv_rope = qkv_buf->view({ntok, nh + nkvh * 2, dh});
        auto q_buf = qkv_rope->slice(1, 0, nh);           // [ntok, nh, dh] - Query部分
        auto k_buf = qkv_rope->slice(1, nh, nkvh);       // [ntok, nkvh, dh] - Key部分
        auto v_buf = qkv_rope->slice(1, nh+nkvh, nkvh);  // [ntok, nkvh, dh] - Value部分

        // ====================================================================
        // 3.4 QK归一化 (可选)
        // ====================================================================
        std::cout << "QK Normalization" << std::endl;
        if (has_qk_norm) {
            // 对Q和K分别应用RMSNorm
            // TODO: 实现QK归一化
            // RUN_INFINI(infiniopRMSNorm(..., q_buf, rsrc.w_attn_q_norm[layer_idx], q_buf, ...));
            // RUN_INFINI(infiniopRMSNorm(..., k_buf, rsrc.w_attn_k_norm[layer_idx], k_buf, ...));
            rmsnorm(q_buf, q_buf, rsrc.w_attn_q_norm[layer_idx], meta.epsilon);
            rmsnorm(k_buf, k_buf, rsrc.w_attn_k_norm[layer_idx], meta.epsilon);
        }

        // ====================================================================
        // 3.5 RoPE位置编码应用
        // ====================================================================
        // 将正弦余弦编码应用到Q和K张量
        // 输入: q_buf: [ntok, nh, dh], k_buf: [ntok, nkvh, dh]
        // 位置编码: sin_cos_for_pos: [ntok, dh]
        // TODO: 实现RoPE旋转位置编码
        std::cout << "Position Embedding" << std::endl;
        rope(q_buf, k_buf, pos_ids_buf, rsrc.sin_table, rsrc.cos_table);
        // 注意：rope函数会同时处理q和k，所以第二个调用可能不需要

        // ====================================================================
        // 3.6 注意力计算
        // ====================================================================

        // 3.6.1 KV缓存处理
        // 将当前的K和V写入到KV缓存中，用于自回归生成
        // TODO: 实现KV缓存写入
        // update_kv_cache(kv_caches, layer_idx, k_buf, v_buf, ...);

        // 3.6.2 读取历史KV (如果有缓存)
        // TODO: 从KV缓存中读取历史K和V
        // [seq_len_cached + ntok, nkvh, dh]

        // 3.6.3 GQA处理 - 重复KV以匹配Q的维度
        // 将[n_tok, nkvh, dh]的K和V重复ngroup次变为[n_tok, nh, dh]
        std::shared_ptr<Tensor> k_repeated = Tensor::buffer(dt_logits, {ntok, nh, dh}, rsrc.memory_pool);
        std::shared_ptr<Tensor> v_repeated = Tensor::buffer(dt_logits, {ntok, nh, dh}, rsrc.memory_pool);
        // TODO: 实现KV重复操作
        // repeat_kv(k_buf, k_repeated, ngroup);
        // repeat_kv(v_buf, v_repeated, ngroup);

        // 3.6.4 注意力权重计算
        // Q @ K^T / sqrt(dh) -> [ntok, nh, seq_len]
        std::shared_ptr<Tensor> attn_weights = Tensor::buffer(dt_logits, {ntok, nh, ntok}, rsrc.memory_pool);
        // TODO: 实现注意力权重计算
        // RUN_INFINI(infiniopBatchedGemm(..., q_buf, k_repeated, attn_weights, ...));
        // attn_weights = attn_weights / sqrt(dh);

        // 3.6.5 注意力归一化
        // TODO: 实现Softmax归一化
        // RUN_INFINI(infiniopSoftmax(..., attn_weights, attn_weights, ...));

        // 3.6.6 注意力输出计算
        // attn_weights @ V -> [ntok, nh, dh]
        std::shared_ptr<Tensor> attn_output = Tensor::buffer(dt_logits, {ntok, nh, dh}, rsrc.memory_pool);
        // TODO: 实现注意力输出计算
        // RUN_INFINI(infiniopBatchedGemm(..., attn_weights, v_repeated, attn_output, ...));

        // 3.6.7 输出投影
        // 将多头输出合并: [ntok, nh, dh] -> [ntok, d]
        // 然后通过输出投影层: [ntok, d] × [d, d] = [ntok, d]
        std::shared_ptr<Tensor> attn_proj = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
        // TODO: 实现注意力输出投影
        // RUN_INFINI(infiniopGemm(..., attn_output, rsrc.w_attn_out[layer_idx], attn_proj, ...));

        // ====================================================================
        // 3.7 残差连接和后注意力归一化
        // ====================================================================
        // 残差连接: layer_input + attn_proj
        std::shared_ptr<Tensor> residual1 = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
        // TODO: 实现残差加法
        // RUN_INFINI(infiniopAdd(..., layer_input, attn_proj, residual1, ...));

        // 后注意力归一化
        std::shared_ptr<Tensor> mlp_input = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
        // TODO: 实现RMSNorm
        // RUN_INFINI(infiniopRMSNorm(..., residual1, rsrc.w_ffn_norm[layer_idx], mlp_input, ...));

        // ====================================================================
        // 3.8 专家混合(MoE)处理
        // ====================================================================

        // 3.8.1 路由器计算
        // 输入: [ntok, d] -> 输出: [ntok, num_experts] (每个token对每个专家的权重)
        std::shared_ptr<Tensor> router_logits = Tensor::buffer(dt_logits, {ntok, meta.num_experts}, rsrc.memory_pool);
        // TODO: 实现路由器线性层
        // RUN_INFINI(infiniopGemm(..., mlp_input, w_router, router_logits, ...));

        // 3.8.2 TopK专家选择
        // 对每个token选择top_k个专家
        // TODO: 实现TopK选择和Softmax归一化
        // top_k_experts, top_k_weights = topk_softmax(router_logits, k=2);

        // 3.8.3 专家计算
        // 初始化MoE输出为0
        std::shared_ptr<Tensor> moe_output = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
        // RUN_INFINI(infinirtMemsetAsync(moe_output->data(), 0, ...));

        // 对每个专家进行计算
        for (uint32_t expert_idx = 0; expert_idx < meta.num_experts; ++expert_idx) {
            // 找到被路由到当前专家的token
            // TODO: 找到被路由到当前专家的token索引

            // 3.8.3.1 专家FFN计算
            // Gate + Up投影: [token_count, d] × [d, 2*di_expert] = [token_count, 2*di_expert]
            // std::shared_ptr<Tensor> gate_up_output = Tensor::buffer(dt_logits, {1, 2 * di_expert}, rsrc.memory_pool);
            // TODO: 实现Gate+Up投影
            // RUN_INFINI(infiniopGemm(..., expert_tokens, w_gate_up[layer_idx][expert_idx], gate_up_output, ...));

            // 激活函数: 通常使用SiLU (Swish)
            // TODO: 实现SiLU激活函数
            // RUN_INFINI(infiniopSilu(..., gate_up_output, gate_up_output, ...));

            // Down投影: [token_count, 2*di_expert] × [2*di_expert, d] = [token_count, d]
            // std::shared_ptr<Tensor> expert_output = Tensor::buffer(dt_logits, {expert_token_count, d}, rsrc.memory_pool);
            // TODO: 实现Down投影
            // RUN_INFINI(infiniopGemm(..., gate_up_output, w_down[layer_idx][expert_idx], expert_output, ...));

            // 3.8.3.2 加权求和
            // 将专家输出按路由权重加权加到总输出中
            // TODO: 实现加权求和
            // moe_output += expert_output * expert_weights
        }

        // ====================================================================
        // 3.9 第二个残差连接
        // ====================================================================
        // residual1 + moe_output -> next_layer_input
        // std::shared_ptr<Tensor> next_layer_input = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
        // TODO: 实现残差加法
        // RUN_INFINI(infiniopAdd(..., residual1, moe_output, next_layer_input, ...));

        // 更新下一层的输入
        // layer_input = next_layer_input;

        // 同步当前层的计算
        // RUN_INFINI(infinirtStreamSynchronize(stream));
    }

    // ====================================================================
    // 4. 最终输出层
    // ====================================================================

    // 4.1 最终归一化
    // std::shared_ptr<Tensor> final_norm = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    // TODO: 实现最终RMSNorm
    // RUN_INFINI(infiniopRMSNorm(..., layer_input, rsrc.w_out_norm, final_norm, ...));

    // 4.2 输出投影到词汇表
    // 输入: [ntok, d] × 输出权重: [d, dvoc] = [ntok, dvoc]
    // std::shared_ptr<Tensor> logits = Tensor::buffer(dt_logits, {ntok, dvoc}, rsrc.memory_pool);
    // TODO: 实现输出投影
    // RUN_INFINI(infiniopGemm(..., final_norm, rsrc.w_out_embd, logits, ...));

    // ====================================================================
    // 5. 采样/解码
    // ====================================================================

    // // 5.1 如果是推理模式，对每个序列的最后一个token进行采样
    // for (uint32_t req_idx = 0; req_idx < nreq; ++req_idx) {
    //     uint32_t last_token_pos = req_pos[req_idx] + req_lens[req_idx] - 1;

    //     // 获取最后一个token的logits
    //     std::shared_ptr<Tensor> last_token_logits = logits->slice(0, last_token_pos, last_token_pos + 1); // [1, dvoc]

    //     // 5.2 应用温度
    //     if (temperature[req_idx] != 0.0) {
    //         // TODO: 实现温度缩放: logits = logits / temperature
    //     }

    //     // 5.3 TopK/TopP采样
    //     if (topk[req_idx] > 1) {
    //         // TODO: 实现TopK采样
    //     } else if (topp[req_idx] < 1.0) {
    //         // TODO: 实现TopP (Nucleus) 采样
    //     } else {
    //         // 直接取argmax
    //         // TODO: 实现Argmax
    //         // RUN_INFINI(infiniopArgmax(..., last_token_logits, &output[req_idx], ...));
    //     }
    // }

    // 同步所有计算
    RUN_INFINI(infinirtStreamSynchronize(stream));

    // 清理推理上下文
    setInferenceContext(nullptr);

    std::cout << "InferDeviceBatch completed" << std::endl;
}

__C void
forwardBatchLLaDA(struct LLaDAModel *model,
                  const uint32_t *tokens, uint32_t ntok,
                  const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                  struct KVCache **kv_caches,
                  void *logits){
    std::cout << "[DEBUG] forwardBatchLLaDA called with single-threaded mode" << std::endl;

    // Set request data
    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.kv_caches = kv_caches;
    model->req.output = nullptr;
    model->req.logits = logits;
    model->req.temperature = nullptr;
    model->req.topk = nullptr;
    model->req.topp = nullptr;

    // Single-threaded implementation - directly call inference
    if (model->dev_ids.empty()) {
        std::cerr << "[ERROR] No devices available!" << std::endl;
        return;
    }

    // Use the first device (single-threaded)
    int idev = 0;
    int ndev = 1;
    int dev_id = model->dev_ids[idev];

    std::cout << "[DEBUG] Using device " << dev_id << " for forward pass" << std::endl;

    // Create device resource (temporary for single-threaded call)
    LLaDADeviceResource rsrc;

    // Create communication handle (single device, no comm)
    infinicclComm_t comm = nullptr;

    try {
        // Direct call to inference function using model's device
        launchDevice(model->meta, model->weights, &rsrc, model->states[idev], model->req,
                    model->device, idev, ndev, dev_id, comm);

        std::cout << "[DEBUG] Forward pass completed successfully" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Forward pass failed: " << e.what() << std::endl;
    }
}

// 目前实现了资源的分配
void launchDevice(const LLaDAMeta & meta, const LLaDAWeights *weights, LLaDADeviceResource *rsrc, InferState &state, InferRequest &req,
                  infiniDevice_t device, int idev, int ndev, int dev_id, infinicclComm_t comm){
    std::cout << "launch device" << std::endl;
    // Create Device Resource
    createDeviceResource(rsrc, &meta, weights, device, idev, ndev, dev_id, comm);

    std::cout << "Cache Manager initing ..." << std::endl;
    CacheManager cache_manager(100); 

    std::cout << "Context Initing" << std::endl;
    InferenceContext ctx(rsrc->handle, rsrc->memory_pool, &cache_manager, rsrc->stream);

    // Set the inference context for this thread
    setInferenceContext(&ctx);

    // while (true) {
    //     std::unique_lock<std::mutex> lock(state.mtx);
    //     state.cv_start.wait(lock, [&] { return state.proceed || state.exit_flag; });
    //     // quit if exit_flag is set
    //     if (state.exit_flag) {
    //         break;
    //     }
    //     std::cout << "Infering Device Batch" << std::endl;
    //     inferDeviceBatch(meta, *rsrc, idev, ndev, req.tokens, req.ntok,
    //                      req.req_lens, req.nreq, req.req_pos, req.kv_caches,
    //                      req.temperature, req.topk, req.topp, req.output, req.logits);

    //     state.proceed = false;
    //     lock.unlock();
    //     state.cv_done.notify_one();
    // }

    // Debug: Output token information
    std::cout << "[DEBUG] Token Information:" << std::endl;
    std::cout << "[DEBUG] Number of requests (nreq): " << req.nreq << std::endl;
    std::cout << "[DEBUG] Total number of tokens (ntok): " << req.ntok << std::endl;
    std::cout << "[DEBUG] Tokens pointer: " << static_cast<const void*>(req.tokens) << std::endl;
    std::cout << "[DEBUG] Request lens pointer: " << static_cast<const void*>(req.req_lens) << std::endl;
    std::cout << "[DEBUG] Request pos pointer: " << static_cast<const void*>(req.req_pos) << std::endl;

    // Check for null pointers
    if (req.req_lens == nullptr) {
        std::cout << "[ERROR] req.req_lens is null!" << std::endl;
        return; // Early exit to prevent segfault
    }
    if (req.req_pos == nullptr) {
        std::cout << "[ERROR] req.req_pos is null!" << std::endl;
        return; // Early exit to prevent segfault
    }
    if (req.tokens == nullptr) {
        std::cout << "[ERROR] req.tokens is null!" << std::endl;
        return; // Early exit to prevent segfault
    }

    std::cout << "[DEBUG] Request lengths (req_lens): ";
    for (uint32_t i = 0; i < req.nreq; ++i) {
        std::cout << req.req_lens[i];
        if (i < req.nreq - 1) std::cout << ", ";
    }
    std::cout << std::endl;

    std::cout << "[DEBUG] Request positions (req_pos): ";
    for (uint32_t i = 0; i < req.nreq; ++i) {
        std::cout << req.req_pos[i];
        if (i < req.nreq - 1) std::cout << ", ";
    }
    std::cout << std::endl;

    // Check for potential out-of-bounds issues before accessing
    std::cout << "[DEBUG] Checking for potential issues:" << std::endl;
    for (uint32_t i = 0; i < req.nreq; ++i) {
        std::cout << "[DEBUG] Request " << i << ": pos=" << req.req_pos[i]
                  << ", len=" << req.req_lens[i];
        if (req.req_pos[i] < 0 || req.req_pos[i] >= req.ntok) {
            std::cout << " [ERROR: Invalid position! " << req.req_pos[i] << " >= " << req.ntok << "]";
        }
        if (req.req_lens[i] < 0 || req.req_pos[i] + req.req_lens[i] > req.ntok) {
            std::cout << " [ERROR: Length exceeds total tokens! " << req.req_pos[i] << " + " << req.req_lens[i] << " > " << req.ntok << "]";
        }
        std::cout << std::endl;
    }

    // Only output tokens if it's safe
    if (req.ntok > 0) {
        // Output first few tokens for each request
        for (uint32_t i = 0; i < req.nreq; ++i) {
            if (req.req_pos[i] >= 0 && req.req_pos[i] < req.ntok) {
                std::cout << "[DEBUG] Request " << i << " tokens (first 10): ";
                uint32_t start_idx = req.req_pos[i];
                uint32_t available_tokens = req.ntok - start_idx;
                uint32_t tokens_to_show = std::min(static_cast<uint32_t>(10), std::min(available_tokens, req.req_lens[i]));
                for (uint32_t j = start_idx; j < start_idx + tokens_to_show; ++j) {
                    std::cout << req.tokens[j];
                    if (j < start_idx + tokens_to_show - 1) std::cout << ", ";
                }
                if (req.req_lens[i] > tokens_to_show) std::cout << "...";
                std::cout << std::endl;
            } else {
                std::cout << "[DEBUG] Request " << i << ": Invalid position " << req.req_pos[i] << std::endl;
            }
        }
    } else {
        std::cout << "[DEBUG] No tokens to display (ntok = 0)" << std::endl;
    }

    inferDeviceBatch(meta, *rsrc, idev, ndev, req.tokens, req.ntok,
                         req.req_lens, req.nreq, req.req_pos, req.kv_caches,
                         req.temperature, req.topk, req.topp, req.output, req.logits);

    // Clean-Up
    std::cout << "Clearing Context" << std::endl;
    releaseDeviceResource(*rsrc);
    setInferenceContext(nullptr); // Clear the context when done

}


// TODO: not void just for tmp
LLaDAModel::LLaDAModel(const LLaDAMeta *_meta, const LLaDAWeights *weights, infiniDevice_t device_, std::vector<int> device_ids)  {
    std::cout << "Starting Distri Deploy Model" << std::endl;
    //debugPrint(_meta); // Alright Until this place
    int ndev = int(device_ids.size());
    std::cout << "The nums of dev is " << ndev << std::endl;
    meta = *_meta;  // Copy meta data
    this->weights = weights;  // Store weights pointer
    device = device_;
    dev_ids = device_ids;
    dev_resources = std::vector<LLaDADeviceResource>(ndev);
    states = std::vector<InferState>(ndev);
    threads.resize(ndev);
    RUN_INFINI(infinirtInit());
    auto comms = std::vector<infinicclComm_t>(ndev, nullptr);
    if (ndev > 1) {
        RUN_INFINI(infinicclCommInitAll(device, comms.data(), ndev, dev_ids.data()));
    }
    std::cout << "OK LET's US ROCK! " << std::endl;
    // for (int i = 0; i < ndev; i++) {
    //     std::cout << "Launch Device " << i << " Thread" << std::endl; 
    //     threads[i] = std::thread(launchDevice, std::cref(*_meta), weights, &dev_resources[i], std::ref(states[i]), std::ref(req), device, i, ndev, dev_ids[i], comms[i]);
    // }
    //launchDevice(std::cref(*_meta), weights, &dev_resources[0], std::ref(states[0]), std::ref(req), device, 0, ndev, dev_ids[0], comms[0]);

}

// 和 Pythoh 交互的C++接口
__C struct LLaDAModel * createLLaDAModel(const LLaDAMeta * meta,
                                         const LLaDAWeights * weights,
                                         infiniDevice_t device,
                                         int ndev,
                                         const int *dev_ids) {
    std::vector<int> device_ids(ndev);
    std::copy(dev_ids, dev_ids + ndev, device_ids.begin());
    std::cout << "Take a See!" << std::endl;
    //debugPrint(meta);
    LLaDAModel *model = new LLaDAModel(meta, weights, device, device_ids); // 测试代码编写在该函数体内部

    //1. 测试launchDevice
    return model;
}


__C void destroyLlaDAMoEModel(){

}