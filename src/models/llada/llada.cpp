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
#include <fstream>
#include <numeric>
#include <iostream>
#include <cstring>  // for memcpy


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
        w_ffn_norm, w_ffn_gate_up, w_ffn_down, w_expert_router, w_expert_gate, w_expert_up, w_expert_down;
    for (size_t layer = 0; layer < meta->nlayer; layer++) {
        w_attn_norm.push_back(
            getAttnNorm(meta, weights, layer));
        w_attn_qkv.push_back(
            getAttnQKV(meta, weights, layer, idev, ndev));

        if (weights->attn_q_norm != nullptr) {
            w_attn_q_norm.push_back(
                getAttnQNorm(meta, weights, layer));
            w_attn_k_norm.push_back(
                getAttnKNorm(meta, weights, layer));
        }
        w_attn_out.push_back(
            getAttnO(meta, weights, layer, idev, ndev)
        );
        w_ffn_norm.push_back(
            getFFNNorm(meta, weights, layer)
        );
        w_expert_router.push_back(
            getExpertRouter(meta, weights, layer, idev, ndev, meta->num_experts)
        );

        w_expert_gate.push_back(
            getExpertGate(meta, weights, layer, idev, ndev)
        );
        w_expert_down.push_back(
            getExpertDown(meta, weights, layer, idev, ndev)
        );
        w_expert_up.push_back(
            getExpertUp(meta, weights, layer, idev, ndev)
        );

        // w_ffn_down.push_back(
        //     getFFNDown(meta, weights, layer, idev, ndev));
    }
    std::cout << "Check out expert router size " << "Routers have " << w_expert_router.size() << " Shape is " << w_expert_router[0]->info() << std::endl; 




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
        .w_attn_q_norm = w_attn_q_norm,
        .w_attn_k_norm = w_attn_k_norm,
        .w_attn_out = w_attn_out,
        .w_ffn_norm = w_ffn_norm,
        .w_expert_gate   = w_expert_gate,
        .w_expert_up     = w_expert_up,
        .w_expert_down   = w_expert_down,
        .w_expert_router = w_expert_router,
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
        CacheManager cache_manager(100);
        InferenceContext ctx(rsrc.handle, rsrc.memory_pool, &cache_manager, rsrc.stream);
        setInferenceContext(&ctx);

        auto nlayer = meta.nlayer;         // 16
        auto nkvh = meta.nkvh / ndev;      // 每个设备的KV头数 (GQA分组查询) 16
        auto nh = meta.nh / ndev;          // 每个设备的注意力头数 16
        auto ngroup = nh / nkvh;           // 每个设备的组数 (多少个Q头共享一个K/V头) 1
        auto dctx = meta.dctx;             // 最大上下文长度 8192
        auto dh = meta.dh;                 // 每个头的维度.  128
        auto d = meta.d;                   // 模型隐藏层维度 2048(157184 * 2048词表)
        auto dt_logits = meta.dt_logits;   // 输出数据类型 19
        auto di_dense = meta.di_dense / ndev;  // 每个设备的密集FFN中间维度 8192
        auto di_expert = meta.di_expert / ndev; // 每个设备的专家FFN中间维度 1024
        auto dvoc = meta.dvoc;             // 词汇表大小 157184
        auto stream = rsrc.stream;
        bool has_qkv_bias = rsrc.b_attn_qkv.size() > 0;      // 是否有QKV偏置
        bool has_qk_norm = rsrc.w_attn_q_norm.size() > 0 && rsrc.w_attn_k_norm.size() > 0;  // 是否有QK归一化
        
        auto nexperts = meta.num_experts;

        // Allocate buffers

        auto logits_in = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
        auto logits_out = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
        auto qkv_buf = Tensor::buffer(dt_logits, {ntok, (nh + nkvh * 2) * dh}, rsrc.memory_pool);
        auto gate_up_buf = Tensor::buffer(dt_logits, {ntok, 2 * di_dense}, rsrc.memory_pool);
        auto o_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, rsrc.memory_pool);
        auto prob_buf = Tensor::buffer(dt_logits, {nreq, dvoc}, rsrc.memory_pool);
        auto result_buf = Tensor::buffer(INFINI_DTYPE_I64, {nreq}, rsrc.memory_pool);
        auto result_cpu = std::vector<int64_t>(nreq);
        auto attention_output = Tensor::buffer(dt_logits, {ntok, nh, dh});
        auto router_logits_buf = Tensor::buffer(dt_logits, {ntok, nexperts}, rsrc.memory_pool);


        auto qkv_rope = qkv_buf->view({ntok, nh + nkvh * 2, dh});
        auto shape_qkv = qkv_rope->shape();
        auto q_buf = qkv_rope->slice(1, 0, nh);      // 0 ---> nh q
        auto k_buf = qkv_rope->slice(1, nh, nkvh);  //nh  ---> nh+nkvh k_buf
        auto v_buf = qkv_rope->slice(1, nh + nkvh, nkvh); //nh+nkvh  ---> nh+nkvh*2 v_buf
        

        // Prepare inputs
        std::cout << "Preparing Input" << std::endl;
        auto batch_pos_ids = std::vector<uint32_t>(ntok);
        size_t req_start = 0;
        for (uint32_t req = 0; req < nreq; req++) {
            for (uint32_t i = 0; i < req_lens[req]; i++) {
                batch_pos_ids[req_start + i] = req_pos[req] + i;
            }
            req_start += req_lens[req];
        }
        std::shared_ptr<Tensor> pos_ids_buf;
        pos_ids_buf = Tensor::buffer(INFINI_DTYPE_U32, {ntok}, rsrc.memory_pool);
        RUN_INFINI(infinirtMemcpyAsync(pos_ids_buf->data(), batch_pos_ids.data(), sizeof(uint32_t) * ntok,
                                        INFINIRT_MEMCPY_H2D, stream));
        // std::shared_ptr<Tensor> hidden_states = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
        // for (uint32_t i = 0; i < ntok; i++) {   
        //     RUN_INFINI(infinirtMemcpyAsync(hidden_states->data(i * d),
        //                                 rsrc.w_in_embd->data(tokens[i] * d),
        //                                 dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
        // } // embedding weight and python slide is same
        for (uint32_t i = 0; i < ntok; i++) {   
            RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d),
                                        rsrc.w_in_embd->data(tokens[i] * d),
                                        dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
        } // embedding weight and python slide is same

        for(uint32_t layer = 0; layer < nlayer; layer ++){
            // 1. Before Attention
            // rms norm
            rmsnorm(logits_out, logits_in, rsrc.w_attn_norm[layer], meta.epsilon);
            // qkv_proj
            std::cout << "QKV BUF IS " << qkv_buf->info() << std::endl;
            std::cout << "Logits OUT BUF IS " << logits_out->info() << std::endl;
            std::cout << "Attentioin BUF IS " << rsrc.w_attn_qkv[layer]->info() << std::endl;
            linear(qkv_buf, logits_out, rsrc.w_attn_qkv[layer], 1.0, 0.0, nullptr, has_qkv_bias ? rsrc.b_attn_qkv[layer] : nullptr);

            if (has_qk_norm) {
                rmsnorm(q_buf, q_buf, rsrc.w_attn_q_norm[layer], meta.epsilon);
                rmsnorm(k_buf, k_buf, rsrc.w_attn_k_norm[layer], meta.epsilon);
            }
            // rope 
            std::cout << "Position Embedding" << std::endl;
            rope(q_buf, q_buf, pos_ids_buf, rsrc.sin_table, rsrc.cos_table);
            rope(k_buf, k_buf, pos_ids_buf, rsrc.sin_table, rsrc.cos_table); // llada modeling_lladamoe.py:390

            std::cout << "q buf info " << q_buf->info() << std::endl; // [54, 16, 128] [ntok, nh, dh]
            std::cout << "k buf info " << k_buf->info() << std::endl; // [54, 16, 128] [ntok, nh, dh]
            std::cout << "v buf info " << v_buf->info() << std::endl; // [54, 16, 128] [ntok, nh, dh]
            std::cout << "cache info " << std::endl;
            std::cout << "req pos is " << req_pos[0] << std::endl;
            std::cout << "req len is " << req_lens[0] << std::endl;
            std::cout << "KV cache info ";
            std::cout << kv_caches[0]->k[0][0]->permute({1, 0, 2})->info() << std::endl;  // [16， 54， 128]
            std::cout << "output info " << attention_output->info() << std::endl;
            BiAttention(attention_output, q_buf->permute({1, 0, 2}), k_buf->permute({1, 0, 2}), v_buf->permute({1, 0, 2}), kv_caches[0]->k[0][0]->permute({1, 0, 2}), kv_caches[0]->v[0][0]->permute({1, 0, 2}), 0);


            // 创建新张量来存储 dimMerge 的结果
            auto o_buf = Tensor::buffer(meta.dt_logits, {ntok, nh * dh}, rsrc.memory_pool);
            rearrange(o_buf, attention_output->dimMerge(1, 2));

            std::cout << logits_in->info() << std::endl;
            std::cout << o_buf->info() << std::endl;
            std::cout << rsrc.w_attn_out[layer]->info() << std::endl;
            std::cout << "logits_in contiguous: " << logits_in->isContigous() << std::endl;  
            std::cout << "o_buf contiguous: " << o_buf->isContigous() << std::endl;  
            std::cout << "weight contiguous: " << rsrc.w_attn_out[layer]->isContigous() << std::endl;
            linear(logits_in, o_buf, rsrc.w_attn_out[layer], 1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr); //self.o_proj(attn_output)

            rmsnorm(logits_out, logits_in, rsrc.w_ffn_norm[layer], meta.epsilon);

            std::cout << "Starting MoE layer " << layer << std::endl;
            std::cout << "  logits_out shape: " << logits_out->info() << std::endl;
            std::cout << "  expert_router weight shape: " << rsrc.w_expert_router[layer]->info() << std::endl;

            linear(router_logits_buf, logits_out, rsrc.w_expert_router[layer], 1.0, 0.0, nullptr, nullptr);

            std::cout << "router_logits_buf info:" << router_logits_buf->info() << std::endl;
            std::cout << "router_logits_buf contiguous: " << router_logits_buf->isContigous() << std::endl;
            router_logits_buf->debug();

            // ==================== MoE 路由和专家计算 ====================
            // 参考 modeling_lladamoe.py: LLaDAMoESparseMoeBlock::forward
            // 路由: hidden_states -> gate -> router_logits -> topkrouter -> expert_indices, expert_weights
            // 专家: hidden_states -> expert[gate][up] -> swiglu -> expert[down] -> weighted sum

            // 创建路由输出缓冲区: [ntok, 8]
            auto router_values = Tensor::buffer(meta.dt_logits, {ntok, 8}, rsrc.memory_pool);
            auto router_indices = Tensor::buffer(INFINI_DTYPE_I32, {ntok, 8}, rsrc.memory_pool);

            // 创建 correction_bias，暂时使用零偏置
            auto correction_bias = Tensor::buffer(meta.dt_logits, {nexperts}, rsrc.memory_pool);
            std::vector<float> correction_bias_cpu(nexperts, 0.0f);
            RUN_INFINI(infinirtMemcpy(correction_bias->data(), correction_bias_cpu.data(),
                                     nexperts * sizeof(float), INFINIRT_MEMCPY_H2D));

            // 调用 topkrouter 进行路由
            topkrouter(router_values, router_indices, router_logits_buf,
                      correction_bias, 1.0f, 8); // topk=8, routed_scaling_factor=2.5

            // 复制路由结果回CPU，用于按 token 顺序遍历专家
            std::vector<float> router_values_cpu(ntok * 8);
            std::vector<int> router_indices_cpu(ntok * 8);
            RUN_INFINI(infinirtMemcpy(router_values_cpu.data(), router_values->data(),
                                     router_values_cpu.size() * sizeof(float), INFINIRT_MEMCPY_D2H));
            RUN_INFINI(infinirtMemcpy(router_indices_cpu.data(), router_indices->data(),
                                     router_indices_cpu.size() * sizeof(int), INFINIRT_MEMCPY_D2H));

            std::cout << "=== MoE Routing Info ===" << std::endl;
            std::cout << "  ntok: " << ntok << ", nexperts: " << nexperts << ", topk: 8" << std::endl;
            std::cout << "  router_values shape: " << router_values->info() << std::endl;
            std::cout << "  router_indices shape: " << router_indices->info() << std::endl;

            // 创建 MoE 输出缓冲区: [ntok, d]
            auto moe_output = Tensor::buffer(meta.dt_logits, {ntok, d}, rsrc.memory_pool);

            // 创建临时缓冲区用于专家计算
            auto expert_gate_buf = Tensor::buffer(meta.dt_logits, {1, di_expert}, rsrc.memory_pool); // [1, 64]
            auto expert_up_buf = Tensor::buffer(meta.dt_logits, {1, di_expert}, rsrc.memory_pool);   // [1, 64]
            auto expert_down_buf = Tensor::buffer(meta.dt_logits, {1, d}, rsrc.memory_pool);         // [1, 2048]

            // 每个 token 通过 top-k 专家，加权求和
            // 参考 modeling_lladamoe.py:698-710 的实现
            for (size_t itok = 0; itok < ntok; ++itok) {
                // 获取当前 token 的隐藏状态: [1, d]
                auto hidden_states_i = logits_out->slice(0, itok, 1); //[1, 2048]
                // 获取当前 token 的 MoE 输出位置: [1, d]
                auto moe_output_i = moe_output->slice(0, itok, 1);    //[1, 2048]
                // 获取临时缓冲区: [1, di_expert], [1, d]
                auto expert_gate_buf_i = expert_gate_buf->slice(0, 0, 1); //
                auto expert_up_buf_i = expert_up_buf->slice(0, 0, 1);
                auto expert_down_buf_i = expert_down_buf->slice(0, 0, 1);

                // 遍历 top-k 专家，加权累加
                for (size_t k = 0; k < 8; ++k) {
                    // 获取专家索引和权重
                    int expert_idx = router_indices_cpu[itok * 8 + k];
                    float expert_weight = router_values_cpu[itok * 8 + k];

                    // 验证 expert_idx 是否在有效范围内
                    if (expert_idx < 0 || expert_idx >= (int)nexperts) {
                        std::cerr << "ERROR: Invalid expert_idx=" << expert_idx
                                  << " for token " << itok << ", expert " << k
                                  << ", nexperts=" << nexperts << std::endl;
                        continue;
                    }

                    // Debug: 打印权重 shape 信息（第一个 token, 第一个专家）
                    if (itok == 0 && k == 0) {
                        std::cout << "  expert_gate weight shape: " << rsrc.w_expert_gate[layer]->info() << std::endl;
                        std::cout << "  expert_up weight shape: " << rsrc.w_expert_up[layer]->info() << std::endl;
                        std::cout << "  expert_down weight shape: " << rsrc.w_expert_down[layer]->info() << std::endl;
                        std::cout << "  expert_gate_w after slice: " << rsrc.w_expert_gate[layer]->slice(0, expert_idx, 1)->info() << std::endl;
                        std::cout << "  expert_up_w after slice: " << rsrc.w_expert_up[layer]->slice(0, expert_idx, 1)->info() << std::endl;
                        std::cout << "  expert_down_w after slice: " << rsrc.w_expert_down[layer]->slice(0, expert_idx, 1)->info() << std::endl;
                    }

                    // 计算专家输出: hidden_states @ expert_gate -> silu * expert_up @ expert_down
                    // gate_proj: [d, di_expert] (从权重中切片并降维)
                    // 切片后是 [1, d, di_expert]，需要降维为 [d, di_expert]
                    auto expert_gate_w = rsrc.w_expert_gate[layer]->slice(0, expert_idx, 1)->view({d, di_expert});
                    // up_proj: [d, di_expert] (从权重中切片并降维)
                    auto expert_up_w = rsrc.w_expert_up[layer]->slice(0, expert_idx, 1)->view({d, di_expert});
                    // down_proj: [di_expert, d] (从权重中切片并降维)
                    auto expert_down_w = rsrc.w_expert_down[layer]->slice(0, expert_idx, 1)->view({di_expert, d});

                    // gate_output = hidden_states @ expert_gate_w: [1, di_expert]
                    // 注意：这里使用 linear 会自动处理 dtype 转换
                    std::cout << "=== Before GEMM Call ===" << std::endl;
                    std::cout << "expert_gate_buf_i (output): " << expert_gate_buf_i->info() << std::endl;
                    std::cout << "expert_gate_buf_i contiguous: " << expert_gate_buf_i->isContigous() << std::endl;
                    std::cout << "hidden_states_i (input): " << hidden_states_i->info() << std::endl;
                    std::cout << "hidden_states_i contiguous: " << hidden_states_i->isContigous() << std::endl;
                    std::cout << "expert_gate_w (weight): " << expert_gate_w->info() << std::endl;
                    std::cout << "expert_gate_w contiguous: " << expert_gate_w->isContigous() << std::endl;

                    std::cout << "1111" << std::endl;
                    linear(expert_gate_buf_i, hidden_states_i, expert_gate_w, 1.0, 0.0, nullptr, nullptr);
                    // up_output = hidden_states @ expert_up_w: [1, di_expert]

                    std::cout << "1111" << std::endl;
                    linear(expert_up_buf_i, hidden_states_i, expert_up_w, 1.0, 0.0, nullptr, nullptr);
                    // swiglu = silu(gate_output) * up_output: [1, di_expert]

                    std::cout << "1111" << std::endl;
                    swiglu(expert_gate_buf_i, expert_up_buf_i, expert_gate_buf_i);

                    std::cout << "1111" << std::endl;
                    // expert_output = swiglu @ expert_down_w: [1, d] (带权重)
                    // 当 k==0 时，直接写入 moe_output_i；当 k>0 时，累加到 moe_output_i
                    if (k == 0) {
                        std::cout << "1111" << std::endl;
                        std::cout << "Calling linear with moe_output_i, k=" << k << ", itok=" << itok << std::endl;
                        std::cout << "  moe_output_i: " << moe_output_i->info() << std::endl;
                        std::cout << "  expert_gate_buf_i: " << expert_gate_buf_i->info() << std::endl;
                        std::cout << "  expert_down_w: " << expert_down_w->info() << std::endl;
                        // 第一个专家：直接写入，不累加
                        linear(moe_output_i, expert_gate_buf_i, expert_down_w, expert_weight, 0.0, nullptr, nullptr);
                        std::cout << "1111" << std::endl;
                    } else {
                        std::cout << "1111" << std::endl;
                        // 后续专家：累加到 moe_output_i
                        // 先计算 expert_down_buf_i = expert_gate_buf_i @ expert_down_w * expert_weight
                        linear(expert_down_buf_i, expert_gate_buf_i, expert_down_w, expert_weight, 0.0, nullptr, nullptr);
                        // 再累加到 moe_output_i
                        add(moe_output_i, moe_output_i, expert_down_buf_i);
                    }
                }
            }
            std::cout << "2222" << std::endl;
            // 残差连接: logits_in = logits_in + moe_output
            // 直接使用 add 函数进行加法
            add(logits_in, logits_in, moe_output);

            std::cout << "=== MoE Computation Completed ===" << std::endl;
            std::cout << "  moe_output shape: " << moe_output->info() << std::endl;
            std::cout << "  Final logits_in shape: " << logits_in->info() << std::endl;
            // =====================================================
        }


        // 复制最终的 logits 到 host 内存（如果提供了 last_logits）
        if (last_logits != nullptr) {
            RUN_INFINI(infinirtMemcpy(last_logits, logits_in->data(),
                                     logits_in->shape()[0] * logits_in->shape()[1] * dsize(dt_logits),
                                     INFINIRT_MEMCPY_D2H));
        }

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
                uint32_t start_idx = req.req_pos[i];
                uint32_t available_tokens = req.ntok - start_idx;
                uint32_t tokens_to_show = std::min(static_cast<uint32_t>(10), std::min(available_tokens, req.req_lens[i]));
                for (uint32_t j = start_idx; j < start_idx + tokens_to_show; ++j) {
                    std::cout << req.tokens[j];
                    if (j < start_idx + tokens_to_show - 1) std::cout << ", ";
                }
                if (req.req_lens[i] > tokens_to_show) std::cout << "...";
                std::cout << std::endl;
            } 
        }
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
    std::cout << "Initing LLaDA model in Cpp side " << std::endl;
    int ndev = int(device_ids.size());
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

    LLaDAModel *model = new LLaDAModel(meta, weights, device, device_ids); // 测试代码编写在该函数体内部

    return model;
}


__C void destroyLlaDAMoEModel(){

}