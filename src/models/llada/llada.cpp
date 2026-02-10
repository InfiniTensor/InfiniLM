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
#include <cmath>    // for exp, log
#include <limits>   // for numeric_limits

void createDeviceResource(LLaDADeviceResource *rsrc, const LLaDAMeta * meta,
                          const LLaDAWeights *weights, infiniDevice_t device, int idev,
                          int ndev, int dev_id,
                          infinicclComm_t comm){
    //Print(meta);
    RUN_INFINI(infinirtSetDevice(device, dev_id));
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle);
    infinirtStream_t stream;
    infinirtStreamCreate(&stream);
    debugPrint(meta);
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
    std::cout << "memory init" << std::endl;
    auto memory_pool = std::make_shared<MemoryPool>(128 * 1024 * 1024);

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
    RUN_INFINI(infinirtDeviceSynchronize());
}

void releaseDeviceResource(LLaDADeviceResource &rsrc){
}

__C void
inferBatchLLaDA(struct LLaDAModel *model,
                const uint32_t *tokens, uint32_t ntok,
                const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                struct KVCache **kv_caches,
                const float *temperature, const uint32_t *topk, const float *topp,
                uint32_t *output) {

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


    // Create device resource (temporary for single-threaded call)
    LLaDADeviceResource rsrc;

    // Create communication handle (single device, no comm)
    infinicclComm_t comm = nullptr;

    try {
        // Direct call to inference function using model's device
        launchDevice(model->meta, model->weights, &rsrc, model->states[idev], model->req,
                    model->device, idev, ndev, dev_id, comm);


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
        std::cout << " INFERDEVICE LLM " << std::endl;
        std::cout << ntok << std::endl;
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
        std::cout << "ROPE META " << meta.theta << std::endl;
        std::cout << "ROPE META " << meta.dctx << std::endl;
        // std::cout << "\n1. 全局配置:" << std::endl;
        // std::cout << "   总层数: " << meta.nlayer << std::endl;
        // std::cout << "   总头数: " << meta.nh << std::endl;
        // std::cout << "   总KV头数: " << meta.nkvh << std::endl;
        // std::cout << "   隐藏层维度: " << meta.d << std::endl;
        // std::cout << "   头维度: " << meta.dh << std::endl;
        // std::cout << "   词汇表大小: " << meta.dvoc << std::endl;

        // std::cout << "\n2. 每个设备配置:" << std::endl;
        // std::cout << "   每个设备头数: " << nh << std::endl;
        // std::cout << "   每个设备KV头数: " << nkvh << std::endl;
        // std::cout << "   分组数 (GQA中的G): " << ngroup << std::endl;
        // std::cout << "NTOKEN " << ntok << std::endl;


        // Allocate buffers
        auto logits_in = Tensor::buffer(dt_logits, {ntok * d}, rsrc.memory_pool);
        auto logits_out = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
        auto qkv_buf = Tensor::buffer(dt_logits, {ntok, (nh + nkvh * 2) * dh}, rsrc.memory_pool);
        auto q_buf   = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
        auto k_buf   = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
        auto v_buf   = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
        auto gate_up_buf = Tensor::buffer(dt_logits, {ntok, 2 * di_dense}, rsrc.memory_pool);
        auto o_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, rsrc.memory_pool);
        auto prob_buf = Tensor::buffer(dt_logits, {nreq, dvoc}, rsrc.memory_pool);
        auto result_buf = Tensor::buffer(INFINI_DTYPE_I64, {nreq}, rsrc.memory_pool);
        auto result_cpu = std::vector<int64_t>(nreq);
        auto attention_output = Tensor::buffer(dt_logits, {ntok, nh, dh});
        auto router_logits_buf = Tensor::buffer(dt_logits, {ntok, nexperts}, rsrc.memory_pool); 

        auto qkv_rope = qkv_buf->view({ntok, nh + nkvh * 2, dh});
        auto shape_qkv = qkv_rope->shape();

        // Prepare inputs
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
        
        // problem                                        
        // for (uint32_t i = 0; i < ntok; i++) {   
        //     RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d),
        //                                     rsrc.w_in_embd->data(tokens[i] * d),
        //                                     dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
        // } // embedding weight and python slide is same

        for (uint32_t i = 0; i < ntok; i++) {
            RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d),
                                            rsrc.w_in_embd->data(tokens[i] * d),
                                            dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
        } // embedding weight and python slide is same

        // 同步异步拷贝操作
        RUN_INFINI(infinirtStreamSynchronize(stream));

        // std::vector<int32_t> host_data(24);
        // std::iota(host_data.begin(), host_data.end(), 1); // 填充 1,2,...,24
        // auto tensor_i32 = Tensor::weight(host_data.data(), INFINI_DTYPE_I32, {24});
        // tensor_i32->debug();
        // tensor_i32 = tensor_i32->view({4, 6});
        // tensor_i32->debug();
        // std::cout << logits_in->isContigous() << std::endl;
        // logits_in->debug();
        infinirtDeviceSynchronize();
        logits_in = logits_in->view({ntok, d}); //  bug
        //logits_in->debug();
        infinirtDeviceSynchronize();
        std::cout << "Get Info " << std::endl;
        // std::cout << logits_in->info() << std::endl;
        // logits_in->debug();
        infinirtDeviceSynchronize();
        // std::cout << logits_in->isContigous() << std::endl;

        // auto logits_test = Tensor::buffer(dt_logits, {d}, rsrc.memory_pool);
        // RUN_INFINI(infinirtMemcpyAsync(logits_test->data(),
        //                                 rsrc.w_in_embd->data(156895 * d),
        //                                 dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
        // logits_test->debug();


        std::cout << "Tokens (" << ntok << " 个): ";
        for (uint32_t i = 0; i < ntok; i++) {
            std::cout << tokens[i];
            if (i < ntok - 1) std::cout << ", ";
        }
        std::cout << std::endl;

        
        for(uint32_t layer = 0; layer < 1; layer ++){
            rmsnorm(logits_out, logits_in, rsrc.w_attn_norm[layer], meta.epsilon);
            logits_out->debug("/home/featurize/work/My_InfiniLM/hidden_states_190_2048.bin");
            // rsrc.w_attn_norm[layer]->debug();
            // logits_out->debug();
            // std::cout << rsrc.w_attn_qkv[layer]->info() << std::endl;
            auto qkv = rsrc.w_attn_qkv[layer];  // 原始权重

            auto q_weight = qkv->slice(0, 0, d); //[6144, 2048] ---> [2048 * 3, 2048]
            //q_weight->debug();
            
            // q_weight->debug("/home/featurize/work/My_InfiniLM/attention/q_weight_buf.bin");
            auto k_weight = qkv->slice(0, d, d);
            // k_weight->debug("/home/featurize/work/My_InfiniLM/attention/k_weight_buf.bin");
            auto v_weight = qkv->slice(0, 2 * d, d);
            //q_buf->debug();
            // v_weight->debug("/home/featurize/work/My_InfiniLM/attention/v_weight_buf.bin");
            linear(q_buf, logits_out, q_weight->permute({1, 0}), 1.0, 0.0, nullptr,  nullptr); // q_buf

            //logits_out->debug();
            std::cout << "q buf" << std::endl;
            q_buf->debug("/home/featurize/work/My_InfiniLM/q_buf_190_2048.bin");

            linear(k_buf, logits_out, k_weight->permute({1, 0}), 1.0, 0.0, nullptr,  nullptr); // k_buf
            std::cout << "k buf" << std::endl;
            k_buf->debug("/home/featurize/work/My_InfiniLM/k_buf_190_2048.bin");

            std::cout << "v buf" << std::endl;
            linear(v_buf, logits_out, v_weight->permute({1, 0}), 1.0, 0.0, nullptr, nullptr); // [ntok, 2048]
            v_buf->debug("/home/featurize/work/My_InfiniLM/v_buf_190_2048.bin");

            q_buf = q_buf->view({ntok * nh, dh});

            k_buf = k_buf->view({ntok * nh, dh});

            rmsnorm(q_buf, q_buf, rsrc.w_attn_q_norm[layer], meta.epsilon); // [ntok, 2048] ??
            std::cout << "q norm" << std::endl;
            //rsrc.w_attn_q_norm[layer]->debug();
            q_buf->debug("/home/featurize/work/My_InfiniLM/q_buf_190_2048_norm.bin");

            rmsnorm(k_buf, k_buf, rsrc.w_attn_k_norm[layer], meta.epsilon); // [ntok, 2048] ??
            std::cout << "k norm" << std::endl;
            k_buf->debug("/home/featurize/work/My_InfiniLM/k_buf_190_2048_norm.bin");

            q_buf = q_buf->view({ntok, nh, dh});
            k_buf = k_buf->view({ntok, nkvh, dh});
            v_buf = v_buf->view({ntok, nkvh, dh});
            
            // q_buf->debug();
            // k_buf->debug();
            rope_v2(q_buf, q_buf, pos_ids_buf, rsrc.sin_table, rsrc.cos_table);
            
            std::cout << "SIN " << std::endl;
            rsrc.sin_table->debug("/home/featurize/work/My_InfiniLM/layer_0_weights/sin.bin");
            rope_v2(k_buf, k_buf, pos_ids_buf, rsrc.sin_table, rsrc.cos_table);
            std::cout << "COUT " << std::endl;
            rsrc.cos_table->debug("/home/featurize/work/My_InfiniLM/layer_0_weights/cos.bin");
            std::cout << "Q rope " << std::endl;
            q_buf->debug("/home/featurize/work/My_InfiniLM/layer_0_weights/q_buf_190_16_128_rope.bin");
            std::cout << "K rope " << std::endl;
            k_buf->debug("/home/featurize/work/My_InfiniLM/layer_0_weights/k_buf_190_16_128_rope.bin");

            q_buf = q_buf->view({nh, ntok, dh});
            k_buf = k_buf->view({nkvh, ntok, dh});
            v_buf = v_buf->view({nkvh, ntok, dh});
            std::cout << "Q_BUF" << q_buf->info() << "K_BUF" << k_buf->info() << std::endl;
            // q k buf 16 190 128 
            auto qk_gemm = Tensor::buffer(dt_logits, {nh, ntok, ntok}, rsrc.memory_pool);
            auto k_transposed = k_buf->permute({0, 2, 1}); // 16 128 190        
            //std::cout << "Q_BUF" << q_buf->info() << "K_BUF" << k_buf->info() << std::endl;
            linear(qk_gemm, q_buf, k_transposed, 1.f / float(sqrt(dh)) , 0.0, nullptr, nullptr);
            // qk_gemm->debug();
            // causalSoftmax(qk_gemm, qk_gemm);
            // auto attn_buf = Tensor::buffer(dt_logits, {nh, ntok, dh}, rsrc.memory_pool);
            // linear(attn_buf, qk_gemm, v_buf->permute({1, 0, 2}), 1.0, 0.0, nullptr, nullptr);
        }


        // for(uint32_t layer = 0; layer < 1; layer ++){
        //     // 1. Before Attention
        //     // rms norm
        //     rmsnorm(logits_out, logits_in, rsrc.w_attn_norm[layer], meta.epsilon);
        //     // qkv_proj
        //     linear(qkv_buf, logits_out, rsrc.w_attn_qkv[layer], 1.0, 0.0, nullptr, has_qkv_bias ? rsrc.b_attn_qkv[layer] : nullptr);

        //     if (has_qk_norm) {
        //         rmsnorm(q_buf, q_buf, rsrc.w_attn_q_norm[layer], meta.epsilon);
        //         rmsnorm(k_buf, k_buf, rsrc.w_attn_k_norm[layer], meta.epsilon);
        //     }

        //     // rope
        //     rope(q_buf, q_buf, pos_ids_buf, rsrc.sin_table, rsrc.cos_table);
        //     rope(k_buf, k_buf, pos_ids_buf, rsrc.sin_table, rsrc.cos_table); // llada modeling_lladamoe.py:390

        //     BiAttention(attention_output, q_buf->permute({1, 0, 2}), k_buf->permute({1, 0, 2}), v_buf->permute({1, 0, 2}), kv_caches[0]->k[0][0]->permute({1, 0, 2}), kv_caches[0]->v[0][0]->permute({1, 0, 2}), 0);


        //     // 创建新张量来存储 dimMerge 的结果
        //     auto o_buf = Tensor::buffer(meta.dt_logits, {ntok, nh * dh}, rsrc.memory_pool);
        //     rearrange(o_buf, attention_output->dimMerge(1, 2));

        //     linear(logits_in, o_buf, rsrc.w_attn_out[layer], 1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr); //self.o_proj(attn_output)

        //     rmsnorm(logits_out, logits_in, rsrc.w_ffn_norm[layer], meta.epsilon);


        //     // ==================== MoE 路由和专家计算 ====================
        //     // 逐个 token 处理：每个 token [1, d] -> router -> [1, nexperts] -> topkrouter -> expert_indices, expert_weights
        //     // 专家: hidden_states -> expert[gate][up] -> swiglu -> expert[down] -> weighted sum

        //     // 创建 MoE 输出缓冲区: [ntok, d]
        //     auto moe_output = Tensor::buffer(meta.dt_logits, {ntok, d}, rsrc.memory_pool);

        //     // 创建临时缓冲区用于 router 计算和专家计算
        //     auto router_logits_token = Tensor::buffer(INFINI_DTYPE_F32, {1, nexperts}, rsrc.memory_pool); // [1, nexperts] 单个 token 的 router 输出 - 必须是 F32 用于 softmax + topk
        //     auto correction_bias = Tensor::buffer(INFINI_DTYPE_F32, {nexperts}, rsrc.memory_pool); // [nexperts] 偏置（全零）- 注意：必须是 1D tensor
        //     auto router_values = Tensor::buffer(INFINI_DTYPE_F32, {1, 8}, rsrc.memory_pool); // [1, 8] top-k 的值（权重）- 注意：必须是 2D tensor with shape [batch, topk]
        //     auto router_indices = Tensor::buffer(INFINI_DTYPE_I32, {1, 8}, rsrc.memory_pool); // [1, 8] top-k 的索引（专家编号）- 注意：必须是 2D tensor with shape [batch, topk]

        //     // 用于 host 端读取路由结果
        //     auto router_values_cpu = std::vector<float>(8);
        //     auto router_indices_cpu = std::vector<int32_t>(8);

        //     // 临时缓冲区用于专家计算
        //     auto expert_gate_buf = Tensor::buffer(meta.dt_logits, {1, di_expert}, rsrc.memory_pool); // [1, 64]
        //     auto expert_up_buf = Tensor::buffer(meta.dt_logits, {1, di_expert}, rsrc.memory_pool);   // [1, 64]
        //     auto expert_down_buf = Tensor::buffer(meta.dt_logits, {1, d}, rsrc.memory_pool);         // [1, 2048]



        //     // 每个 token 通过 top-k 专家，加权求和
        //     // 参考 modeling_lladamoe.py:698-710 的实现
        //     for (size_t itok = 0; itok < ntok; ++itok) {
        //         std::cout << "Tok " << itok << std::endl;
        //         // 获取当前 token 的隐藏状态: [1, d]
        //         auto hidden_states_i = logits_out->slice(0, itok, 1); //[1, 2048]
        //         // 获取当前 token 的 MoE 输出位置: [1, d]
        //         auto moe_output_i = moe_output->slice(0, itok, 1);    //[1, 2048]
        //         // 获取临时缓冲区: [1, di_expert], [1, d]
        //         auto expert_gate_buf_i = expert_gate_buf->slice(0, 0, 1); //
        //         auto expert_up_buf_i = expert_up_buf->slice(0, 0, 1);
        //         auto expert_down_buf_i = expert_down_buf->slice(0, 0, 1);

        //         // Step 1: 计算当前 token 的 router logits: hidden_states_i [1, 2048] @ w_expert_router [2048, nexperts] -> [1, nexperts]
        //         linear(router_logits_token, hidden_states_i, rsrc.w_expert_router[layer], 1.0, 0.0, nullptr, nullptr);
        //         RUN_INFINI(infinirtStreamSynchronize(stream));

        //         // Step 2: 直接在 CPU 上进行 top-k 计算，避免使用 topkrouter 算子
        //         // 将 router_logits 拷贝到 CPU
        //         std::vector<float> router_logits_cpu(nexperts);
        //         RUN_INFINI(infinirtMemcpy(router_logits_cpu.data(), router_logits_token->data(), sizeof(float) * nexperts, INFINIRT_MEMCPY_D2H));
        //         RUN_INFINI(infinirtStreamSynchronize(stream));

        //         // CPU 端计算：softmax + top-8
        //         // Step 2a: Softmax (correction_bias 全零，routed_scaling_factor = 1.0)
        //         std::vector<float> softmax_probs(nexperts);
        //         float max_logit = *std::max_element(router_logits_cpu.begin(), router_logits_cpu.end());
        //         float sum_exp = 0.0f;
        //         for (size_t i = 0; i < nexperts; ++i) {
        //             softmax_probs[i] = std::exp(router_logits_cpu[i] - max_logit);
        //             sum_exp += softmax_probs[i];
        //         }
        //         for (size_t i = 0; i < nexperts; ++i) {
        //             softmax_probs[i] /= sum_exp;
        //         }

        //         // Step 2b: Top-8 selection
        //         std::vector<std::pair<float, int32_t>> expert_scores(nexperts);
        //         for (size_t i = 0; i < nexperts; ++i) {
        //             expert_scores[i] = {softmax_probs[i], static_cast<int32_t>(i)};
        //         }
        //         std::partial_sort(expert_scores.begin(), expert_scores.begin() + 8, expert_scores.end(),
        //             [](const auto& a, const auto& b) { return a.first > b.first; });

        //         // 保存 top-8 结果
        //         for (size_t k = 0; k < 8; ++k) {
        //             router_values_cpu[k] = expert_scores[k].first;
        //             router_indices_cpu[k] = expert_scores[k].second;
        //         }

        //         // 将 CPU 计算结果拷贝回 GPU
        //         RUN_INFINI(infinirtMemcpy(router_values->data(), router_values_cpu.data(), sizeof(float) * 8, INFINIRT_MEMCPY_H2D));
        //         RUN_INFINI(infinirtMemcpy(router_indices->data(), router_indices_cpu.data(), sizeof(int32_t) * 8, INFINIRT_MEMCPY_H2D));
        //         RUN_INFINI(infinirtStreamSynchronize(stream));

        //         // 遍历 top-k 专家，加权累加
        //         for (size_t k = 0; k < 8; ++k) {
        //             int expert_idx = router_indices_cpu[k];
        //             float expert_weight = router_values_cpu[k];

        //             // 验证 expert_idx 是否在有效范围内
        //             if (expert_idx < 0 || expert_idx >= (int)nexperts) {
        //                 std::cerr << "ERROR: Invalid expert_idx=" << expert_idx
        //                           << " for token " << itok << ", expert " << k
        //                           << ", nexperts=" << nexperts << std::endl;
        //                 continue;
        //             }

        //             // 计算专家输出: hidden_states @ expert_gate -> silu * expert_up @ expert_down
        //             // gate_proj: [d, di_expert] (从权重中切片并降维)
        //             auto expert_gate_w = rsrc.w_expert_gate[layer]->slice(0, expert_idx, 1)->view({d, di_expert});
        //             auto expert_up_w = rsrc.w_expert_up[layer]->slice(0, expert_idx, 1)->view({d, di_expert});
        //             auto expert_down_w = rsrc.w_expert_down[layer]->slice(0, expert_idx, 1)->view({di_expert, d});

        //             // gate_output = hidden_states @ expert_gate_w: [1, di_expert]
        //             linear(expert_gate_buf_i, hidden_states_i, expert_gate_w, 1.0, 0.0, nullptr, nullptr);

        //             // up_output = hidden_states @ expert_up_w: [1, di_expert]
        //             linear(expert_up_buf_i, hidden_states_i, expert_up_w, 1.0, 0.0, nullptr, nullptr);

        //             // swiglu = silu(gate_output) * up_output: [1, di_expert]
        //             swiglu(expert_gate_buf_i, expert_up_buf_i, expert_gate_buf_i);

        //             // expert_output = swiglu @ expert_down_w: [1, d] (带权重)
        //             if (k == 0) {
        //                 // 第一个专家：直接写入，不累加
        //                 linear(moe_output_i, expert_gate_buf_i, expert_down_w, expert_weight, 0.0, nullptr, nullptr);
        //             } else {
        //                 // 后续专家：累加到 moe_output_i
        //                 linear(expert_down_buf_i, expert_gate_buf_i, expert_down_w, expert_weight, 0.0, nullptr, nullptr);
        //                 add(moe_output_i, moe_output_i, expert_down_buf_i);
        //             }
        //         }
        //     }
        //     // 残差连接: logits_in = logits_in + moe_output
        //     // 直接使用 add 函数进行加法
        //     add(logits_in, logits_in, moe_output);
        // }

        // std::cout << "finish" << std::endl;
        // // 复制最终的 logits 到 host 内存（如果提供了 last_logits）
        // if (last_logits != nullptr) {
        //     RUN_INFINI(infinirtMemcpy(last_logits, logits_in->data(),
        //                              logits_in->shape()[0] * logits_in->shape()[1] * dsize(dt_logits),
        //                              INFINIRT_MEMCPY_D2H));
        // }

        // RUN_INFINI(infinirtStreamSynchronize(stream));
        // // 清理推理上下文
        // setInferenceContext(nullptr);
}
__C void
forwardBatchLLaDA(struct LLaDAModel *model,
                  const uint32_t *tokens, uint32_t ntok,
                  const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                  struct KVCache **kv_caches,
                  void *logits){

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