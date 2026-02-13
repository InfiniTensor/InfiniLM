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

            // //logits_out->debug();
            // std::cout << "q buf" << std::endl;
            // q_buf->debug("/home/featurize/work/My_InfiniLM/q_buf_190_2048.bin");

            linear(k_buf, logits_out, k_weight->permute({1, 0}), 1.0, 0.0, nullptr,  nullptr); // k_buf
            // std::cout << "k buf" << std::endl;
            // k_buf->debug("/home/featurize/work/My_InfiniLM/k_buf_190_2048.bin");

            std::cout << "v buf" << std::endl;
            linear(v_buf, logits_out, v_weight->permute({1, 0}), 1.0, 0.0, nullptr, nullptr); // [ntok, 2048]
            // v_buf->debug("/home/featurize/work/My_InfiniLM/v_buf_190_2048.bin");

            q_buf = q_buf->view({ntok * nh, dh});

            k_buf = k_buf->view({ntok * nkvh, dh});  // 修复：使用 nkvh 而不是 nh

            rmsnorm(q_buf, q_buf, rsrc.w_attn_q_norm[layer], meta.epsilon);
            std::cout << "q norm" << std::endl;
            q_buf->debug("/home/featurize/work/My_InfiniLM/q_buf_190_2048_norm.bin");

            rmsnorm(k_buf, k_buf, rsrc.w_attn_k_norm[layer], meta.epsilon);
            std::cout << "k norm" << std::endl;
            k_buf->debug("/home/featurize/work/My_InfiniLM/k_buf_190_2048_norm.bin");

            // // 由于内存排布限制，reshape 为 [ntok, dh, nh]
            q_buf = q_buf->view({ntok, dh, nh});  // 190 128 16
            k_buf = k_buf->view({ntok, dh, nkvh});// 190 128 16
            v_buf = v_buf->view({ntok, dh, nkvh});// 190 128 16

            auto src_q = q_buf->view({ntok, nh, dh});
            auto dst_q = Tensor::buffer(src_q->dtype(), {ntok, nh, dh}, rsrc.memory_pool);
            rearrange(dst_q, src_q);
            rope_v2(dst_q, dst_q, pos_ids_buf, rsrc.sin_table, rsrc.cos_table);
            q_buf = dst_q;

            auto src_k = k_buf->view({ntok, nh, dh});
            auto dst_k = Tensor::buffer(src_k->dtype(), {ntok, nh, dh}, rsrc.memory_pool);
            rearrange(dst_k, src_k);
            rope_v2(dst_k, dst_k, pos_ids_buf, rsrc.sin_table, rsrc.cos_table);
            k_buf = dst_k;
            k_buf->debug("/home/featurize/work/My_InfiniLM/k_roped.bin");


            auto src_v = v_buf->view({ntok, nh, dh});
            auto dst_v = Tensor::buffer(src_v->dtype(), {ntok, nh, dh}, rsrc.memory_pool);
            rearrange(dst_v, src_v);
            v_buf = dst_v;
            v_buf->debug("/home/featurize/work/My_InfiniLM/v_viewd.bin");

            // ============ Attention 计算 ============
            // 转换维度为 [nh, ntok, dh]
            q_buf = q_buf->permute({1, 0, 2});  // [190, 16, 128] -> [16, 190, 128]
            k_buf = k_buf->permute({1, 0, 2});  // [190, 16, 128] -> [16, 190, 128]
            v_buf = v_buf->permute({1, 0, 2});  // [190, 16, 128] -> [16, 190, 128]
            v_buf->debug("/home/featurize/work/My_InfiniLM/v_permute.bin");
            auto v_buf_permuted = Tensor::buffer(v_buf->dtype(), {nkvh, ntok, dh}, rsrc.memory_pool);  
            // 使用 rearrange 操作进行维度重排，这会自动处理连续性  
            rearrange(v_buf_permuted, v_buf);  
            v_buf = v_buf_permuted;
            v_buf->debug("/home/featurize/work/My_InfiniLM/v_permute_rerange.bin");
            
            // 1. 计算 QK^T: [16, 190, 128] x [16, 128, 190] -> [16, 190, 190]
            auto k_transposed = k_buf->permute({0, 2, 1});  // [16, 190, 128] -> [16, 128, 190]
            auto qk_gemm = Tensor::buffer(dt_logits, {nh, ntok, ntok}, rsrc.memory_pool);
            linear(qk_gemm, q_buf, k_transposed, 1.f / float(sqrt(dh)), 0.0, nullptr, nullptr);
            qk_gemm->debug("/home/featurize/work/My_InfiniLM/qk_gemm.bin");

            // // 2. Softmax (causal)
            softmax(qk_gemm, qk_gemm, 2); // 16 190 190 
            qk_gemm->debug("/home/featurize/work/My_InfiniLM/qk_gemm_softmax.bin"); 

            // // 3. Attention
            // qk_gemm->debug("/home/featurize/work/My_InfiniLM/qk_softmax.bin");
            auto attn_buf = Tensor::buffer(dt_logits, {nh, ntok, dh}, rsrc.memory_pool);
            linear(attn_buf, qk_gemm, v_buf, 1.0, 0.0, nullptr, nullptr);
            attn_buf->debug("/home/featurize/work/My_InfiniLM/attn_buf.bin");

            // // 3. 计算 Attention x V: [16, 190, 190] x [16, 190, 128] -> [16, 190, 128]
            // auto attn_buf = Tensor::buffer(dt_logits, {nh, ntok, dh}, rsrc.memory_pool);
            // linear(attn_buf, qk_gemm, v_buf, 1.0, 0.0, nullptr, nullptr);
            // attn_buf->debug("/home/featurize/work/My_InfiniLM/attn_output.bin");

            // // 4. 转换回 [ntok, nh, dh]
            // attn_buf = attn_buf->permute({1, 0, 2});  // [16, 190, 128] -> [190, 16, 128]

            // // 保存调试信息
            // std::cout << "Attention 完成" << std::endl;
        }

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