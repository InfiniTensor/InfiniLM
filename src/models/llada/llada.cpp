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
        CacheManager cache_manager(100);
        InferenceContext ctx(rsrc.handle, rsrc.memory_pool, &cache_manager, rsrc.stream);
        setInferenceContext(&ctx);

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
        
        auto nexperts = meta.num_experts;

        // Allocate buffers
        std::cout << "Allocating  Buffer " << std::endl;
        auto logits_in = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
        auto logits_out = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
        auto qkv_buf = Tensor::buffer(dt_logits, {ntok, (nh + nkvh * 2) * dh}, rsrc.memory_pool);
        auto gate_up_buf = Tensor::buffer(dt_logits, {ntok, 2 * di_dense}, rsrc.memory_pool);
        auto o_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, rsrc.memory_pool);
        auto prob_buf = Tensor::buffer(dt_logits, {nreq, dvoc}, rsrc.memory_pool);
        auto result_buf = Tensor::buffer(INFINI_DTYPE_I64, {nreq}, rsrc.memory_pool);
        auto result_cpu = std::vector<int64_t>(nreq);

        auto router_logits_buf = Tensor::buffer(dt_logits, {di_dense, nexperts}, rsrc.memory_pool);
        
        std::cout << "Slice Qkv buffer" << std::endl;
        auto qkv_rope = qkv_buf->view({ntok, nh + nkvh * 2, dh});
        auto shape_qkv = qkv_rope->shape();
        std::cout << "[";
        for(size_t element : shape_qkv){

            std::cout << element << ", ";
        }
        std::cout << "]" << std::endl;
        auto q_buf = qkv_rope->slice(1, 0, nh);      // 0 ---> nh q
        auto k_buf = qkv_rope->slice(1, nh, nkvh);  //nh  ---> nh+nkvh k_buf
        auto v_buf = qkv_rope->slice(1, nh + nkvh, nkvh); //nh+nkvh  ---> nh+nkvh*2 v_buf

        std::cout << "Allocating  Buffer Finish" << std::endl;
        // Attention
        // attention inner
        size_t max_qk_size = 0;
        size_t max_seq_len = 0;

        std::cout << "Get Max Seq Len" << std::endl;
        for (uint32_t req = 0; req < nreq; req++) {
                auto past_len = req_pos[req];
                auto seq_len = req_lens[req];
                auto total_len = past_len + seq_len;

                max_qk_size = std::max(max_qk_size, size_t(seq_len * total_len));
                max_seq_len = std::max(max_seq_len, size_t(seq_len));
        }

        std::cout << "Get Attention Buffer "<< std::endl;
        auto qk_buf = Tensor::buffer(dt_logits, {nh * max_qk_size}, rsrc.memory_pool);
        auto rearrange_q_buf = Tensor::buffer(dt_logits, {nkvh, ngroup * max_seq_len, dh}, rsrc.memory_pool);
        auto q_rearrange = rearrange_q_buf->view({nkvh, ngroup, max_seq_len, dh});
        auto attn_val_buf = Tensor::buffer(dt_logits, {nkvh, ngroup * max_seq_len, dh}, rsrc.memory_pool);
        auto attn_val_gemm = attn_val_buf->view({nkvh, ngroup, max_seq_len, dh});

    
        auto batch_pos_ids = std::vector<uint32_t>(ntok);
        for (uint32_t req_idx = 0; req_idx < nreq; ++req_idx) {
            uint32_t start_pos = req_pos[req_idx];
            uint32_t req_len = req_lens[req_idx];
            for (uint32_t i = 0; i < req_len; ++i) {
                batch_pos_ids[start_pos + i] = start_pos + i;
            }
        }
        std::shared_ptr<Tensor> pos_ids_buf;
        if (rsrc.device == INFINI_DEVICE_CPU) {
            pos_ids_buf = Tensor::weight(batch_pos_ids.data(), INFINI_DTYPE_U32, {ntok});
        } else { // GPU
            pos_ids_buf = Tensor::buffer(INFINI_DTYPE_U32, {ntok}, rsrc.memory_pool);
            RUN_INFINI(infinirtMemcpyAsync(pos_ids_buf->data(), batch_pos_ids.data(), sizeof(uint32_t) * ntok,
                                        INFINIRT_MEMCPY_H2D, stream));
        }
        std::shared_ptr<Tensor> hidden_states = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
        for (uint32_t i = 0; i < ntok; i++) {
            // 查找词嵌入表: [vocab_size, d] -> 输出token i的嵌入向量
            RUN_INFINI(infinirtMemcpyAsync(hidden_states->data(i * d),
                                        rsrc.w_in_embd->data(tokens[i] * d),
                                        dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
        }

        for (uint32_t layer_idx = 0; layer_idx < nlayer; ++layer_idx) {
                std::cout << "Processing layer " << layer_idx << std::endl;
                std::shared_ptr<Tensor> attn_input = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);

                rmsnorm(attn_input, hidden_states, rsrc.w_attn_norm[layer_idx], meta.epsilon);

                std::shared_ptr<Tensor> qkv_buf = Tensor::buffer(dt_logits, {ntok, (nh + nkvh * 2) * dh}, rsrc.memory_pool);

                linear(qkv_buf, attn_input, rsrc.w_attn_qkv[layer_idx], 1.0, 0.0, nullptr, nullptr);
    
                
                auto qkv_rope = qkv_buf->view({ntok, nh + nkvh * 2, dh});
                auto q_buf = qkv_rope->slice(1, 0, nh);           // [ntok, nh, dh] - Query部分
                auto k_buf = qkv_rope->slice(1, nh, nkvh);       // [ntok, nkvh, dh] - Key部分
                auto v_buf = qkv_rope->slice(1, nh+nkvh, nkvh);  // [ntok, nkvh, dh] - Value部分

                
                if (has_qk_norm) {
                    std::cout << "Use Qk norm" << std::endl;
                    rmsnorm(q_buf, q_buf, rsrc.w_attn_q_norm[layer_idx], meta.epsilon);
                    rmsnorm(k_buf, k_buf, rsrc.w_attn_k_norm[layer_idx], meta.epsilon);
                }
                
                rope(q_buf, q_buf, pos_ids_buf, rsrc.sin_table, rsrc.cos_table);
                rope(k_buf, k_buf, pos_ids_buf, rsrc.sin_table, rsrc.cos_table);
                
                uint32_t token_offset = 0;
                for(uint32_t req = 0; req < nreq; req ++){
                    auto past_len = req_pos[req];
                    auto seq_len  = req_lens[req];
                    auto total_len = past_len + seq_len;
                    // 提取当前请求的数据
                    auto o = o_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});
                    auto q = qkv_rope->slice({{0, token_offset, seq_len}, {1, 0, nh}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});;
                    auto k = qkv_rope->slice({{0, token_offset, seq_len}, {1, nh, nkvh}});
                    auto v = qkv_rope->slice({{0, token_offset, seq_len}, {1, nh + nkvh, nkvh}});
                    // self attention
                    // concat
                    rearrange(kv_caches[req]->k[idev][layer_idx]->slice(0, past_len, seq_len), k);
                    rearrange(kv_caches[req]->v[idev][layer_idx]->slice(0, past_len, seq_len), v);
                    // qk
                    rearrange(q_rearrange->slice(2, 0, seq_len), q);
                    auto qk_gemm = qk_buf->slice(0, 0, nh * seq_len * total_len)->view({nkvh, ngroup * seq_len, total_len});
                    auto k_gemm = kv_caches[req]->k[idev][layer_idx]->slice(0, 0, total_len)->permute({1, 2, 0});
                    linear(qk_gemm, rearrange_q_buf->slice(1, 0, ngroup * seq_len), k_gemm, 1.f / float(sqrt(dh)), 0.f, nullptr, nullptr);
                    // softmax
                    auto qk_softmax = qk_gemm->view({nh, seq_len, total_len});
                    causalSoftmax(qk_softmax, qk_softmax);
                    auto v_gemm = kv_caches[req]->v[idev][layer_idx]->slice(0, 0, total_len)->permute({1, 0, 2});
                    linear(attn_val_buf->slice(1, 0, ngroup * seq_len), qk_gemm, v_gemm, 1.f, 0.f, nullptr, nullptr);
                    // rearrange attn val
                    rearrange(o, attn_val_gemm->slice(2, 0, seq_len));
                    token_offset += seq_len;
                }
                 // o_proj
                linear(logits_in, o_buf, rsrc.w_attn_out[layer_idx], 1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr); // only rank 0 adds residual
                        // All_reduce if distributed
                if (rsrc.comm != nullptr) {
                    RUN_INFINI(infinicclAllReduce(
                        logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                        INFINICCL_SUM, rsrc.comm, stream));
                    RUN_INFINI(infinirtStreamSynchronize(stream));
                }
        
                // 2. FFN Expert LLaDAMoESparseMoeBlock
                // linear()                   //router_logits = self.gate(hidden_states)
        
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