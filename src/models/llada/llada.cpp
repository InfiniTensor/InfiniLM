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
    std::cout << "entering infer batch" << std::endl;

    std::cout << "Calucute Hyper Parameter" << std::endl;
   
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