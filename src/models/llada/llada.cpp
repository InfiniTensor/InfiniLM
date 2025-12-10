#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../inference_context.hpp"
#include "infinicore_infer.h"

#include "llada_impl.hpp"
#include "llada_weight.hpp"

#include <random>
#include <thread>
#include <vector>

// #TODO:这个是草稿版
void createDeviceResource(LLaDADeviceResource *rsrc, const LLaDAMeta * meta,
                          const LLaDAWeights *weights, infiniDevice_t device, int idev,
                          int ndev, int dev_id,
                          infinicclComm_t comm){
    std::cout << "Set Device" << std::endl;
    RUN_INFINI(infinirtSetDevice(device, dev_id));
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle);
    infinirtStream_t stream;
    infinirtStreamCreate(&stream);
    

    std::cout << "Set Weight" << std::endl;
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

void inferDeviceBatch(const LLaDAMeta &meta, LLaDADeviceResource &rsrc,
                      uint32_t idev, uint32_t ndev,
                      const uint32_t *tokens, uint32_t ntok,
                      const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                      struct KVCache **kv_caches,
                      const float *temperature, const uint32_t *topk, const float *topp,
                      uint32_t *output, void *last_logits){
    std::cout << "Infer Device Batch" << std::endl;
}

__C void
inferBatchLLaDA(struct LLaDAModel *model,
                const uint32_t *tokens, uint32_t ntok,
                const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                struct KVCache **kv_caches,
                const float *temperature, const uint32_t *topk, const float *topp,
                uint32_t *output){
    std::cout << "Infer Batch LLaDA is be called";
}

__C void
forwardBatchLLaDA(struct LLaDAModel *model,
                  const uint32_t *tokens, uint32_t ntok,
                  const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                  struct KVCache **kv_caches,
                  void *logits){
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

    for (size_t idev = 0; idev < model->dev_ids.size(); idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].proceed = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }
    for (size_t i = model->dev_ids.size(); i > 0; i--) {
        auto idev = i - 1;
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].cv_done.wait(lock, [&] { return !(model->states[idev].proceed); });
        lock.unlock();
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

    while (true) {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.cv_start.wait(lock, [&] { return state.proceed || state.exit_flag; });
        // quit if exit_flag is set
        if (state.exit_flag) {
            break;
        }
        std::cout << "Infering Device Batch" << std::endl;
        inferDeviceBatch(meta, *rsrc, idev, ndev, req.tokens, req.ntok,
                         req.req_lens, req.nreq, req.req_pos, req.kv_caches,
                         req.temperature, req.topk, req.topp, req.output, req.logits);

        state.proceed = false;
        lock.unlock();
        state.cv_done.notify_one();
    }

    // Clean-Up
    std::cout << "Clearing Context" << std::endl;
    releaseDeviceResource(*rsrc);
    setInferenceContext(nullptr); // Clear the context when done

}


// TODO: not void just for tmp
LLaDAModel::LLaDAModel(const LLaDAMeta *_meta, const LLaDAWeights *weights, infiniDevice_t device_, std::vector<int> device_ids)  {
    std::cout << "Starting Distri Deploy Model" << std::endl;
    int ndev = int(device_ids.size());
    std::cout << "The nums of dev is " << ndev << std::endl;
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

    for (int i = 0; i < ndev; i++) {
        std::cout << "Launch Device " << i << " Thread" << std::endl; 
        threads[i] = std::thread(launchDevice, std::cref(meta), weights, &dev_resources[i], std::ref(states[i]), std::ref(req), device, i, ndev, dev_ids[i], comms[i]);
        //launchDevice(_meta, weights, &dev_resources[i], std::ref(states[i]), std::ref(req), device, i, ndev, dev_ids[i], comms[i]);
    }

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

    //1. 测试launchDevice
    return model;
}


__C void destroyLlaDAMoEModel(){

}