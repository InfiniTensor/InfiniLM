#include "llava_impl.hpp"
#include "llava_weight.hpp"

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../inference_context.hpp"
#include "infinicore_infer/models/llava.h"

#include <random>
#include <thread>
#include <vector>

// LLaVA设备资源创建函数，模仿jiuge.cpp的createDeviceResource
void createLlavaDeviceResource(LlavaDeviceResource *rsrc, const LlavaMeta *meta,
                             const LlavaWeights *weights,
                             infiniDevice_t device, int idev, int ndev, int dev_id,
                             infinicclComm_t comm) {
    // 🏗️ 初始化设备资源 - 统一线程架构只需要一套resource
    rsrc->device = device;
    rsrc->device_id = dev_id;

    infiniopCreateHandle(&rsrc->handle);
    infinirtStreamCreate(&rsrc->stream);
    rsrc->comm = comm;

    // TODO: 初始化memory_pool和weights（参考jiuge.cpp）
}

void releaseDeviceResource(LlavaDeviceResource &res) {
    infinirtDeviceSynchronize();
    // Release individual Tensors
    res.w_in_embd.reset();
    res.w_out_norm.reset();
    res.w_out_embd.reset();
    res.sin_table.reset();
    res.cos_table.reset();
    for (auto &t : res.w_attn_norm) {
        t.reset();
    }
    res.w_attn_norm.clear();
    for (auto &t : res.w_attn_qkv) {
        t.reset();
    }
    res.w_attn_qkv.clear();
    for (auto &t : res.b_attn_qkv) {
        t.reset();
    }
    res.b_attn_qkv.clear();
    for (auto &t : res.w_attn_out) {
        t.reset();
    }
    res.w_attn_out.clear();
    for (auto &t : res.w_ffn_norm) {
        t.reset();
    }
    res.w_ffn_norm.clear();
    for (auto &t : res.w_ffn_gate_up) {
        t.reset();
    }
    res.w_ffn_gate_up.clear();
    for (auto &t : res.w_ffn_down) {
        t.reset();
    }
    res.w_ffn_down.clear();
    infiniopDestroyHandle(res.handle);
    res.handle = nullptr;
    infinirtStreamDestroy(res.stream);
    res.stream = nullptr;
    infinicclCommDestroy(res.comm);
    res.comm = nullptr;
}





// LLaVA设备工作线程函数，严格按照jiuge.cpp的launchDevice结构
void launchLlavaDevice(const LlavaMeta &meta, const LlavaWeights *weights,
                     LlavaDeviceResource *rsrc, LlavaInferState &state,
                     LlavaRequest &req,
                     infiniDevice_t device, int idev, int ndev, int dev_id,
                     infinicclComm_t comm) {
    // Create Device Resource
    // 初始化设备资源
    createLlavaDeviceResource(rsrc, &meta, weights, device, idev, ndev, dev_id, comm);

    CacheManager cache_manager(100);
    InferenceContext ctx(rsrc->handle, rsrc->memory_pool, &cache_manager, rsrc->stream);
    setInferenceContext(&ctx);

    // 通知主线程：这个设备已经加载完成
    // TODO: 没有检查现在标志位是否靠谱
    {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.loaded = true;
        lock.unlock();
        state.cv_stage.notify_one();
    }

    // Infer Loop
    // 进入推理循环（这个线程会一直运行）
    while (true) {
        std::unique_lock<std::mutex> lock(state.mtx);
        // 关键点：线程在这里停下来等待！
        state.cv_stage.wait(lock, [&] { return state.proceed || state.exit_flag; });
        // quit if exit_flag is set
        if (state.exit_flag) {
            break;  // 退出线程
        }

        // TODO: 执行推理
        // // 占位符：简单返回一个token
        // if (req.output && req.batch_size > 0) {
        //     req.output[0] = 1;
        // }

        // inferDeviceBatch(meta, *rsrc, idev, ndev, req.tokens, req.ntok,
        //                  req.req_lens, req.nreq, req.req_pos, req.kv_caches,
        //                  req.temperature, req.topk, req.topp, req.output, req.logits);



        state.proceed = false;  // 重置信号
        lock.unlock();
        // 通知主线程：这个设备完成了推理
        state.cv_stage.notify_one();
    }
    // Clean-Up
    releaseDeviceResource(*rsrc);
    setInferenceContext(nullptr); // Clear the context when done
}



// 模仿jiuge.cpp的LlavaModel constructor
LlavaModel::LlavaModel(const LlavaMeta *_meta, const LlavaWeights *weights,
                      infiniDevice_t device_, std::vector<int> device_ids) : meta(*_meta) {
    int ndev = int(device_ids.size());
    device = device_;
    dev_ids = device_ids;
    dev_resources = std::vector<LlavaDeviceResource>(ndev);  // 每个设备的资源
    states = std::vector<LlavaInferState>(ndev);              // 每个设备的状态
    threads.resize(ndev);                                   // 每个设备的线程

    RUN_INFINI(infinirtInit());

    auto comms = std::vector<infinicclComm_t>(ndev, nullptr);
    if (ndev > 1) {
        RUN_INFINI(infinicclCommInitAll(device, comms.data(), ndev, dev_ids.data()));
    }

    // 🧵🧵🧵 这里创建线程！
    for (int i = 0; i < ndev; i++) {
        threads[i] = std::thread(
            launchLlavaDevice, 
            std::cref(meta), 
            weights, 
            &dev_resources[i], 
            std::ref(states[i]), 
            std::ref(req), 
            device, 
            i, 
            ndev, 
            dev_ids[i], 
            comms[i]);

        // ⏳ 线程立即启动，进入launchLlavaDevice函数
        // 😴 在cv_stage.wait()处开始休眠等待
    }

    // 等待所有设备线程加载完成 - 使用cv_load与jiuge.cpp保持一致
    for (int i = 0; i < ndev; i++) {
        std::unique_lock<std::mutex> lock(states[i].mtx);
        states[i].cv_stage.wait(lock, [&] { return states[i].loaded; });
        lock.unlock();
    }
}


// // 最简单的统一推理接口
// void LlavaModel::inferBatchLlava(const uint32_t* input_tokens, const void* image_data,
//                                void** kv_caches, const char* mode, uint32_t batch_size,
//                                uint32_t* output) {
//     // 暂时只是占位符实现
//     if (output && batch_size > 0) {
//         output[0] = 1;  // 返回一个简单的token
//     }
// }

// // 各阶段执行函数的占位符实现
// void LlavaModel::executeVisionStage() {
//     // 占位符
// }

// void LlavaModel::executePrefillStage() {
//     // 占位符
// }

// void LlavaModel::executeCompressStage() {
//     // 占位符
// }

// void LlavaModel::executeDecodeStage() {
//     // 占位符
// }

// void LlavaModel::workerLoop() {
//     // 占位符
// }




// API implementations - 模仿jiuge.cpp的createJiugeModel
__C struct LlavaModel *createLlavaModel(const LlavaMeta *meta,
                                        const LlavaWeights *weights,
                                        infiniDevice_t device,
                                        int ndev,
                                        const int *dev_ids) {
    std::vector<int> device_ids_vec(ndev);
    std::copy(dev_ids, dev_ids + ndev, device_ids_vec.begin());
    LlavaModel *model = new LlavaModel(meta, weights, device, device_ids_vec);
    return model;
}

__C void destroyLlavaModel(struct LlavaModel *model) {
    if (!model) {
        return;
    }

    auto ndev = model->dev_resources.size();

    // 通知所有设备线程退出
    for (size_t idev = 0; idev < ndev; idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].exit_flag = true;
        lock.unlock();
        model->states[idev].cv_stage.notify_one();
    }

    // 等待所有线程结束
    for (size_t idev = 0; idev < ndev; idev++) {
        model->threads[idev].join();
    }

    delete model;
}

// 暂时注释掉其他复杂的API函数，只保留最基本的