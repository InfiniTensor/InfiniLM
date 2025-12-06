#include "llava_impl.hpp"
#include "llava_weight.hpp"

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../inference_context.hpp"
#include "infinicore_infer/models/llava.h"

#include <random>
#include <thread>
#include <vector>
#include <fstream>
#include <iomanip>

// LLaVA设备资源创建函数，模仿jiuge.cpp的createDeviceResource
void createLlavaDeviceResource(LlavaDeviceResource *rsrc, const LlavaMeta *meta,
                             const LlavaWeights *weights,
                             infiniDevice_t device, int idev, int ndev, int dev_id,
                             infinicclComm_t comm) {
    RUN_INFINI(infinirtSetDevice(device, dev_id));
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle);
    infinirtStream_t stream;
    infinirtStreamCreate(&stream);

    // 初始化memory_pool
    auto memory_pool = std::make_shared<MemoryPool>(128 * 1024 * 1024);

    // 初始化Language Model权重（暂时为空，复用jiuge结构）
    std::vector<std::shared_ptr<Tensor>> w_attn_norm, w_attn_qkv, b_attn_qkv, w_attn_q_norm, w_attn_k_norm, w_attn_out,
        w_ffn_norm, w_ffn_gate_up, w_ffn_down;

    // 初始化Vision Encoder权重
    auto vision_patch_embed_weight = getPatchEmbedWeight(meta, weights);
    auto vision_position_embedding = createPositionEmbedding(meta, weights); // 从meta中获取形状
    auto vision_class_token = getClassToken(meta, weights); // 从meta中获取形状
    // auto vision_class_embedding = getClassToken(meta);

    // 临时创建language model权重（将来应该从weights中加载）
    std::shared_ptr<Tensor> w_in_embd = nullptr;
    std::shared_ptr<Tensor> w_out_norm = nullptr;
    std::shared_ptr<Tensor> w_out_embd = nullptr;
    std::shared_ptr<Tensor> sin_table = nullptr;
    std::shared_ptr<Tensor> cos_table = nullptr;

    *rsrc = LlavaDeviceResource{
        device,
        dev_id,
        handle,
        w_in_embd, w_out_norm, w_out_embd, sin_table, cos_table,
        w_attn_norm, w_attn_qkv, b_attn_qkv, w_attn_q_norm, w_attn_k_norm, w_attn_out,
        w_ffn_norm, w_ffn_gate_up, w_ffn_down,
        vision_patch_embed_weight,
        vision_position_embedding, // TODO: 不知道是什么但先放这儿
        vision_class_token, // TODO: 不知道是什么但先放这儿
        stream,
        comm,
        memory_pool,
    };
    RUN_INFINI(infinirtDeviceSynchronize());
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


// LLaVA视觉编码设备层推理函数（模仿inferDeviceBatch）
void inferDeviceBatchVision(const LlavaMeta &meta, LlavaDeviceResource &rsrc,
                           uint32_t idev, uint32_t ndev,
                           const void *image_data, uint32_t *output) {

// inputs["input_ids"].shape: torch.Size([1, 593])
// shape of weight: torch.Size([1024, 3, 14, 14])
// shape of input: torch.Size([1, 3, 336, 336])
// shape of output: torch.Size([1, 1024, 24, 24])
// Debug: print image_data pointer

    // === 1. 准备参数 ===
    auto vision_embed_dim = meta.vision_meta.vision_embed_dim; // 1024
    auto image_size = meta.vision_meta.image_size; // 336
    auto patch_size = meta.vision_meta.patch_size; // 14
    auto dt_logits = meta.language_meta.dt_logits; // F16
    auto stream = rsrc.stream;

    // 计算patches数量
    auto patches_per_dim = image_size / patch_size; // 24
    // auto total_patches = patches_per_dim * patches_per_dim;

    // === 2. 准备buffer ===
    // auto input_image_tensor_f32 = Tensor::buffer(INFINI_DTYPE_F32, {1, 3, image_size, image_size}, rsrc.memory_pool);
    auto input_image_tensor = Tensor::buffer(dt_logits, {1, 3, image_size, image_size}, rsrc.memory_pool);
    auto patch_embed_output = Tensor::buffer(dt_logits, {1, vision_embed_dim, patches_per_dim, patches_per_dim}, rsrc.memory_pool);

    // 复制输入图像数据
    RUN_INFINI(infinirtMemcpyAsync(input_image_tensor->data(), image_data,
                                  image_size * image_size * 3 * sizeof(uint16_t),
                                  INFINIRT_MEMCPY_H2D, stream));

    // printf("DEBUG: input_image_tensor after memcpy:\n");
    // input_image_tensor->debug_first_n(10);

    // === 3. CLIPVisionEmbeddings Forward ===
    // Step 1: Patch Embedding (Conv2d)

    printf("DEBUG: Running Conv2d: input [1,3,%ld,%ld] -> output [1,%ld,%ld,%ld]\n",
           image_size, image_size, vision_embed_dim, patches_per_dim, patches_per_dim);

    // 准备卷积参数
    std::vector<size_t> pads = {0, 0};  // 无padding
    std::vector<size_t> strides = {static_cast<size_t>(patch_size), static_cast<size_t>(patch_size)};
    std::vector<size_t> dilations = {1, 1};

    // patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # Conv2d
    conv2d(patch_embed_output, input_image_tensor, rsrc.vision_patch_embed_weight,
           nullptr, pads, strides, dilations); // （1，1024，24，24）

    // flatten 2D patch -> [batch, embed_dim, total_patches]
    auto total_patches = patches_per_dim * patches_per_dim;
    auto patch_embed_flat = patch_embed_output->view({1, vision_embed_dim, total_patches});

    // transpose -> [batch, total_patches, embed_dim]
    auto patch_embed_transposed = patch_embed_flat->permute({0, 2, 1});
    // 创建 class embedding buffer
    // class_embeds = self.class_embedding.expand(batch_size, 1, -1)
    // assume batch=1

    auto class_embed_tensor = Tensor::buffer(dt_logits, {1, 1, vision_embed_dim}, rsrc.memory_pool); 
    // Tensor: shape[ 1 1 1024 ]
    RUN_INFINI(infinirtMemcpyAsync(class_embed_tensor->data(),
                                rsrc.vision_class_token->data(),
                                sizeof(uint16_t) * vision_embed_dim,
                                INFINIRT_MEMCPY_D2D, stream));


    // embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
    auto embeddings = Tensor::buffer(dt_logits, {1, 1 + total_patches, vision_embed_dim}, rsrc.memory_pool);
    // [ 1 577 1024 ]

    // 1) 把 class token 放到 embeddings[:, 0:1, :]
    rearrange(embeddings->slice(1, 0, 1), class_embed_tensor); // 注意：slice(dim=1, start=0, length=1)

    // 2) 把所有 patch token 放到 embeddings[:, 1:1+T, :]
    rearrange(embeddings->slice(1, 1, total_patches), patch_embed_transposed); // 注意：slice(dim=1, start=1, length=total_patches)

    printf("DEBUG: embeddings after concat:\n");
    embeddings->debug_first_n(10);
    rsrc.vision_position_embedding->debug_first_n(10);

    // 3) 加 position embedding （pos tensor 必须是 [1, 1+T, C]）
    add(embeddings, embeddings, rsrc.vision_position_embedding);

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

        inferDeviceBatchVision(meta, *rsrc, idev, ndev, 
                                req.image_data, req.output);

        // // === LLaVA四阶段推理流程 ===
        // // 阶段1: Vision Encoder (如果有图像)
        // if (req.image_data != nullptr) {
        //     state.current_stage = 1;
        //     state.stage_completed = false;
        //     lock.unlock();
        //     state.cv_stage.notify_one(); // 通知主线程进入阶段1

        //     // TODO: 实现vision encoding
        //     // encodeVisionFeatures(meta, *rsrc, req.image_data, state.vision_features);

        //     lock.lock();
        //     state.stage_completed = true;
        //     state.current_stage = 2;
        // }

        // // 阶段2: MultiModal Projector (如果有图像特征)
        // if (state.vision_features != nullptr) {
        //     lock.unlock();
        //     state.cv_stage.notify_one(); // 通知主线程进入阶段2

        //     // TODO: 实现multimodal projection
        //     // projectMultiModalFeatures(meta, *rsrc, state.vision_features, state.projected_features);

        //     lock.lock();
        //     state.stage_completed = true;
        //     state.current_stage = 3;
        // }

        // // 阶段3: Language Model Prefill (包含KV-Cache)
        // state.current_stage = 3;
        // state.stage_completed = false;
        // lock.unlock();
        // state.cv_stage.notify_one(); // 通知主线程进入阶段3

        // // TODO: 实现language model prefill
        // // 这里调用Jiuge的推理逻辑来处理text tokens + projected vision features
        // // inferDeviceBatchLanguage(meta, *rsrc, idev, ndev, req.input_tokens, req.ntok,
        // //                          req.req_lens, req.nreq, req.req_pos, req.kv_caches,
        // //                          req.temperature, req.topk, req.topp, req.output, nullptr);

        // lock.lock();
        // state.stage_completed = true;
        // state.current_stage = 4;

        // // 阶段4: KV-Cache Compression (可选)
        // if (req.kv_caches != nullptr && state.stage_completed) {
        //     lock.unlock();
        //     state.cv_stage.notify_one(); // 通知主线程进入阶段4

        //     // TODO: 实现KV-Cache压缩 (Future: 集成Fastcache)
        //     // compressKVCaches(meta, *rsrc, req.kv_caches);

        //     lock.lock();
        //     state.stage_completed = true;
        // }

        // // 简单占位符：返回一个token (临时)
        // if (req.output && req.batch_size > 0) {
        //     req.output[0] = 1; // 暂时返回固定token
        // }



        state.proceed = false;  // 重置信号
        lock.unlock();
        // 通知主线程：这个设备完成了推理
        state.cv_stage.notify_one();
    }
    // Clean-Up
    releaseDeviceResource(*rsrc);
    setInferenceContext(nullptr); // Clear the context when done
}



// // LLaVA四阶段统一推理实现
// void LlavaModel::inferBatchLlava(const uint32_t* input_tokens, const void* image_data,
//                                 void** kv_caches, uint32_t batch_size,
//                                 uint32_t* output) {
//     // 1. 设置推理请求参数
//     req.input_tokens = input_tokens;
//     req.image_data = image_data;
//     req.kv_caches = kv_caches;
//     req.batch_size = batch_size;
//     req.ntok = batch_size; // 简化：假设每个请求只有一个token
//     req.nreq = 1; // 简化：假设只有一个请求
//     req.output = output;

//     // 2. 启动所有设备线程
//     auto ndev = dev_resources.size();
//     for (size_t i = 0; i < ndev; i++) {
//         std::unique_lock<std::mutex> lock(states[i].mtx);
//         states[i].proceed = true;
//         lock.unlock();
//         states[i].cv_stage.notify_one(); // 发出推理开始信号
//     }

//     // 3. 等待所有设备完成
//     for (size_t i = 0; i < ndev; i++) {
//         std::unique_lock<std::mutex> lock(states[i].mtx);
//         states[i].cv_stage.wait(lock, [&] { return !(states[i].proceed); });
//         lock.unlock();
//     }

//     // 4. 清理请求参数
//     req.input_tokens = nullptr;
//     req.image_data = nullptr;
//     req.kv_caches = nullptr;
//     req.output = nullptr;
// }

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

// C API: 批量视觉编码（用于Python接口）
__C void inferBatchLlavaVison(struct LlavaModel *model,
                           const void *image_data,
                           void *output) {
    if (!model || !image_data || !output) {
        return;
    }

    // 1. 设置推理参数（模仿inferBatchJiuge）
    // TODO: 感觉这里的req结构可能要逐渐改的像 struct InferRequest 
    model->req.input_tokens = nullptr;  // vision encoding不需要input_tokens
    model->req.image_data = image_data;
    model->req.kv_caches = nullptr;     // vision encoding不需要kv_caches
    model->req.batch_size = 1;          // 简化：假设batch_size为1
    model->req.ntok = 0;               // vision encoding不需要tokens
    model->req.nreq = 1;               // 简化：假设一个请求
    model->req.output = (uint32_t*)output;  // 将output转换为uint32_t指针

    //////////////////////////////////////////////
    auto vision_embed_dim = model->meta.vision_meta.vision_embed_dim;
    auto num_patches = model->meta.vision_meta.num_patches;
    auto total_features = vision_embed_dim * num_patches;
    printf("inferBatchLlavaVison called: image_data=%p, output=%p\n", image_data, output);
    printf("Vision config: embed_dim=%zu, num_patches=%zu, total_features=%zu\n",
           vision_embed_dim, num_patches, total_features);
    //////////////////////////////////////////////


    // 2. 通知所有设备线程开始工作（模仿inferBatchJiuge）
    // TODO: 注意，和jiuge不一样的地方在于，我们这里现在只有一个信号量
    for (size_t idev = 0; idev < model->dev_ids.size(); idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].proceed = true;  // 设置信号
        lock.unlock();
        model->states[idev].cv_stage.notify_one();  // 唤醒线程（LLaVA使用cv_stage）
    }

    // 3. 等待所有设备线程完成工作（模仿inferBatchJiuge）
    for (size_t i = model->dev_ids.size(); i > 0; i--) {
        auto idev = i - 1;
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].cv_stage.wait(lock, [&] { return !(model->states[idev].proceed); });
        lock.unlock();
    }

    printf("inferBatchLlavaVison: vision encoding completed\n");
}

// 暂时注释掉其他复杂的API函数，只保留最基本的