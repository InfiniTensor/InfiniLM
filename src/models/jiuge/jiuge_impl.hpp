#ifndef JIUGE_IMPL_H
#define JIUGE_IMPL_H

#include "infinicore_infer.h"

#include "../../allocator.hpp"
#include "../../tensor.hpp"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

struct JiugeDeviceResource {
    // Device
    infiniDevice_t device;
    int device_id;
    infiniopHandle_t handle;
    // Weights
    std::shared_ptr<Tensor> w_in_embd, w_out_norm, w_out_embd, sin_table,
        cos_table;
    std::vector<std::shared_ptr<Tensor>> w_attn_norm, w_attn_qkv, b_attn_qkv, w_attn_q_norm, w_attn_k_norm,w_attn_out,
        w_ffn_norm, w_ffn_gate_up, w_ffn_down;
    // Streams
    infinirtStream_t stream;
    // Communicator
    infinicclComm_t comm;

    std::shared_ptr<MemoryPool> memory_pool;
};

struct InferState {
    std::mutex mtx;
    std::condition_variable cv_load, cv_start, cv_done;
    bool loaded = false;
    bool proceed = false;
    bool exit_flag = false;
};

// 1. mtx (互斥锁)
// 作用: 保护每个线程自己的状态变量访问
// 特点: 每个线程有独立的mutex，避免线程间竞争
// 使用场景: 在修改loaded、proceed、exit_flag时加锁

// 2. cv_load (加载完成条件变量)
// 作用: 设备线程等待主线程的加载信号
// 流程:
// 设备线程：cv_load.wait() - 等待任务
// 主线程：cv_load.notify_one() - 分配任务
// 语义: "数据已准备好，可以开始执行"

// 3. cv_start (开始执行条件变量)
// 作用: 控制设备线程开始执行推理
// 流程:
// 主线程：cv_start.notify_one() - 允许开始
// 设备线程：cv_start.wait() - 等待开始许可
// 语义: "可以开始执行推理"

// 4. cv_done (执行完成条件变量)
// 作用: 设备线程通知主线程任务完成
// 流程:
// 设备线程：cv_done.notify_one() - 报告完成
// 主线程：cv_done.wait() - 等待完成
// 语义: "推理已完成"

// 5. loaded (加载状态标志)
// 作用: 标识任务数据是否已加载完成
// 值: false → true (主线程设置)，true → false (设备线程重置)

// 6. proceed (执行状态标志)
// 作用: 控制是否允许继续执行下一步
// 值: false → true (主线程授权执行)

// 7. exit_flag (退出标志)
// 作用: 通知线程优雅退出
// 值: false → true (程序结束时设置)



struct InferRequest {
    const uint32_t *tokens;
    uint32_t ntok;
    const uint32_t *req_lens;
    uint32_t nreq;
    const uint32_t *req_pos;
    const uint32_t *kv_pos;
    struct KVCache **kv_caches;
    uint32_t n_override;
    const uint32_t *override_pos;
    const void *override_embeds;
    const float *temperature;
    const uint32_t *topk;
    const float *topp;
    uint32_t *output;
    void *logits;
};

struct JiugeModel {
    JiugeMeta meta;
    infiniDevice_t device;
    std::vector<int> dev_ids;
    std::vector<JiugeDeviceResource> dev_resources;
    std::vector<InferState> states;
    std::vector<std::thread> threads;
    InferRequest req;

    JiugeModel(const JiugeMeta *, const JiugeWeights *, infiniDevice_t device, std::vector<int> device_ids);
};

#include "../../cache.hpp"

#endif
