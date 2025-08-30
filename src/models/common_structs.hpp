#ifndef COMMON_STRUCTS_H
#define COMMON_STRUCTS_H

#include "../tensor.hpp" // KVCache depends on Tensor
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <cstdint> // For uint32_t

// These structs are generic and can be shared between dense and MoE models.

struct InferState {
    std::mutex mtx;
    std::condition_variable cv_load, cv_start, cv_done;
    bool loaded = false;
    bool proceed = false;
    bool exit_flag = false;
};

struct InferRequest {
    const uint32_t *tokens;
    uint32_t ntok;
    const uint32_t *req_lens;
    uint32_t nreq;
    const uint32_t *req_pos;
    struct KVCache **kv_caches;
    const float *temperature;
    const uint32_t *topk;
    const float *topp;
    uint32_t *output;
    void *logits;
};

struct KVCache {
    std::vector<std::vector<std::shared_ptr<Tensor>>> k, v;
};

#endif // COMMON_STRUCTS_H
