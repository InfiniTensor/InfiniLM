#pragma once

#include "cache_manager.hpp"
#include "jiuge/jiuge_impl.hpp"
#include "jiuge/jiuge_weight.hpp"
#include <cassert>

struct InferenceContext {
    DeviceResource *rsrc;
    CacheManager *cache_manager;
    infinirtStream_t stream;
    std::shared_ptr<Storage> workspace_storage;
    size_t current_workspace_size = 0;

    InferenceContext(DeviceResource *rsrc, CacheManager *cache_manager, infinirtStream_t stream);

    void ensure_workspace(size_t required_size);

    void add(std::shared_ptr<Tensor> c,
             std::shared_ptr<Tensor> a,
             std::shared_ptr<Tensor> b);
    void rmsnorm(std::shared_ptr<Tensor> y,
                 std::shared_ptr<Tensor> x,
                 std::shared_ptr<Tensor> w,
                 float epsilon);
    void gemm(std::shared_ptr<Tensor> c,
              std::shared_ptr<Tensor> a,
              std::shared_ptr<Tensor> b,
              float alpha, float beta);
    void rearrange(std::shared_ptr<Tensor> dst,
                   std::shared_ptr<Tensor> src);
    void rope(std::shared_ptr<Tensor> q,
              std::shared_ptr<Tensor> k,
              std::shared_ptr<Tensor> pos,
              std::shared_ptr<Tensor> sin,
              std::shared_ptr<Tensor> cos);
    void causalSoftmax(std::shared_ptr<Tensor> y,
                       std::shared_ptr<Tensor> x);
    void swiglu(std::shared_ptr<Tensor> out,
                std::shared_ptr<Tensor> up,
                std::shared_ptr<Tensor> gate);
    void randomSample(std::shared_ptr<Tensor> out,
                      std::shared_ptr<Tensor> prob,
                      float random_val, float top_p, uint32_t top_k, float temperature);

    void linear(std::shared_ptr<Tensor> c,
                std::shared_ptr<Tensor> a,
                std::shared_ptr<Tensor> b,
                float alpha, float beta,
                std::shared_ptr<Tensor> residual,
                std::shared_ptr<Tensor> bias);
};

namespace {
thread_local InferenceContext *tls_inference_context = nullptr;
}

inline InferenceContext &getInferenceContext() {
    assert(tls_inference_context != nullptr && "InferenceContext not set for this thread");
    return *tls_inference_context;
}

inline void setInferenceContext(InferenceContext *ctx) {
    tls_inference_context = ctx;
}

inline void add(std::shared_ptr<Tensor> c, std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    getInferenceContext().add(c, a, b);
}

inline void rmsnorm(std::shared_ptr<Tensor> y, std::shared_ptr<Tensor> x,
                    std::shared_ptr<Tensor> w, float epsilon) {
    getInferenceContext().rmsnorm(y, x, w, epsilon);
}

inline void gemm(std::shared_ptr<Tensor> c, std::shared_ptr<Tensor> a,
                 std::shared_ptr<Tensor> b, float alpha, float beta) {
    getInferenceContext().gemm(c, a, b, alpha, beta);
}

inline void rearrange(std::shared_ptr<Tensor> dst, std::shared_ptr<Tensor> src) {
    getInferenceContext().rearrange(dst, src);
}

inline void rope(std::shared_ptr<Tensor> q, std::shared_ptr<Tensor> k,
                 std::shared_ptr<Tensor> pos, std::shared_ptr<Tensor> sin,
                 std::shared_ptr<Tensor> cos) {
    getInferenceContext().rope(q, k, pos, sin, cos);
}

inline void causalSoftmax(std::shared_ptr<Tensor> y, std::shared_ptr<Tensor> x) {
    getInferenceContext().causalSoftmax(y, x);
}

inline void swiglu(std::shared_ptr<Tensor> out, std::shared_ptr<Tensor> up,
                   std::shared_ptr<Tensor> gate) {
    getInferenceContext().swiglu(out, up, gate);
}

inline void randomSample(std::shared_ptr<Tensor> out, std::shared_ptr<Tensor> prob,
                         float random_val, float top_p, uint32_t top_k, float temperature) {
    getInferenceContext().randomSample(out, prob, random_val, top_p, top_k, temperature);
}

inline void linear(std::shared_ptr<Tensor> c, std::shared_ptr<Tensor> a,
                   std::shared_ptr<Tensor> b, float alpha, float beta,
                   std::shared_ptr<Tensor> residual, std::shared_ptr<Tensor> bias) {
    getInferenceContext().linear(c, a, b, alpha, beta, residual, bias);
}
