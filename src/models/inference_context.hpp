// inference_context.hpp
#pragma once

#include "cache_manager.hpp"
#include "jiuge/jiuge_impl.hpp"
#include "jiuge/jiuge_weight.hpp"

struct InferenceContext {
    DeviceResource *rsrc;
    CacheManager *cache_manager;
    infinirtStream_t stream;
    std::shared_ptr<Storage> workspace_storage;
    size_t current_workspace_size = 0;

    InferenceContext(DeviceResource *rsrc, CacheManager *cache_manager, infinirtStream_t stream);

    void ensure_workspace(size_t required_size);
    void rmsnorm(std::shared_ptr<Tensor> y,
                 std::shared_ptr<Tensor> x,
                 std::shared_ptr<Tensor> w,
                 float epsilon);
    void gemm(std::shared_ptr<Tensor> c, std::shared_ptr<TensorDesc> c_desc_overwrite,
              std::shared_ptr<Tensor> a, std::shared_ptr<TensorDesc> a_desc_overwrite,
              std::shared_ptr<Tensor> b, std::shared_ptr<TensorDesc> b_desc_overwrite,
              float alpha, float beta);
    void rearrange(std::shared_ptr<Tensor> dst, std::shared_ptr<TensorDesc> dst_desc_overwrite,
                   std::shared_ptr<Tensor> src, std::shared_ptr<TensorDesc> src_desc_overwrite);
    void rope(std::shared_ptr<Tensor> q,
              std::shared_ptr<Tensor> k,
              std::shared_ptr<Tensor> pos,
              std::shared_ptr<Tensor> sin,
              std::shared_ptr<Tensor> cos);
    void causalSoftmax(std::shared_ptr<Tensor> y, std::shared_ptr<TensorDesc> y_desc_overwrite,
                       std::shared_ptr<Tensor> x, std::shared_ptr<TensorDesc> x_desc_overwrite);
    void swiglu(std::shared_ptr<Tensor> out, std::shared_ptr<Tensor> up, std::shared_ptr<Tensor> gate);
    void randomSample(std::shared_ptr<Tensor> out, std::shared_ptr<TensorDesc> out_desc_overwrite,
                      std::shared_ptr<Tensor> prob, std::shared_ptr<TensorDesc> prob_desc_overwrite,
                      float random_val, float top_p, uint32_t top_k, float temperature);
};
