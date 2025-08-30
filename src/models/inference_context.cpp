#include "inference_context.hpp"
#include "../tensor.hpp"
#include "../utils.hpp"
#include <numeric>
#include <functional>

InferenceContext::InferenceContext(DeviceResource *rsrc, CacheManager *cache_manager, infinirtStream_t stream)
    : rsrc(rsrc), cache_manager(cache_manager), stream(stream) {}

void InferenceContext::ensure_workspace(size_t required_size) {
    if (required_size > current_workspace_size || !workspace_storage) {
        workspace_storage = Storage::createFromPool(required_size, rsrc->memory_pool);
        current_workspace_size = required_size;
    }
}

void InferenceContext::add(std::shared_ptr<Tensor> c,
                           std::shared_ptr<Tensor> a,
                           std::shared_ptr<Tensor> b) {
    size_t key = CacheManager::createDescriptorKey(c, a, b);

    infiniopAddDescriptor_t desc;
    if (!cache_manager->getAddDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateAddDescriptor(rsrc->handle, &desc, c->desc(), a->desc(), b->desc()));
        cache_manager->putAddDescriptor(key, desc);
    }

    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetAddWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();

    RUN_INFINI(infiniopAdd(
        desc, workspace, workspace_size,
        c->data(), a->data(), b->data(), stream));
}

void InferenceContext::rmsnorm(std::shared_ptr<Tensor> y,
                               std::shared_ptr<Tensor> x,
                               std::shared_ptr<Tensor> w,
                               float epsilon) {
    size_t key = CacheManager::createDescriptorKey(y, x, w);

    infiniopRMSNormDescriptor_t desc;
    if (!cache_manager->getRMSNormDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateRMSNormDescriptor(
            rsrc->handle, &desc, y->desc(), x->desc(), w->desc(), epsilon));
        cache_manager->putRMSNormDescriptor(key, desc);
    }

    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();

    RUN_INFINI(infiniopRMSNorm(
        desc, workspace, workspace_size,
        y->data(), x->data(), w->data(), stream));
}

void InferenceContext::gemm(std::shared_ptr<Tensor> c,
                            std::shared_ptr<Tensor> a,
                            std::shared_ptr<Tensor> b,
                            float alpha, float beta) {
    size_t key = CacheManager::createDescriptorKey(c, a, b);

    infiniopGemmDescriptor_t desc;
    if (!cache_manager->getGemmDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateGemmDescriptor(rsrc->handle, &desc, c->desc(), a->desc(), b->desc()));
        cache_manager->putGemmDescriptor(key, desc);
    }

    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();

    RUN_INFINI(infiniopGemm(
        desc, workspace, workspace_size,
        c->data(), a->data(), b->data(), alpha, beta, stream));
}

void InferenceContext::rearrange(std::shared_ptr<Tensor> dst,
                                 std::shared_ptr<Tensor> src) {
    size_t key = CacheManager::createDescriptorKey(dst, src);

    infiniopRearrangeDescriptor_t desc;
    if (!cache_manager->getRearrangeDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateRearrangeDescriptor(rsrc->handle, &desc, dst->desc(), src->desc()));
        cache_manager->putRearrangeDescriptor(key, desc);
    }

    RUN_INFINI(infiniopRearrange(
        desc,
        dst->data(),
        src->data(),
        stream));
}

void InferenceContext::rope(std::shared_ptr<Tensor> q,
                            std::shared_ptr<Tensor> k,
                            std::shared_ptr<Tensor> pos,
                            std::shared_ptr<Tensor> sin,
                            std::shared_ptr<Tensor> cos) {
    size_t key = CacheManager::createDescriptorKey(q, k, pos, sin, cos);

    infiniopRoPEDescriptor_t desc;
    if (!cache_manager->getRoPEDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateRoPEDescriptor(
            rsrc->handle, &desc, q->desc(), k->desc(),
            pos->desc(), sin->desc(), cos->desc()));
        cache_manager->putRoPEDescriptor(key, desc);
    }

    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetRoPEWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();

    RUN_INFINI(infiniopRoPE(
        desc, workspace, workspace_size,
        q->data(), k->data(), pos->data(),
        sin->data(), cos->data(), stream));
}

void InferenceContext::causalSoftmax(std::shared_ptr<Tensor> y,
                                     std::shared_ptr<Tensor> x) {
    size_t key = CacheManager::createDescriptorKey(y, x);

    infiniopCausalSoftmaxDescriptor_t desc;
    if (!cache_manager->getCausalSoftmaxDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateCausalSoftmaxDescriptor(
            rsrc->handle, &desc, y->desc(), x->desc()));
        cache_manager->putCausalSoftmaxDescriptor(key, desc);
    }

    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetCausalSoftmaxWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();

    RUN_INFINI(infiniopCausalSoftmax(desc, workspace, workspace_size,
                                     y->data(), x->data(), stream));
}

void InferenceContext::swiglu(std::shared_ptr<Tensor> out,
                              std::shared_ptr<Tensor> up,
                              std::shared_ptr<Tensor> gate) {
    size_t key = CacheManager::createDescriptorKey(out, up, gate);

    infiniopSwiGLUDescriptor_t desc;
    if (!cache_manager->getSwiGLUDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateSwiGLUDescriptor(
            rsrc->handle, &desc, out->desc(), up->desc(), gate->desc()));
        cache_manager->putSwiGLUDescriptor(key, desc);
    }

    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetSwiGLUWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();

    RUN_INFINI(infiniopSwiGLU(desc, workspace, workspace_size,
                              out->data(), up->data(), gate->data(), stream));
}

void InferenceContext::randomSample(std::shared_ptr<Tensor> out,
                                    std::shared_ptr<Tensor> prob,
                                    float random_val, float top_p, uint32_t top_k, float temperature) {
    size_t key = CacheManager::createDescriptorKey(out, prob);

    infiniopRandomSampleDescriptor_t desc;
    if (!cache_manager->getRandomSampleDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateRandomSampleDescriptor(
            rsrc->handle, &desc, out->desc(), prob->desc()));
        cache_manager->putRandomSampleDescriptor(key, desc);
    }

    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetRandomSampleWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();

    RUN_INFINI(infiniopRandomSample(
        desc, workspace, workspace_size,
        out->data(), prob->data(),
        random_val, top_p, top_k, temperature,
        stream));
}

void InferenceContext::linear(std::shared_ptr<Tensor> c,
                              std::shared_ptr<Tensor> a,
                              std::shared_ptr<Tensor> b,
                              float alpha, float beta,
                              std::shared_ptr<Tensor> residual,
                              std::shared_ptr<Tensor> bias) {
    bool residual_flag = residual != nullptr;

    if (bias && !residual) {
        int ndim_diff = c->ndim() - 1;
        ASSERT_EQ(bias->ndim(), 1);
        ASSERT_EQ(bias->shape()[0], c->shape()[ndim_diff]);
        std::vector<ptrdiff_t> strides(ndim_diff, 0);
        strides.push_back(bias->strides()[0]);
        rearrange(c, bias->view_as(c->shape(), strides));
        residual = c;
    }

    if (residual) {
        if (residual->data() == c->data()) {
            if (beta == 0.0) {
                gemm(c, a, b, alpha, 1.0);
            } else {
                auto c_copy = Tensor::buffer(c->dtype(), c->shape(), rsrc->memory_pool);
                c_copy->copyFrom(c, rsrc->handle, stream);
                gemm(c, a, b, alpha, beta);
                add(c, c, c_copy);
            }
        } else {
            gemm(c, a, b, alpha, beta);
            add(c, c, residual);
        }
    } else {
        gemm(c, a, b, alpha, beta);
    }

    if (bias && residual_flag) {
        int ndim_diff = c->ndim() - 1;
        ASSERT_EQ(bias->ndim(), 1);
        ASSERT_EQ(bias->shape()[0], c->shape()[ndim_diff]);
        std::vector<ptrdiff_t> strides(ndim_diff, 0);
        strides.push_back(bias->strides()[0]);
        add(c, c, bias->view_as(c->shape(), strides));
    }
}

void InferenceContext::gather(std::shared_ptr<Tensor> output,
                              std::shared_ptr<Tensor> input,
                              const std::vector<uint32_t> &indices,
                              int dim) {
    // 1. 准备索引张量：将 CPU 上的 vector 索引上传到 GPU
    auto index_tensor = Tensor::buffer(INFINI_DTYPE_I32, output->shape(), rsrc->memory_pool);
    RUN_INFINI(infinirtMemcpyAsync(index_tensor->data(), indices.data(), indices.size() * sizeof(uint32_t),
                                   INFINIRT_MEMCPY_H2D, stream));

    // 2. 创建描述符 (并利用缓存)
    size_t key = CacheManager::createDescriptorKey(output, input, index_tensor);
    infiniopGatherDescriptor_t desc;
    if (!cache_manager->getGatherDescriptor(key, desc)) {
    RUN_INFINI(infiniopCreateGatherDescriptor(
        rsrc->handle, &desc, output->desc(), input->desc(), dim, index_tensor->desc()));
        cache_manager->putGatherDescriptor(key, desc);
    }

    // 3. 准备工作空间
    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetGatherWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();

    // 4. 执行 Gather 操作
    RUN_INFINI(infiniopGather(
        desc, workspace, workspace_size,
        output->data(), input->data(), index_tensor->data(), stream));
}

void InferenceContext::scatter_add(std::shared_ptr<Tensor> target,
                                   std::shared_ptr<Tensor> source,
                                   const std::vector<uint32_t> &indices,
                                   int dim) {

    // 使用 Gather-Add-Scatter 模式实现
    
    // 1. 准备索引张量 (与 gather 共享)
    auto index_tensor = Tensor::buffer(INFINI_DTYPE_I32, source->shape(), rsrc->memory_pool);
    RUN_INFINI(infinirtMemcpyAsync(index_tensor->data(), indices.data(), indices.size() * sizeof(uint32_t),
                                   INFINIRT_MEMCPY_H2D, stream));

    // 2. Gather: 从 target 中取出需要更新的原始值
    auto original_values = Tensor::buffer(source->dtype(), source->shape(), rsrc->memory_pool);
    gather(original_values, target, indices, dim);

    // 3. Add: 将 source (新值) 和 original_values (原始值) 相加
    auto updated_values = Tensor::buffer(source->dtype(), source->shape(), rsrc->memory_pool);
    add(updated_values, original_values, source);

    // 4. Scatter: 将相加后的结果写回 target 的原始位置
    // 创建描述符
    size_t key = CacheManager::createDescriptorKey(target, updated_values, index_tensor);
    infiniopScatterDescriptor_t desc;
    if (!cache_manager->getScatterDescriptor(key, desc)) {
RUN_INFINI(infiniopCreateScatterDescriptor(
    rsrc->handle, &desc, target->desc(), target->desc(), updated_values->desc(), index_tensor->desc(), dim));
        cache_manager->putScatterDescriptor(key, desc);
    }

    // 准备工作空间
    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetScatterWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();

    // 执行 Scatter 操作
    RUN_INFINI(infiniopScatter(
        desc, workspace, workspace_size,
        target->data(), updated_values->data(), index_tensor->data(), source->data(), stream));
}


void InferenceContext::scale(std::shared_ptr<Tensor> y,
                               std::shared_ptr<Tensor> x,
                               float alpha) {
    // 使用gemm实现标量缩放: y = alpha * x
    if (y.get() != x.get()) {
        size_t x_nelem = std::accumulate(x->shape().begin(), x->shape().end(), 1ULL, std::multiplies<size_t>());
        RUN_INFINI(infinirtMemcpyAsync(y->data(), x->data(), 
                                       x_nelem * dsize(x->dtype()), 
                                       INFINIRT_MEMCPY_D2D, stream));
    }
    
    // 使用gemm实现缩放: y = alpha * y + 0 * y
    auto ones = Tensor::buffer(x->dtype(), {1, 1}, rsrc->memory_pool);
    float one_value = 1.0f;
    RUN_INFINI(infinirtMemcpyAsync(ones->data(), &one_value, sizeof(float), INFINIRT_MEMCPY_H2D, stream));
    
    size_t total_elements = std::accumulate(x->shape().begin(), x->shape().end(), 1ULL, std::multiplies<size_t>());
    auto y_flat = y->view({total_elements, 1});
    gemm(y_flat, y_flat, ones, alpha, 0.0f);
}

void InferenceContext::scale(std::shared_ptr<Tensor> y,
                               std::shared_ptr<Tensor> x,
                               const std::vector<float> &weights) {
    // 先复制数据
    if (y.get() != x.get()) {
        size_t x_nelem = std::accumulate(x->shape().begin(), x->shape().end(), 1ULL, std::multiplies<size_t>());
        RUN_INFINI(infinirtMemcpyAsync(y->data(), x->data(), 
                                       x_nelem * dsize(x->dtype()), 
                                       INFINIRT_MEMCPY_D2D, stream));
    }
    
    // 为每个token应用对应的权重
    size_t num_tokens = weights.size();
    size_t d = y->shape()[1]; // hidden dimension
    
    for (size_t i = 0; i < num_tokens; ++i) {
        auto token_output = y->slice(0, i, 1); // 取出第i个token的输出
        auto ones = Tensor::buffer(y->dtype(), {1, 1}, rsrc->memory_pool);
        float one_value = 1.0f;
        RUN_INFINI(infinirtMemcpyAsync(ones->data(), &one_value, sizeof(float), INFINIRT_MEMCPY_H2D, stream));
        
        auto token_flat = token_output->view({d, 1});
        gemm(token_flat, token_flat, ones, weights[i], 0.0f);
    }
}

void InferenceContext::zeros(std::shared_ptr<Tensor> t) {
    // 暂时使用简单的临时实现，将tensor的所有值设为0
    // 创建一个同样大小的零值tensor，然后复制过去
    size_t nelem = std::accumulate(t->shape().begin(), t->shape().end(), 1ULL, std::multiplies<size_t>());
    std::vector<float> zero_data(nelem, 0.0f);
    
    if (t->dtype() == INFINI_DTYPE_F32) {
        RUN_INFINI(infinirtMemcpyAsync(t->data(), zero_data.data(), 
                                       nelem * sizeof(float), 
                                       INFINIRT_MEMCPY_H2D, stream));
    } else {
        // 对于其他数据类型，暂时跳过实现
        // 在实际使用中可能需要根据dtype进行转换
    }
}

void InferenceContext::normalize(std::shared_ptr<Tensor> y,
                                   std::shared_ptr<Tensor> x,
                                   int dim,
                                   float epsilon) {
    // normalize算子是就地操作，先复制x到y
    if (y.get() != x.get()) {
        size_t x_nelem = std::accumulate(x->shape().begin(), x->shape().end(), 1ULL, std::multiplies<size_t>());
        RUN_INFINI(infinirtMemcpyAsync(y->data(), x->data(), 
                                       x_nelem * dsize(x->dtype()), 
                                       INFINIRT_MEMCPY_D2D, stream));
    }
    
    size_t key = CacheManager::createDescriptorKey(y);

    infiniopNormalizeDescriptor_t desc;
    if (!cache_manager->getNormalizeDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateNormalizeDescriptor(
            rsrc->handle, &desc, y->desc()));
        cache_manager->putNormalizeDescriptor(key, desc);
    }

    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetNormalizeWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();

    RUN_INFINI(infiniopNormalize(
        desc, workspace, workspace_size,
        y->data(), stream));
}

void InferenceContext::topk_fun(std::shared_ptr<Tensor> values,
                              std::shared_ptr<Tensor> indices,
                              std::shared_ptr<Tensor> input,
                              uint32_t k,
                              int dim) {
    size_t key = CacheManager::createDescriptorKey(values, indices, input);

    infiniopTopKDescriptor_t desc;
    if (!cache_manager->getTopKDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateTopKDescriptor(
            rsrc->handle, &desc, input->desc(), values->desc(), indices->desc(), k, dim, true, true));
        cache_manager->putTopKDescriptor(key, desc);
    }

    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetTopKWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();

    RUN_INFINI(infiniopTopK(
        desc, workspace, workspace_size,
        input->data(), values->data(), indices->data(), stream));
}