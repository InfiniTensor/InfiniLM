#include "inference_context.hpp"
#include "../tensor.hpp"
#include "../utils.hpp"

InferenceContext::InferenceContext(DeviceResource *rsrc, CacheManager *cache_manager, infinirtStream_t stream)
    : rsrc(rsrc), cache_manager(cache_manager), stream(stream) {}

void InferenceContext::ensure_workspace(size_t required_size) {
    if (required_size > current_workspace_size) {
        workspace_storage = Storage::createFromPool(required_size, rsrc->memory_pool);
        current_workspace_size = required_size;
    }
}

void InferenceContext::rmsnorm(std::shared_ptr<Tensor> y,
                               std::shared_ptr<Tensor> x,
                               std::shared_ptr<Tensor> w,
                               float epsilon) {
    size_t key = CacheManager::createDescriptorKey(y->tdesc(), x->tdesc(), w->tdesc(), nullptr, nullptr);

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

void InferenceContext::gemm(std::shared_ptr<Tensor> c, std::shared_ptr<TensorDesc> c_desc_overwrite,
                            std::shared_ptr<Tensor> a, std::shared_ptr<TensorDesc> a_desc_overwrite,
                            std::shared_ptr<Tensor> b, std::shared_ptr<TensorDesc> b_desc_overwrite,
                            float alpha, float beta) {
    size_t key = CacheManager::createDescriptorKey(
        c_desc_overwrite ? c_desc_overwrite : c->tdesc(),
        a_desc_overwrite ? a_desc_overwrite : a->tdesc(),
        b_desc_overwrite ? b_desc_overwrite : b->tdesc(),
        nullptr, nullptr);

    infiniopGemmDescriptor_t desc;
    if (!cache_manager->getGemmDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateGemmDescriptor(
            rsrc->handle, &desc,
            c_desc_overwrite ? c_desc_overwrite->desc() : c->desc(),
            a_desc_overwrite ? a_desc_overwrite->desc() : a->desc(),
            b_desc_overwrite ? b_desc_overwrite->desc() : b->desc()));
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

void InferenceContext::rearrange(std::shared_ptr<Tensor> dst, std::shared_ptr<TensorDesc> dst_desc_overwrite,
                                 std::shared_ptr<Tensor> src, std::shared_ptr<TensorDesc> src_desc_overwrite) {
    size_t key = CacheManager::createDescriptorKey(
        dst_desc_overwrite ? dst_desc_overwrite : dst->tdesc(),
        src_desc_overwrite ? src_desc_overwrite : src->tdesc(),
        nullptr, nullptr, nullptr);

    infiniopRearrangeDescriptor_t desc;
    if (!cache_manager->getRearrangeDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateRearrangeDescriptor(
            rsrc->handle, &desc,
            dst_desc_overwrite ? dst_desc_overwrite->desc() : dst->desc(),
            src_desc_overwrite ? src_desc_overwrite->desc() : src->desc()));
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
    size_t key = CacheManager::createDescriptorKey(q->tdesc(), k->tdesc(), pos->tdesc(), sin->tdesc(), cos->tdesc());

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

void InferenceContext::causalSoftmax(std::shared_ptr<Tensor> y, std::shared_ptr<TensorDesc> y_desc_overwrite,
                                     std::shared_ptr<Tensor> x, std::shared_ptr<TensorDesc> x_desc_overwrite) {
    size_t key = CacheManager::createDescriptorKey(
        y_desc_overwrite ? y_desc_overwrite : y->tdesc(),
        x_desc_overwrite ? x_desc_overwrite : x->tdesc(),
        nullptr, nullptr, nullptr);

    infiniopCausalSoftmaxDescriptor_t desc;
    if (!cache_manager->getCausalSoftmaxDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateCausalSoftmaxDescriptor(
            rsrc->handle, &desc,
            y_desc_overwrite ? y_desc_overwrite->desc() : y->desc(),
            x_desc_overwrite ? x_desc_overwrite->desc() : x->desc()));
        cache_manager->putCausalSoftmaxDescriptor(key, desc);
    }

    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetCausalSoftmaxWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();

    RUN_INFINI(infiniopCausalSoftmax(desc, workspace, workspace_size,
                                     y->data(), x->data(), stream));
}

void InferenceContext::swiglu(std::shared_ptr<Tensor> out, std::shared_ptr<Tensor> up, std::shared_ptr<Tensor> gate) {
    size_t key = CacheManager::createDescriptorKey(out->tdesc(), up->tdesc(), gate->tdesc(), nullptr, nullptr);

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

void InferenceContext::randomSample(std::shared_ptr<Tensor> out, std::shared_ptr<TensorDesc> out_desc_overwrite,
                                    std::shared_ptr<Tensor> prob, std::shared_ptr<TensorDesc> prob_desc_overwrite,
                                    float random_val, float top_p, uint32_t top_k, float temperature) {
    size_t key = CacheManager::createDescriptorKey(
        out_desc_overwrite ? out_desc_overwrite : out->tdesc(),
        prob_desc_overwrite ? prob_desc_overwrite : prob->tdesc(),
        nullptr, nullptr, nullptr);

    infiniopRandomSampleDescriptor_t desc;
    if (!cache_manager->getRandomSampleDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateRandomSampleDescriptor(
            rsrc->handle, &desc,
            out_desc_overwrite ? out_desc_overwrite->desc() : out->desc(),
            prob_desc_overwrite ? prob_desc_overwrite->desc() : prob->desc()));
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
