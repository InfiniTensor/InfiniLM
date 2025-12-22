#include "inference_context.hpp"
#include "../tensor.hpp"
#include "../utils.hpp"
#include <atomic>

thread_local InferenceContext *tls_inference_context = nullptr;

InferenceContext::InferenceContext(infiniopHandle_t op_handle_, std::shared_ptr<MemoryPool> memory_pool_, CacheManager *cache_manager, infinirtStream_t stream)
    : op_handle(op_handle_), memory_pool(memory_pool_), cache_manager(cache_manager), stream(stream) {}

void InferenceContext::ensure_workspace(size_t required_size) {
    if (required_size > current_workspace_size || !workspace_storage) {
        workspace_storage = Storage::createFromPool(required_size, memory_pool);
        current_workspace_size = required_size;
    }
}

void InferenceContext::add(std::shared_ptr<Tensor> c,
                           std::shared_ptr<Tensor> a,
                           std::shared_ptr<Tensor> b) {
    size_t key = CacheManager::createDescriptorKey(c, a, b);

    infiniopAddDescriptor_t desc;
    if (!cache_manager->getAddDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateAddDescriptor(op_handle, &desc, c->desc(), a->desc(), b->desc()));
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
            op_handle, &desc, y->desc(), x->desc(), w->desc(), epsilon));
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

void InferenceContext::layernorm(std::shared_ptr<Tensor> output,
                                 std::shared_ptr<Tensor> input_standardization,
                                 std::shared_ptr<Tensor> input_std_deviation,
                                 std::shared_ptr<Tensor> input,
                                 std::shared_ptr<Tensor> weight,
                                 std::shared_ptr<Tensor> bias,
                                 float eps) {
    // 构造 descriptor key（把所有相关 tensor 都参与 key）
    size_t key = CacheManager::createDescriptorKey(
        output, input_standardization, input_std_deviation, input, weight, bias);

    infiniopLayerNormDescriptor_t desc;
    if (!cache_manager->getLayerNormDescriptor(key, desc)) {
        // create descriptor: 注意 eps 必须是最后一个参数（与绑定的 C API 一致）
        RUN_INFINI(infiniopCreateLayerNormDescriptor(
            op_handle,
            &desc,
            output->desc(),                    // output_desc
            input_standardization->desc(),     // input_standardization_desc
            input_std_deviation->desc(),       // input_std_deviation_desc
            input->desc(),                     // input_desc
            weight ? weight->desc() : nullptr, // weight_desc (gamma)
            bias ? bias->desc() : nullptr,     // bias_desc (beta) or nullptr
            eps                                 // epsilon (最后)
        ));
        cache_manager->putLayerNormDescriptor(key, desc);
    }

    // 获取 workspace 大小并确保 workspace 足够
    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetLayerNormWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();

    // 调用 kernel（最后一个参数是 stream）
    RUN_INFINI(infiniopLayerNorm(
        desc,
        workspace,
        workspace_size,
        output->data(),
        input_standardization->data(),
        input_std_deviation->data(),
        input->data(),
        weight ? weight->data() : nullptr,
        bias ? bias->data() : nullptr,
        stream));
}

void InferenceContext::gemm(std::shared_ptr<Tensor> c,
                            std::shared_ptr<Tensor> a,
                            std::shared_ptr<Tensor> b,
                            float alpha, float beta) {
    // printf("--------------------------------------------\n");
    // printf("%s\n", a->info().c_str()); // for debug
    // printf("%s\n", b->info().c_str()); // for debug
    // printf("%s\n", c->info().c_str()); // for debug
    size_t key = CacheManager::createDescriptorKey(c, a, b);

    infiniopGemmDescriptor_t desc;
    if (!cache_manager->getGemmDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateGemmDescriptor(op_handle, &desc, c->desc(), a->desc(), b->desc()));
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
        RUN_INFINI(infiniopCreateRearrangeDescriptor(op_handle, &desc, dst->desc(), src->desc()));
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
                            std::shared_ptr<Tensor> cos,
                            infiniopRoPEAlgo_t algo) {
    size_t key = CacheManager::createDescriptorKey(q, k, pos, sin, cos);
    hash_combine(key, std::hash<int>()(algo));

    infiniopRoPEDescriptor_t desc;
    if (!cache_manager->getRoPEDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateRoPEDescriptor(
            op_handle, &desc, q->desc(), k->desc(),
            pos->desc(), sin->desc(), cos->desc(), algo));
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
            op_handle, &desc, y->desc(), x->desc()));
        cache_manager->putCausalSoftmaxDescriptor(key, desc);
    }

    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetCausalSoftmaxWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();

    RUN_INFINI(infiniopCausalSoftmax(desc, workspace, workspace_size,
                                     y->data(), x->data(), stream));
}


void InferenceContext::Softmax(std::shared_ptr<Tensor> y, std::shared_ptr<Tensor> x, int axis) {
    size_t key = CacheManager::createDescriptorKey(y, x);
    hash_combine(key, std::hash<int>()(axis));

    infiniopSoftmaxDescriptor_t desc;
    if (!cache_manager->getSoftmaxDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateSoftmaxDescriptor(
            op_handle, &desc, y->desc(), x->desc(), axis));
        cache_manager->putSoftmaxDescriptor(key, desc);
    }

    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetSoftmaxWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();

    RUN_INFINI(infiniopSoftmax(desc, workspace, workspace_size,
                               y->data(), x->data(), stream));
}

void InferenceContext::topkrouter(std::shared_ptr<Tensor> values,  // F32
                                  std::shared_ptr<Tensor> indices, // I32
                                  std::shared_ptr<Tensor> x,
                                  std::shared_ptr<Tensor> correction_bias, // F32
                                  float routed_scaling_factor,
                                  size_t topk) {
    size_t key = CacheManager::createDescriptorKey(values, indices, x, correction_bias);

    infiniopTopkrouterDescriptor_t desc;
    if (!cache_manager->getTopkrouterDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateTopkrouterDescriptor(
            op_handle, &desc, x->desc(), correction_bias->desc()));
        cache_manager->putTopkrouterDescriptor(key, desc);
    }

    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetTopkrouterWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();

    RUN_INFINI(infiniopTopkrouter(desc, workspace, workspace_size,
                                  values->data(), indices->data(), x->data(), correction_bias->data(),
                                  routed_scaling_factor, topk, stream));
}

void InferenceContext::swiglu(std::shared_ptr<Tensor> out,
                              std::shared_ptr<Tensor> up,
                              std::shared_ptr<Tensor> gate) {
    size_t key = CacheManager::createDescriptorKey(out, up, gate);

    infiniopSwiGLUDescriptor_t desc;
    if (!cache_manager->getSwiGLUDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateSwiGLUDescriptor(
            op_handle, &desc, out->desc(), up->desc(), gate->desc()));
        cache_manager->putSwiGLUDescriptor(key, desc);
    }

    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetSwiGLUWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();

    RUN_INFINI(infiniopSwiGLU(desc, workspace, workspace_size,
                              out->data(), up->data(), gate->data(), stream));
}



void InferenceContext::relu(std::shared_ptr<Tensor> y,
                            std::shared_ptr<Tensor> x) {
    size_t key = CacheManager::createDescriptorKey(y, x);

    infiniopReluDescriptor_t desc;
    if (!cache_manager->getReluDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateReluDescriptor(op_handle, &desc, y->desc(), x->desc()));
        cache_manager->putReluDescriptor(key, desc);
    }

    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetReluWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();
    
    RUN_INFINI(infiniopRelu(desc,
                            workspace,
                            workspace_size,
                            y->data(), x->data(), stream));
}

void InferenceContext::geluTanh(std::shared_ptr<Tensor> y,
                                 std::shared_ptr<Tensor> x) {
    size_t key = CacheManager::createDescriptorKey(y, x);

    infiniopGeluTanhDescriptor_t desc;
    if (!cache_manager->getGeluTanhDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateGeluTanhDescriptor(op_handle, &desc, y->desc(), x->desc()));
        cache_manager->putGeluTanhDescriptor(key, desc);
    }

    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetGeluTanhWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();
    
    RUN_INFINI(infiniopGeluTanh(desc,
                                workspace,
                                workspace_size,
                                y->data(), x->data(), stream));
}

void InferenceContext::layerNorm(std::shared_ptr<Tensor> y,
                                 std::shared_ptr<Tensor> x,
                                 std::shared_ptr<Tensor> w,
                                 std::shared_ptr<Tensor> beta,
                                 float epsilon) {
    ASSERT_VALID_PTR(y);
    ASSERT_VALID_PTR(x);
    ASSERT_VALID_PTR(w);
    ASSERT_VALID_PTR(beta);

    // Some implementations do not support in-place LayerNorm (output aliases input).
    // Keep call sites simple by handling it here.
    std::shared_ptr<Tensor> y_out = y;
    std::shared_ptr<Tensor> y_tmp;
    if (y.get() == x.get() || y->data() == x->data()) {
        y_tmp = Tensor::buffer(y->dtype(), y->shape(), memory_pool);
        y_out = y_tmp;
    }

    // LayerNorm produces two extra outputs (standardization + std deviation). We don't
    // expose them to callers, but descriptors require them, so we allocate temporaries.
    //
    // Keep intermediates in the same dtype as input to support device execution (e.g. Hygon)
    // and avoid dtype-specific assumptions in backend implementations.
    const infiniDtype_t inter_dt = x->dtype();

    // CPU LayerNorm kernel assumes 3D input [B, L, D]. Adapt common 2D tensors [L, D]
    // into [1, L, D] via views to avoid out-of-bounds access.
    std::shared_ptr<Tensor> x_desc = x;
    std::shared_ptr<Tensor> y_desc = y_out;
    std::shared_ptr<Tensor> input_standardization;
    std::shared_ptr<Tensor> input_std_deviation;

    if (x->deviceType() == INFINI_DEVICE_CPU && x->ndim() == 2) {
        const auto &sh = x->shape();
        const auto &st = x->strides();
        const size_t L = sh[0];
        const size_t D = sh[1];
        const ptrdiff_t s0 = st[0];
        const ptrdiff_t s1 = st[1];
        x_desc = x->view_as({1, L, D}, {static_cast<ptrdiff_t>(L) * s0, s0, s1});
        y_desc = y_out->view_as({1, L, D}, {static_cast<ptrdiff_t>(L) * s0, s0, s1});
        input_standardization = Tensor::buffer(inter_dt, {1, L, D}, memory_pool);
        input_std_deviation = Tensor::buffer(inter_dt, {1, L}, memory_pool);
    } else {
        input_standardization = Tensor::buffer(inter_dt, x->shape(), memory_pool);
        std::vector<size_t> std_shape = x->shape();
        if (!std_shape.empty()) {
            std_shape.pop_back(); // stddev drops the normalized (last) dimension
        }
        if (std_shape.empty()) {
            std_shape.push_back(1);
        }
        input_std_deviation = Tensor::buffer(inter_dt, std_shape, memory_pool);
    }

    size_t key = CacheManager::createDescriptorKey(y_desc, x_desc, w, beta);
    uint32_t eps_bits = 0;
    std::memcpy(&eps_bits, &epsilon, sizeof(eps_bits));
    hash_combine(key, std::hash<uint32_t>()(eps_bits));

    infiniopLayerNormDescriptor_t desc;
    if (!cache_manager->getLayerNormDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateLayerNormDescriptor(
            op_handle, &desc,
            y_desc->desc(),
            input_standardization->desc(),
            input_std_deviation->desc(),
            x_desc->desc(),
            w->desc(),
            beta->desc(),
            epsilon));
        cache_manager->putLayerNormDescriptor(key, desc);
    }

    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetLayerNormWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();

    RUN_INFINI(infiniopLayerNorm(
        desc, workspace, workspace_size,
        y_desc->data(),
        input_standardization->data(),
        input_std_deviation->data(),
        x_desc->data(),
        w->data(),
        beta->data(),
        stream));

    if (y_tmp) {
        rearrange(y, y_tmp);
    }
}


void InferenceContext::randomSample(std::shared_ptr<Tensor> out,
                                    std::shared_ptr<Tensor> prob,
                                    float random_val, float top_p, uint32_t top_k, float temperature) {
    size_t key = CacheManager::createDescriptorKey(out, prob);

    infiniopRandomSampleDescriptor_t desc;
    if (!cache_manager->getRandomSampleDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateRandomSampleDescriptor(
            op_handle, &desc, out->desc(), prob->desc()));
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
// bias && !residual
// residual
// residual->data() == c->data()
// beta == 0.0
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
                // std::cout << "1";
                gemm(c, a, b, alpha, 1.0);
            } else {
                auto c_copy = Tensor::buffer(c->dtype(), c->shape(), memory_pool);
                c_copy->copyFrom(c, op_handle, stream);
                // std::cout << "2";
                gemm(c, a, b, alpha, beta);
                add(c, c, c_copy);
            }
        } else {
            // std::cout << "3";
            gemm(c, a, b, alpha, beta);
            add(c, c, residual);
        }
    } else {
        // std::cout << "4";
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

void InferenceContext::dequant(std::shared_ptr<Tensor> weight,
                               std::shared_ptr<Tensor> in_w,
                               std::shared_ptr<Tensor> in_s,
                               std::shared_ptr<Tensor> in_z) {

    size_t key = CacheManager::createDescriptorKey(weight, in_w, in_s, in_z);

    infiniopDequantizeAWQDescriptor_t desc;
    if (!cache_manager->getDequantizeAWQDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateDequantizeAWQDescriptor(op_handle, &desc, weight->desc(), in_w->desc(), in_s->desc(), in_z->desc()));
        cache_manager->putDequantizeAWQDescriptor(key, desc);
    }

    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetDequantizeAWQWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();

    RUN_INFINI(infiniopDequantizeAWQ(
        desc, workspace, workspace_size,
        weight->data(), in_w->data(), in_s->data(), in_z->data(), stream));
}

void InferenceContext::conv2d(std::shared_ptr<Tensor> y,
                             std::shared_ptr<Tensor> x,
                             std::shared_ptr<Tensor> w,
                             std::shared_ptr<Tensor> b,
                             std::vector<size_t> pads,
                             std::vector<size_t> strides,
                             std::vector<size_t> dilations) {
    // 步骤1: 创建缓存键 - 包含所有影响算子行为的参数
    size_t key = CacheManager::createDescriptorKey(y, x, w, b);

    // 将卷积参数也纳入缓存键计算
    void *b_data = b ? b->data() : nullptr;
    for (size_t pad : pads) {
        hash_combine(key, std::hash<int>()(pad));
    }
    for (size_t stride : strides) {
        hash_combine(key, std::hash<int>()(stride));
    }
    for (size_t dilation : dilations) {
        hash_combine(key, std::hash<int>()(dilation));
    }

    // 步骤2: 查找描述符缓存
    infiniopConvDescriptor_t desc;
    auto b_desc = b ? b->desc() : nullptr;
    if (!cache_manager->getConvDescriptor(key, desc)) {

        // std::cout << "X DESC = " << x->info() << std::endl;
        // std::cout << "W DESC = " << w->info() << std::endl;
        // std::cout << "Y DESC = " << y->info() << std::endl;
        // std::cout << "pads: " << pads[0] << ", " << pads[1] << "\n";
        // std::cout << "strides: " << strides[0] << ", " << strides[1] << "\n";
        // std::cout << "dilations: " << dilations[0] << ", " << dilations[1] << "\n";
        // 步骤3: 创建新描述符并缓存
        RUN_INFINI(infiniopCreateConvDescriptor(
            op_handle, &desc, y->desc(), x->desc(), w->desc(), b_desc,
            pads.data(), strides.data(), dilations.data(), pads.size()));
        cache_manager->putConvDescriptor(key, desc);
    }
    // 步骤4: 获取工作空间大小
    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetConvWorkspaceSize(desc, &workspace_size));

    // 步骤5: 确保工作空间足够
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();

    // 步骤6: 执行卷积算子
    RUN_INFINI(infiniopConv(
        desc, workspace, workspace_size,
        y->data(), x->data(), w->data(), b_data, stream));
}

void InferenceContext::quickGelu(std::shared_ptr<Tensor> y,
                                 std::shared_ptr<Tensor> x) {
    // 步骤1: 创建缓存键 - QuickGelu只依赖输入输出张量
    size_t key = CacheManager::createDescriptorKey(y, x);

    // 步骤2: 尝试从缓存中获取描述符
    infiniopQuickGeluDescriptor_t desc;
    if (!cache_manager->getQuickGeluDescriptor(key, desc)) {
        // 步骤3: 创建新的描述符
        RUN_INFINI(infiniopCreateQuickGeluDescriptor(
            op_handle, &desc, y->desc(), x->desc()));

        // 缓存描述符以便复用
        cache_manager->putQuickGeluDescriptor(key, desc);
    }

    // 步骤4: 获取工作空间大小
    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetQuickGeluWorkspaceSize(desc, &workspace_size));

    // 步骤5: 确保工作空间充足
    ensure_workspace(workspace_size);
    void* workspace = workspace_storage->memory();

    // 步骤6: 执行 QuickGelu 算子
    RUN_INFINI(infiniopQuickGelu(
        desc, workspace, workspace_size,
        y->data(), x->data(), stream));
}


void InferenceContext::softmax(std::shared_ptr<Tensor> y,
                                std::shared_ptr<Tensor> x,
                                int axis) {
    // 步骤1: 创建缓存键 - 包含影响算子行为的参数（y, x, axis）
    size_t key = CacheManager::createDescriptorKey(y, x);
    hash_combine(key, std::hash<int>()(axis)); // 将 axis 也纳入 key

    // 步骤2: 查找 Softmax 描述符缓存
    infiniopSoftmaxDescriptor_t desc;
    if (!cache_manager->getSoftmaxDescriptor(key, desc)) {
        // 步骤3: 创建新描述符
        RUN_INFINI(infiniopCreateSoftmaxDescriptor(
            op_handle, &desc, y->desc(), x->desc(), axis));
        // 可以选择缓存
        // cache_manager->putSoftmaxDescriptor(key, desc);
    }

    // 步骤4: 获取工作空间大小
    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetSoftmaxWorkspaceSize(desc, &workspace_size));

    // 步骤5: 确保工作空间充足
    ensure_workspace(workspace_size);
    void* workspace = workspace_storage->memory();

    // 步骤6: 执行 Softmax 算子
    RUN_INFINI(infiniopSoftmax(
        desc, workspace, workspace_size,
        y->data(), x->data(), stream));
}

void InferenceContext::sigmoid(std::shared_ptr<Tensor> y,
                                std::shared_ptr<Tensor> x) {
    // 步骤1: 创建缓存键（y 和 x 决定算子行为）
    size_t key = CacheManager::createDescriptorKey(y, x);

    // 步骤2: 尝试从缓存获取描述符
    infiniopSigmoidDescriptor_t desc;
    if (!cache_manager->getSigmoidDescriptor(key, desc)) {
        // 步骤3: 创建新的描述符
        RUN_INFINI(infiniopCreateSigmoidDescriptor(
            op_handle, &desc, y->desc(), x->desc()));

        // 缓存以供复用
        cache_manager->putSigmoidDescriptor(key, desc);
    }

    // 步骤4: 获取工作空间大小
    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetSigmoidWorkspaceSize(desc, &workspace_size));

    // 步骤5: 确保工作空间充足
    ensure_workspace(workspace_size);
    void* workspace = workspace_storage->memory();

    // 步骤6: 执行 Sigmoid 算子
    RUN_INFINI(infiniopSigmoid(
        desc, workspace, workspace_size,
        y->data(), x->data(), stream));
}

void InferenceContext::gelu(std::shared_ptr<Tensor> output,
                             std::shared_ptr<Tensor> input) {
    // 构造 descriptor key（只需要用 output 和 input 参与 key）
    size_t key = CacheManager::createDescriptorKey(output, input);

    infiniopGeluDescriptor_t desc;
    if (!cache_manager->getGeluDescriptor(key, desc)) {
        // 创建 GELU descriptor
        RUN_INFINI(infiniopCreateGeluDescriptor(
            op_handle,
            &desc,
            output->desc(),   // output_desc
            input->desc()     // input_desc
        ));
        cache_manager->putGeluDescriptor(key, desc);
    }

    // 获取 workspace 大小并确保 workspace 足够
    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetGeluWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();

    // 调用 GELU kernel
    RUN_INFINI(infiniopGelu(
        desc,
        workspace,
        workspace_size,
        output->data(),
        input->data(),
        stream));
}
