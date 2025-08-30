#include "qwen_moe_impl.hpp"
#include "qwen_moe_weight.hpp" // 注意: 此文件需要您后续根据 qwen_weight.hpp 进行创建

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../inference_context.hpp"
#include "infinicore_infer.h" // 应该指向包含了 qwen_moe.h 的主头文件

#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <numeric>      // 需要包含这个头文件
#include <functional>   // 需要包含这个头文件

// 为单个设备创建和加载资源
void createDeviceResourceMoe(DeviceResourceMoe *rsrc, const QwenMoeMeta *meta,
                             const QwenMoeWeights *weights,
                             infiniDevice_t device, int idev,
                             int ndev, int dev_id,
                             infinicclComm_t comm) {
    RUN_INFINI(infinirtSetDevice(device, dev_id));
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle);
    infinirtStream_t stream;
    infinirtStreamCreate(&stream);

    std::vector<std::shared_ptr<Tensor>> w_attn_norm, w_attn_qkv, b_attn_qkv, 
                                         w_attn_q_norm, w_attn_k_norm, w_attn_out; // <-- 已更新
    for (size_t layer = 0; layer < meta->nlayer; layer++) {
        w_attn_norm.push_back(qwen_moe::getAttnNorm(meta, weights, layer));
        w_attn_qkv.push_back(qwen_moe::getAttnQKV(meta, weights, layer, idev, ndev));
        if (weights->attn_qkv_b != nullptr) {
            b_attn_qkv.push_back(qwen_moe::getAttnQKVBias(meta, weights, layer, idev, ndev));
        }
        // --- 已添加加载 QK Norm 的逻辑 ---
        if (weights->attn_q_norm != nullptr) {
            w_attn_q_norm.push_back(qwen_moe::getAttnQNorm(meta, weights, layer));
            w_attn_k_norm.push_back(qwen_moe::getAttnKNorm(meta, weights, layer));
        }
        // ---------------------------------
        w_attn_out.push_back(qwen_moe::getAttnO(meta, weights, layer, idev, ndev));
    }

    // ... (MoE weight loading remains the same) ...
    std::vector<std::shared_ptr<Tensor>> w_ffn_norm, w_moe_gate;
    std::vector<std::shared_ptr<Tensor>> w_moe_experts_gate_up, w_moe_experts_down;
    for (size_t layer = 0; layer < meta->nlayer; layer++) {
        w_ffn_norm.push_back(qwen_moe::getFFNNorm(meta, weights, layer));
        w_moe_gate.push_back(qwen_moe::getMoeGate(meta, weights, layer, idev, ndev));
        for (size_t expert = 0; expert < meta->num_experts; expert++) {
            w_moe_experts_gate_up.push_back(qwen_moe::getMoeExpertGateUp(meta, weights, layer, expert, idev, ndev));
            w_moe_experts_down.push_back(qwen_moe::getMoeExpertDown(meta, weights, layer, expert, idev, ndev));
        }
    }

    auto memory_pool = std::make_shared<MemoryPool>(128 * 1024 * 1024);

    rsrc->device = device;
    rsrc->device_id = dev_id;
    rsrc->handle = handle;
    rsrc->w_in_embd = qwen_moe::getInEmbd(meta, weights);
    rsrc->w_out_norm = qwen_moe::getOutNorm(meta, weights);
    rsrc->w_out_embd = qwen_moe::getOutEmbd(meta, weights);
    rsrc->sin_table = qwen_moe::getSinTable(meta);
    rsrc->cos_table = qwen_moe::getCosTable(meta);
    rsrc->w_attn_norm = std::move(w_attn_norm);
    rsrc->w_attn_qkv = std::move(w_attn_qkv);
    rsrc->b_attn_qkv = std::move(b_attn_qkv);
    rsrc->w_attn_q_norm = std::move(w_attn_q_norm); // <-- 已添加
    rsrc->w_attn_k_norm = std::move(w_attn_k_norm); // <-- 已添加
    rsrc->w_attn_out = std::move(w_attn_out);
    rsrc->w_ffn_norm = std::move(w_ffn_norm);
    rsrc->w_moe_gate = std::move(w_moe_gate);
    rsrc->w_moe_experts_gate_up = std::move(w_moe_experts_gate_up);
    rsrc->w_moe_experts_down = std::move(w_moe_experts_down);
    rsrc->stream = stream;
    rsrc->comm = comm;
    rsrc->memory_pool = memory_pool;

    RUN_INFINI(infinirtDeviceSynchronize());
}

// 释放单个设备的资源
void releaseDeviceResourceMoe(DeviceResourceMoe &rsrc) {
    rsrc.w_in_embd.reset();
    rsrc.w_out_norm.reset();
    rsrc.w_out_embd.reset();
    rsrc.sin_table.reset();
    rsrc.cos_table.reset();
    rsrc.w_attn_norm.clear();
    rsrc.w_attn_qkv.clear();
    rsrc.b_attn_qkv.clear();
    rsrc.w_attn_q_norm.clear();
    rsrc.w_attn_k_norm.clear();
    rsrc.w_attn_out.clear();
    rsrc.w_ffn_norm.clear();
    rsrc.w_moe_gate.clear();
    rsrc.w_moe_experts_gate_up.clear();
    rsrc.w_moe_experts_down.clear();
    RUN_INFINI(infinirtStreamDestroy(rsrc.stream));
    RUN_INFINI(infiniopDestroyHandle(rsrc.handle));
}


void inferDeviceBatchMoe(const QwenMoeMeta &meta, DeviceResourceMoe &rsrc,
                         int idev, int ndev,
                         const uint32_t *tokens, uint32_t ntok,
                         const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                         struct KVCache **kv_caches,
                         const float *temperature, const uint32_t *topk, const float *topp,
                         uint32_t *output, void *last_logits) {
    // --- MoE --- 获取 MoE 特定参数
    auto nlayer = meta.nlayer;
    auto nkvh = meta.nkvh / ndev;
    auto nh = meta.nh / ndev;
    auto ngroup = nh / nkvh;
    auto dh = meta.dh;
    auto d = meta.d;
    auto dt_logits = meta.dt_logits;
    auto dvoc = meta.dvoc;
    auto stream = rsrc.stream;
    bool has_qkv_bias = !rsrc.b_attn_qkv.empty() && rsrc.b_attn_qkv[0] != nullptr;
    bool has_qk_norm = !rsrc.w_attn_q_norm.empty() && rsrc.w_attn_q_norm[0] != nullptr;
    auto num_experts = meta.num_experts;
    auto num_experts_per_tok = meta.num_experts_per_tok;
    auto moe_di = meta.moe_intermediate_size / ndev;

    // --- MoE --- 分配缓冲区 (为 MoE 更新)
    auto logits_in = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto logits_out = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto qkv_buf = Tensor::buffer(dt_logits, {ntok, (nh + nkvh * 2) * dh}, rsrc.memory_pool);
    auto o_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, rsrc.memory_pool);
    
    // --- MoE --- MoE 模块的缓冲区
    auto moe_gate_logits = Tensor::buffer(dt_logits, {ntok, num_experts}, rsrc.memory_pool);
    auto moe_gate_probs = Tensor::buffer(dt_logits, {ntok, num_experts}, rsrc.memory_pool);
    auto topk_weights = Tensor::buffer(dt_logits, {ntok, num_experts_per_tok}, rsrc.memory_pool);
    auto topk_indices = Tensor::buffer(INFINI_DTYPE_I32, {ntok, num_experts_per_tok}, rsrc.memory_pool);
    auto expert_outputs = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto gate_up_buf = Tensor::buffer(dt_logits, {ntok, 2 * moe_di}, rsrc.memory_pool);
    auto gate_buf = gate_up_buf->slice(1, 0, moe_di);
    auto up_buf = gate_up_buf->slice(1, moe_di, moe_di);

    // --- MoE --- 最终采样缓冲区 (与密集模型相同)
    auto prob_buf = Tensor::buffer(dt_logits, {nreq, dvoc}, rsrc.memory_pool);
    auto result_buf = Tensor::buffer(INFINI_DTYPE_I64, {nreq}, rsrc.memory_pool);
    auto result_cpu = std::vector<int64_t>(nreq);

    auto qkv_buf_view = qkv_buf->view({ntok, nh + nkvh * 2, dh});
    auto q_buf = qkv_buf_view->slice(1, 0, nh);
    auto k_buf = qkv_buf_view->slice(1, nh, nkvh);

    size_t max_qk_size = 0;
    size_t max_seq_len = 0;

    for (uint32_t req = 0; req < nreq; req++) {
        auto past_len = req_pos[req];
        auto seq_len = req_lens[req];
        auto total_len = past_len + seq_len;

        max_qk_size = std::max(max_qk_size, size_t(seq_len * total_len));
        max_seq_len = std::max(max_seq_len, size_t(seq_len));
    }

    auto qk_buf = Tensor::buffer(dt_logits, {nh, max_qk_size}, rsrc.memory_pool);
    auto rearrange_q_buf = Tensor::buffer(dt_logits, {nkvh, ngroup * max_seq_len, dh}, rsrc.memory_pool);
    auto q_rearrange = rearrange_q_buf->view({nkvh, ngroup, max_seq_len, dh});
    auto attn_val_buf = Tensor::buffer(dt_logits, {nkvh, ngroup * max_seq_len, dh}, rsrc.memory_pool);
    auto attn_val_gemm = attn_val_buf->view({nkvh, ngroup, max_seq_len, dh});


    // --- 输入准备 (与密集模型相同) ---
    auto batch_pos_ids = std::vector<uint32_t>(ntok);
    size_t req_start = 0;
    for (uint32_t req = 0; req < nreq; req++) {
        for (uint32_t i = 0; i < req_lens[req]; i++) {
            batch_pos_ids[req_start + i] = req_pos[req] + i;
        }
        req_start += req_lens[req];
    }

    std::shared_ptr<Tensor> pos_ids_buf;
    if (rsrc.device == INFINI_DEVICE_CPU) {
        pos_ids_buf = Tensor::weight(batch_pos_ids.data(), INFINI_DTYPE_U32, {ntok});
    } else {
        pos_ids_buf = Tensor::buffer(INFINI_DTYPE_U32, {ntok}, rsrc.memory_pool);
        RUN_INFINI(infinirtMemcpyAsync(pos_ids_buf->data(), batch_pos_ids.data(), sizeof(uint32_t) * ntok,
                                       INFINIRT_MEMCPY_H2D, stream));
    }
    for (uint32_t i = 0; i < ntok; i++) {
        RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d),
                                       rsrc.w_in_embd->data(tokens[i] * d),
                                       dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
    }

    // --- 主要计算循环 ---
    for (uint32_t layer = 0; layer < nlayer; layer++) {
        // 1. 注意力模块 (此模块与密集模型完全相同)
        rmsnorm(logits_out, logits_in, rsrc.w_attn_norm[layer], meta.epsilon);
        linear(qkv_buf, logits_out, rsrc.w_attn_qkv[layer], 1.0, 0.0, nullptr, has_qkv_bias ? rsrc.b_attn_qkv[layer] : nullptr);
        if (has_qk_norm) {
            rmsnorm(q_buf, q_buf, rsrc.w_attn_q_norm[layer], meta.epsilon);
            rmsnorm(k_buf, k_buf, rsrc.w_attn_k_norm[layer], meta.epsilon);
        }
        rope(q_buf, q_buf, pos_ids_buf, rsrc.sin_table, rsrc.cos_table);
        rope(k_buf, k_buf, pos_ids_buf, rsrc.sin_table, rsrc.cos_table);
        
        linear(logits_in, o_buf, rsrc.w_attn_out[layer], 1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr);
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(logits_in->data(), logits_in->data(), ntok * d, dt_logits, INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }
        
        // --- MoE --- 2. MoE 模块 (替换 FFN 模块)
        rmsnorm(logits_out, logits_in, rsrc.w_ffn_norm[layer], meta.epsilon);

        // a. 门控: 为每个 token 计算专家得分
        linear(moe_gate_logits, logits_out, rsrc.w_moe_gate[layer], 1.0, 0.0, nullptr, nullptr);
        
        // b. 路由: 应用 softmax 并找到 top-k 专家
        causalSoftmax(moe_gate_probs, moe_gate_logits);
        topk_fun(topk_weights, topk_indices, moe_gate_probs, num_experts_per_tok);

        // Qwen特有: 权重归一化
        if (meta.norm_topk_prob) {
            normalize(topk_weights, topk_weights, 1, 1e-6);
        }

        // c. 专家计算: 采用更高效的“专家并行”模式
        zeros(expert_outputs); // 清空最终输出缓冲区

        // 将路由结果复制到 CPU，以构建分发计划
        std::vector<int32_t> topk_indices_cpu(ntok * num_experts_per_tok);
        std::vector<float> topk_weights_cpu(ntok * num_experts_per_tok);
        // 计算 topk_indices 的总元素数量
        size_t topk_indices_nelem = std::accumulate(topk_indices->shape().begin(), topk_indices->shape().end(), 1ULL, std::multiplies<size_t>());
        // 计算 topk_weights 的总元素数量
        size_t topk_weights_nelem = std::accumulate(topk_weights->shape().begin(), topk_weights->shape().end(), 1ULL, std::multiplies<size_t>());

        // 使用 "元素数量 * 单个元素大小" 的方式计算总字节数
        RUN_INFINI(infinirtMemcpy(topk_indices_cpu.data(), topk_indices->data(), topk_indices_nelem * dsize(topk_indices->dtype()), INFINIRT_MEMCPY_D2H));
        RUN_INFINI(infinirtMemcpy(topk_weights_cpu.data(), topk_weights->data(), topk_weights_nelem * dsize(topk_weights->dtype()), INFINIRT_MEMCPY_D2H));
        RUN_INFINI(infinirtStreamSynchronize(stream));

        for (uint32_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
            std::vector<uint32_t> token_indices_for_expert;
            std::vector<float> weights_for_expert;
            
            // 在 CPU 上构建分发列表
            for (uint32_t token_i = 0; token_i < ntok; ++token_i) {
                for (uint32_t k = 0; k < num_experts_per_tok; ++k) {
                    if (static_cast<uint32_t>(topk_indices_cpu[token_i * num_experts_per_tok + k]) == expert_idx) {
                        token_indices_for_expert.push_back(token_i);
                        weights_for_expert.push_back(topk_weights_cpu[token_i * num_experts_per_tok + k]);
                    }
                }
            }

            if (token_indices_for_expert.empty()) {
                continue; // 没有 token 分配给这个专家，跳过
            }

            size_t num_tokens_for_expert = token_indices_for_expert.size();
            size_t weight_idx = layer * num_experts + expert_idx;

            // Gather: 收集分配给该专家的所有 token 的 hidden states
            auto expert_input_states = Tensor::buffer(dt_logits, {num_tokens_for_expert, d}, rsrc.memory_pool);
            gather(expert_input_states, logits_out, token_indices_for_expert);

            // Compute: 对这个小批量进行专家 FFN 计算
            auto expert_gate_up = Tensor::buffer(dt_logits, {num_tokens_for_expert, 2 * moe_di}, rsrc.memory_pool);
            auto expert_gate = expert_gate_up->slice(1, 0, moe_di);
            auto expert_up = expert_gate_up->slice(1, moe_di, moe_di);
            
            linear(expert_gate_up, expert_input_states, rsrc.w_moe_experts_gate_up[weight_idx], 1.0, 0.0, nullptr, nullptr);
            swiglu(expert_gate, expert_up, expert_gate);
            
            auto single_expert_output = Tensor::buffer(dt_logits, {num_tokens_for_expert, d}, rsrc.memory_pool);
            linear(single_expert_output, expert_gate, rsrc.w_moe_experts_down[weight_idx], 1.0, 0.0, nullptr, nullptr);

            // Weighting: 将专家输出乘以其路由权重
            // scale(single_expert_output, single_expert_output, weights_for_expert);

            // Scatter_add: 将加权后的结果加回到最终输出缓冲区的原始位置
            scatter_add(expert_outputs, single_expert_output, token_indices_for_expert);
        }   
        
        // 添加残差连接
        add(logits_in, logits_in, expert_outputs);

        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(logits_in->data(), logits_in->data(), ntok * d, dt_logits, INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }
    }

    // --- 最终采样和输出 (此模块与密集模型完全相同) ---
    if (idev == 0) {
        if (last_logits != nullptr) {
            rmsnorm(logits_out, logits_in, rsrc.w_out_norm, meta.epsilon);
            auto last_logits_buf = Tensor::buffer(dt_logits, {ntok, dvoc}, rsrc.memory_pool);
            linear(last_logits_buf, logits_out, rsrc.w_out_embd, 1.0, 0.0, nullptr, nullptr);
            RUN_INFINI(infinirtStreamSynchronize(stream));
            RUN_INFINI(infinirtMemcpy(last_logits, last_logits_buf->data(), dsize(dt_logits) * ntok * dvoc, INFINIRT_MEMCPY_D2H));
        }
        if (output != nullptr) {
            size_t token_offset = 0;
            for (uint32_t req = 0; req < nreq; req++) {
                auto seq_len = req_lens[req];
                token_offset += seq_len;
                rmsnorm(logits_out->slice(0, req, 1),
                        logits_in->slice(0, token_offset - 1, 1),
                        rsrc.w_out_norm,
                        meta.epsilon);
            }
            linear(prob_buf, logits_out->slice(0, 0, nreq), rsrc.w_out_embd, 1.0, 0.0, nullptr, nullptr);
            std::random_device _rd;
            std::mt19937 gen(_rd());
            token_offset = 0;
            for (uint32_t req = 0; req < nreq; req++) {
                auto seq_len = req_lens[req];
                float random_val = std::uniform_real_distribution<float>(0, 1)(gen);
                randomSample(result_buf->slice(0, req, 1)->view_as({}, {}),
                             prob_buf->slice(0, req, 1)->view_as({dvoc}, {1}),
                             random_val, topp[req], topk[req], temperature[req]);
                token_offset += seq_len;
            }
            RUN_INFINI(infinirtStreamSynchronize(stream));
            RUN_INFINI(infinirtMemcpy(result_cpu.data(), result_buf->data(),
                                      sizeof(int64_t) * nreq, INFINIRT_MEMCPY_D2H));
            for (uint32_t req = 0; req < nreq; req++) {
                output[req] = uint32_t(result_cpu[req]);
            }
        }
    }
}




// 每个设备的 worker 线程函数
void launchDeviceMoe(const QwenMoeMeta &meta, const QwenMoeWeights *weights, DeviceResourceMoe *rsrc, InferState &state, InferRequest &req,
                     infiniDevice_t device, int idev, int ndev, int dev_id, infinicclComm_t comm) {
    
    CacheManager cache_manager(100);
    InferenceContext ctx(nullptr, &cache_manager, rsrc->stream);
    setInferenceContext(&ctx);
    
    createDeviceResourceMoe(rsrc, &meta, weights, device, idev, ndev, dev_id, comm);
    {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.loaded = true;
        lock.unlock();
        state.cv_load.notify_one();
    }

    while (true) {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.cv_start.wait(lock, [&] { return state.proceed || state.exit_flag; });
        if (state.exit_flag) break;

        inferDeviceBatchMoe(meta, *rsrc, idev, ndev, req.tokens, req.ntok,
                            req.req_lens, req.nreq, req.req_pos, req.kv_caches,
                            req.temperature, req.topk, req.topp, req.output, req.logits);

        state.proceed = false;
        lock.unlock();
        state.cv_done.notify_one();
    }

    releaseDeviceResourceMoe(*rsrc);
    setInferenceContext(nullptr);
}

// 主模型类的构造函数
QwenMoeModel::QwenMoeModel(const QwenMoeMeta *_meta, const QwenMoeWeights *weights, infiniDevice_t device_, std::vector<int> device_ids) : meta(*_meta) {
    int ndev = int(device_ids.size());
    device = device_;
    dev_ids = device_ids;
    dev_resources = std::vector<DeviceResourceMoe>(ndev);
    states = std::vector<InferState>(ndev);
    threads.resize(ndev);
    RUN_INFINI(infinirtInit());
    auto comms = std::vector<infinicclComm_t>(ndev, nullptr);
    if (ndev > 1) {
        RUN_INFINI(infinicclCommInitAll(device, comms.data(), ndev, dev_ids.data()));
    }

    for (int i = 0; i < ndev; i++) {
        threads[i] = std::thread(launchDeviceMoe, std::cref(meta), weights, &dev_resources[i], std::ref(states[i]), std::ref(req), device, i, ndev, dev_ids[i], comms[i]);
    }
    for (int i = 0; i < ndev; i++) {
        std::unique_lock<std::mutex> lock(states[i].mtx);
        states[i].cv_load.wait(lock, [&] { return states[i].loaded; });
        lock.unlock();
    }
}

// ===================================================================
// 公共 C API 实现
// ===================================================================

extern "C" {

__C __export struct QwenMoeModel *
createQwenMoeModel(const QwenMoeMeta *meta,
                   const QwenMoeWeights *weights,
                   infiniDevice_t device,
                   int ndev,
                   const int *dev_ids) {
    std::cout << "C++: createQwenMoeModel called." << std::endl;
    std::vector<int> device_ids(ndev);
    std::copy(dev_ids, dev_ids + ndev, device_ids.begin());
    QwenMoeModel *model = new QwenMoeModel(meta, weights, device, device_ids);
    return model;
}

__C __export void
destroyQwenMoeModel(struct QwenMoeModel *model) {
    std::cout << "C++: destroyQwenMoeModel called." << std::endl;
    auto ndev = model->dev_resources.size();
    for (size_t idev = 0; idev < ndev; idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].exit_flag = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }
    for (size_t idev = 0; idev < ndev; idev++) {
        model->threads[idev].join();
    }
    delete model;
}

// __C __export struct KVCache *
// createQwenMoeKVCache(const struct QwenMoeModel * model) {
//     // TODO: 实现 KVCache 的创建逻辑
//     return nullptr;
// }

// __C __export void
// dropQwenMoeKVCache(const struct QwenMoeModel * model, struct KVCache * cache) {
//     // TODO: 实现 KVCache 的销毁逻辑
// }

__C __export void
inferQwenMoeBatch(struct QwenMoeModel *model,
                  const uint32_t *tokens, uint32_t ntok,
                  const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                  struct KVCache **kv_caches,
                  const float *temperature, const uint32_t *topk, const float *topp,
                  uint32_t *output) {
    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.kv_caches = kv_caches;
    model->req.output = output;
    model->req.logits = nullptr;
    model->req.temperature = temperature;
    model->req.topk = topk;
    model->req.topp = topp;

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

__C __export void
forwardQwenMoeBatch(struct QwenMoeModel *model,
                    const uint32_t *tokens, uint32_t ntok,
                    const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                    struct KVCache **kv_caches,
                    void *logits) {
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


} // extern "C"
