#include "../../../tensor.hpp"
#include "../../../utils.hpp"
#include "../../inference_context.hpp"
#include "../qwen_device_resource.hpp"
#include "../qwen_model.hpp"
#include "infinicore_infer.h"
#include <random>
#include <thread>
#include <vector>
    
void Qwen3MoEinferDeviceBatch(const Qwen3MoE::Meta *meta, DeviceResource<Qwen3MoE::WeightsTensor> &rsrc,
                              uint32_t idev, uint32_t ndev,
                              const uint32_t *tokens, uint32_t ntok,
                              const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                              struct KVCache **kv_caches,
                              const float *temperature, const uint32_t *topk, const float *topp,
                              uint32_t *output, void *last_logits) {

    // ========================================================================
    // 1. 提取模型配置参数
    // ========================================================================
    auto nlayer = meta->nlayer;              // Transformer层数
    auto nkvh = meta->nkvh / ndev;           // 每个设备的KV头数（分布式时分割）
    auto nh = meta->nh / ndev;                // 每个设备的注意力头数（分布式时分割）
    auto ngroup = nh / nkvh;                  // GQA分组数（Grouped Query Attention）
    auto dh = meta->dh;                       // 每个注意力头的维度
    auto d = meta->d;                         // 模型隐藏层维度
    auto dt_logits = meta->dt_logits;         // logits的数据类型
    auto dvoc = meta->dvoc;                   // 词汇表大小
    auto stream = rsrc.stream;               // CUDA流，用于异步操作

    // ========================================================================
    // 2. 分配主要计算缓冲区
    // ========================================================================
    // 主计算缓冲区：用于存储每层的输入输出
    auto logits_in = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);      // 层输入 [ntok, d]
    auto logits_out = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);    // 层输出 [ntok, d]
    
    // QKV缓冲区：存储query、key、value投影结果
    // 形状: [ntok, (nh + nkvh * 2) * dh]
    // nh个query头 + nkvh个key头 + nkvh个value头
    auto qkv_buf = Tensor::buffer(dt_logits, {ntok, (nh + nkvh * 2) * dh}, rsrc.memory_pool);
    auto qkv_rope = qkv_buf->view({ntok, nh + nkvh * 2, dh});  // 用于RoPE的视图

    // 注意力输出缓冲区
    auto o_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, rsrc.memory_pool);   // 注意力输出 [ntok, nh*dh]
    
    // 采样相关缓冲区
    auto prob_buf = Tensor::buffer(dt_logits, {nreq, dvoc}, rsrc.memory_pool);    // 输出概率分布 [nreq, dvoc]
    auto result_buf = Tensor::buffer(INFINI_DTYPE_I64, {nreq}, rsrc.memory_pool); // 采样结果 [nreq]
    auto result_cpu = std::vector<int64_t>(nreq);                                 // CPU端的结果缓冲区

    // Prepare inputs
    auto batch_pos_ids = std::vector<uint32_t>(ntok);
    size_t req_start = 0;
    for (uint32_t req = 0; req < nreq; req++) {
        for (uint32_t i = 0; i < req_lens[req]; i++) {
            batch_pos_ids[req_start + i] = req_pos[req] + i;
        }
        req_start += req_lens[req];
    }

    // 获取权重张量指针
    const Qwen3MoE::WeightsTensor *g_WeightsTensor = rsrc.weights_tensor_ptr.get();
    if (!g_WeightsTensor) {
        return;  // 权重未加载，直接返回
    }

    std::shared_ptr<Tensor> pos_ids_buf;
    if (rsrc.device == INFINI_DEVICE_CPU) {
        pos_ids_buf = Tensor::weight(batch_pos_ids.data(), INFINI_DTYPE_U32, {ntok});
    } else {
        pos_ids_buf = Tensor::buffer(INFINI_DTYPE_U32, {ntok}, rsrc.memory_pool);
        RUN_INFINI(infinirtMemcpyAsync(pos_ids_buf->data(), batch_pos_ids.data(), sizeof(uint32_t) * ntok,
                                       INFINIRT_MEMCPY_H2D, stream));
    }
    
    // 将输入token嵌入到隐藏空间：从词汇表嵌入矩阵中查找每个token的嵌入向量
    for (uint32_t i = 0; i < ntok; i++) {
        RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d), g_WeightsTensor->w_in_embd->data(tokens[i] * d),
                                       dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
    }

    // Attention
    // attention inner
    size_t max_qk_size = 0;
    size_t max_seq_len = 0;

    for (uint32_t req = 0; req < nreq; req++) {
        auto past_len = req_pos[req];      // 历史长度（已缓存的token数）
        auto seq_len = req_lens[req];      // 当前请求的新token数
        auto total_len = past_len + seq_len; // 总长度（历史 + 当前）

        max_qk_size = std::max(max_qk_size, size_t(seq_len * total_len));
        max_seq_len = std::max(max_seq_len, size_t(seq_len));
    }

    auto qk_buf = Tensor::buffer(dt_logits, {nh, max_qk_size}, rsrc.memory_pool);
    auto rearrange_q_buf = Tensor::buffer(dt_logits, {nkvh, ngroup * max_seq_len, dh}, rsrc.memory_pool);
    auto q_rearrange = rearrange_q_buf->view({nkvh, ngroup, max_seq_len, dh});
    auto attn_val_buf = Tensor::buffer(dt_logits, {nkvh, ngroup * max_seq_len, dh}, rsrc.memory_pool);
    auto attn_val_gemm = attn_val_buf->view({nkvh, ngroup, max_seq_len, dh});

    // Compute
    for (uint32_t ilayer = 0; ilayer < nlayer; ilayer++) {
        auto layer_tensor = g_WeightsTensor->layers[ilayer];

        // 1. Attention
        // rms norm
        rmsnorm(logits_out, logits_in, layer_tensor->w_attn_norm, meta->epsilon);
        // qkv_proj
        linear(qkv_buf, logits_out, layer_tensor->self_attn->w_attn_qkv, 1.0, 0.0, nullptr, layer_tensor->self_attn->b_attn_qkv ? layer_tensor->self_attn->b_attn_qkv : nullptr);

        if (layer_tensor->self_attn->w_attn_qk_norm) {
            auto qkv_buf_view = qkv_buf->view({ntok, nh + nkvh * 2, dh});
            auto q_buf = qkv_buf_view->slice(1, 0, nh);
            auto k_buf = qkv_buf_view->slice(1, nh, nkvh);
            rmsnorm(q_buf, q_buf, layer_tensor->self_attn->w_attn_qk_norm->slice(0, 0, dh), meta->epsilon);
            rmsnorm(k_buf, k_buf, layer_tensor->self_attn->w_attn_qk_norm->slice(0, dh, dh), meta->epsilon);
        }

        // rope
        rope(qkv_rope->slice(1, 0, nh), qkv_rope->slice(1, 0, nh), pos_ids_buf, g_WeightsTensor->sin_table, g_WeightsTensor->cos_table);
        rope(qkv_rope->slice(1, nh, nkvh), qkv_rope->slice(1, nh, nkvh), pos_ids_buf, g_WeightsTensor->sin_table, g_WeightsTensor->cos_table);

        size_t token_offset = 0;
        for (uint32_t req = 0; req < nreq; req++) {
            auto past_len = req_pos[req];      // 该请求的历史长度
            auto seq_len = req_lens[req];      // 该请求的新token数
            auto total_len = past_len + seq_len; // 总长度
            
            // 提取当前请求的Q、K、V
            // Q: [seq_len, nh, dh] -> 重排为 [nkvh, ngroup, seq_len, dh] 用于GQA
            auto o = o_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});
            auto q = qkv_rope->slice({{0, token_offset, seq_len}, {1, 0, nh}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});
            auto k = qkv_rope->slice({{0, token_offset, seq_len}, {1, nh, nkvh}});
            auto v = qkv_rope->slice({{0, token_offset, seq_len}, {1, nh + nkvh, nkvh}});

            // self attention
            // concat
            rearrange(kv_caches[req]->k[idev][ilayer]->slice(0, past_len, seq_len), k);
            rearrange(kv_caches[req]->v[idev][ilayer]->slice(0, past_len, seq_len), v);
            // qk
            rearrange(q_rearrange->slice(2, 0, seq_len), q);
            auto qk_gemm = qk_buf->slice(1, 0, seq_len * total_len)->view({nkvh, ngroup * seq_len, total_len});
            auto k_gemm = kv_caches[req]->k[idev][ilayer]->slice(0, 0, total_len)->permute({1, 2, 0});
            linear(qk_gemm, rearrange_q_buf->slice(1, 0, ngroup * seq_len), k_gemm, 1.f / float(sqrt(dh)), 0.f, nullptr, nullptr);
            // softmax
            auto qk_softmax = qk_buf->slice(1, 0, seq_len * total_len)->view({nh, seq_len, total_len});
            causalSoftmax(qk_softmax, qk_softmax);
            auto v_gemm = kv_caches[req]->v[idev][ilayer]->slice(0, 0, total_len)->permute({1, 0, 2});
            linear(attn_val_buf->slice(1, 0, ngroup * seq_len), qk_gemm, v_gemm, 1.f, 0.f, nullptr, nullptr);
            // rearrange attn val
            rearrange(o, attn_val_gemm->slice(2, 0, seq_len));

            token_offset += seq_len;
        }

        // o_proj
        linear(logits_in, o_buf, layer_tensor->self_attn->w_attn_out, 1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr); // only rank 0 adds residual

        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }

        // 2. FFN
        rmsnorm(logits_out, logits_in, layer_tensor->w_ffn_norm, meta->epsilon);

        // ------------------------------------------------------------------------
        // SparseMLP: 稀疏混合专家网络
        // 每个token根据路由权重选择top-k个专家进行计算
        // ------------------------------------------------------------------------
        {
            std::shared_ptr<Tensor> hidden_states = logits_out;  // MoE的输入

            // MoE配置参数
            size_t moe_intermediate_size = meta->_moe_intermediate_size / ndev;  // 每个设备的专家中间层大小

            // 分配MoE计算缓冲区
            // gate_up_buf: 存储gate和up投影的拼接结果 [1, 2 * moe_intermediate_size]
            auto router_gate_up_buf = Tensor::buffer(dt_logits, {1, 2 * moe_intermediate_size}, rsrc.memory_pool);
            auto router_gate_buf = router_gate_up_buf->slice(1, 0, moe_intermediate_size);  // gate部分
            auto router_up_buf = router_gate_up_buf->slice(1, moe_intermediate_size, moe_intermediate_size);  // up部分

            // 输出缓冲区：存储所有专家输出的加权和
            std::shared_ptr<Tensor> router_states_sum = Tensor::buffer(hidden_states->dtype(), hidden_states->shape(), rsrc.memory_pool);
            
            // 路由logits：每个token对所有专家的路由分数 [ntok, num_experts]
            std::shared_ptr<Tensor> router_logits = Tensor::buffer(dt_logits, {ntok, meta->_num_experts}, rsrc.memory_pool);

            // TopK路由参数
            size_t topk = meta->_num_experts_per_tok;        // 每个token选择的专家数量（通常为4或8）
            bool norm_topk_prob = meta->_norm_topk_prob;     // 是否对topk概率进行归一化

            // TopK路由结果缓冲区
            std::shared_ptr<Tensor> values_gpu = Tensor::buffer(infiniDtype_t::INFINI_DTYPE_F32, {ntok * topk}, rsrc.memory_pool);   // 专家权重 [ntok * topk]
            std::shared_ptr<Tensor> indices_gpu = Tensor::buffer(infiniDtype_t::INFINI_DTYPE_I32, {ntok * topk}, rsrc.memory_pool); // 专家索引 [ntok * topk]
            std::vector<float> values_cpu(ntok * topk, 0.f);   // CPU端权重（用于后续计算）
            std::vector<int> indices_cpu(ntok * topk, 0);      // CPU端索引（用于后续计算）
 
            // ------------------------------------------------------------------------
            // 开始MoE计算
            // ------------------------------------------------------------------------
            auto ffn = layer_tensor->ffn;

            // Step 1: 计算路由logits并执行TopK选择
            // 将hidden_states通过路由门控网络，得到每个token对所有专家的路由分数
            linear(router_logits, hidden_states, ffn->_gate_weight, 1.0, 0.0, nullptr, nullptr); 
            {
                topksoftmax(values_gpu, indices_gpu, router_logits, topk, norm_topk_prob);
                RUN_INFINI(infinirtMemcpy((void *)values_cpu.data(), values_gpu->data(), values_cpu.size() * sizeof(float), INFINIRT_MEMCPY_D2H));
                RUN_INFINI(infinirtMemcpy((void *)indices_cpu.data(), indices_gpu->data(), indices_cpu.size() * sizeof(int), INFINIRT_MEMCPY_D2H));
                RUN_INFINI(infinirtStreamSynchronize(rsrc.stream));
            }

            // Step 2: 对每个token执行MoE计算
            // 每个token根据路由结果，依次经过topk个专家，并将结果加权求和
            {
                for (size_t itok = 0; itok < ntok; ++itok) {
                    // 提取当前token的输入和输出缓冲区
                    std::shared_ptr<Tensor> hidden_states_i = hidden_states->slice(0, itok, 1);  // [1, d]
                    std::shared_ptr<Tensor> router_states_sum_i = router_states_sum->slice(0, itok, 1);  // [1, d]

                    // 第一个专家：初始化输出（alpha * Expert(hidden_states_i)）
                    {
                        int index = indices_cpu[itok * topk + 0];
                        float alpha = values_cpu[itok * topk + 0];
                        linear(router_gate_up_buf, hidden_states_i, layer_tensor->ffn->_experts[index]->w_ffn_gate_up, 1.0, 0.0, nullptr, nullptr);
                        swiglu(router_gate_buf, router_up_buf, router_gate_buf);
                        linear(router_states_sum_i, router_gate_buf, layer_tensor->ffn->_experts[index]->w_ffn_down, alpha, 0.0, nullptr, nullptr);
                    }

                    // 后续专家：累加到已有输出（alpha * Expert(hidden_states_i) + router_states_sum_i）
                    for (size_t k = 1; k < topk; ++k) {
                        int index = indices_cpu[itok * topk + k];
                        float alpha = values_cpu[itok * topk + k];
                        linear(router_gate_up_buf, hidden_states_i, layer_tensor->ffn->_experts[index]->w_ffn_gate_up, 1.0, 0.0, nullptr, nullptr);
                        swiglu(router_gate_buf, router_up_buf, router_gate_buf);
                        // 加权累加（注意这里使用router_states_sum_i作为残差，实现累加）
                        linear(router_states_sum_i, router_gate_buf, 
                               layer_tensor->ffn->_experts[index]->w_ffn_down, alpha, 0.0, router_states_sum_i, nullptr);
                    }
                }

                // 分布式AllReduce：聚合所有设备的MoE输出
                if (rsrc.comm != nullptr) {
                    RUN_INFINI(infinicclAllReduce(
                        router_states_sum->data(), router_states_sum->data(), ntok * d, dt_logits,
                        INFINICCL_SUM, rsrc.comm, stream));
                    RUN_INFINI(infinirtStreamSynchronize(stream));
                }
            }

            // Step 3: 残差连接
            // 将MoE输出与注意力输出相加，完成Transformer块的计算
            add(logits_in, router_states_sum, logits_in);
        }

        // All_reduce if distributed
        // if (rsrc.comm != nullptr) {
        //     RUN_INFINI(infinicclAllReduce(
        //         logits_in->data(), logits_in->data(), ntok * d, dt_logits,
        //         INFINICCL_SUM, rsrc.comm, stream));
        //     RUN_INFINI(infinirtStreamSynchronize(stream));
        // }
    }

    // Sample and Output
    if (idev == 0) {
        if (last_logits != nullptr) {
            rmsnorm(logits_out, logits_in, g_WeightsTensor->w_out_norm, meta->epsilon);
            auto last_logits_buf = Tensor::buffer(dt_logits, {ntok, dvoc}, rsrc.memory_pool);
            linear(last_logits_buf, logits_out, g_WeightsTensor->w_out_embd, 1.0, 0.0, nullptr, nullptr);
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
                        g_WeightsTensor->w_out_norm,
                        meta->epsilon);
            }
            linear(prob_buf, logits_out->slice(0, 0, nreq), g_WeightsTensor->w_out_embd, 1.0, 0.0, nullptr, nullptr);
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

namespace Qwen3MoE {
Model::Model(const Meta *_meta, const Weights *weights, infiniDevice_t device_, std::vector<int> device_ids) : meta(*_meta) {

    // 初始化设备相关参数
    int ndev = int(device_ids.size());  // 设备数量
    device = device_;
    dev_ids = device_ids;
    dev_resources = std::vector<DeviceResource<WeightsTensor>>(ndev);  // 每个设备的资源
    states = std::vector<InferState>(ndev);  // 每个设备的推理状态
    threads.resize(ndev);  // 每个设备的推理线程
    
    // 初始化InfiniRT运行时
    RUN_INFINI(infinirtInit());
    
    // 初始化通信器（用于多设备间的AllReduce）
    auto comms = std::vector<infinicclComm_t>(ndev, nullptr);
    if (ndev > 1) {
        RUN_INFINI(infinicclCommInitAll(device, comms.data(), ndev, dev_ids.data()));
    }

    // 为每个设备启动推理线程
    for (int i = 0; i < ndev; i++) {
        threads[i] = std::thread(launchDevice<WeightsTensor, Meta, Weights>, 
                                  std::cref(meta), weights, &dev_resources[i], 
                                  std::ref(states[i]), std::ref(req), device, i, ndev, 
                                  dev_ids[i], comms[i], Qwen3MoEinferDeviceBatch);
    }
    
    // 等待所有设备完成权重加载
    for (int i = 0; i < ndev; i++) {
        std::unique_lock<std::mutex> lock(states[i].mtx);
        states[i].cv_load.wait(lock, [&] { return states[i].loaded; });
        lock.unlock();
    }
}

}; // namespace Qwen3MoE
