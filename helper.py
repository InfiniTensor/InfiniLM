 auto nlayer = meta.nlayer;
    auto nkvh = meta.nkvh / ndev;      // 每个设备的KV头数 (GQA分组查询)
    auto nh = meta.nh / ndev;          // 每个设备的注意力头数
    auto ngroup = nh / nkvh;           // 每个设备的组数 (多少个Q头共享一个K/V头)
    auto dctx = meta.dctx;             // 最大上下文长度
    auto dh = meta.dh;                 // 每个头的维度
    auto d = meta.d;                   // 模型隐藏层维度
    auto dt_logits = meta.dt_logits;   // 输出数据类型
    auto di_dense = meta.di_dense / ndev;  // 每个设备的密集FFN中间维度
    auto di_expert = meta.di_expert / ndev; // 每个设备的专家FFN中间维度
    auto dvoc = meta.dvoc;             // 词汇表大小
    auto stream = rsrc.stream;
    bool has_qkv_bias = rsrc.b_attn_qkv.size() > 0;      // 是否有QKV偏置
    bool has_qk_norm = rsrc.w_attn_q_norm.size() > 0 && rsrc.w_attn_k_norm.size() > 0;  // 是否有QK归一化

    //Allocate Buffer
    std::shared_ptr<Tensor> o_buf = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    std::cout << "Allocating  Buffer " << std::endl;
    auto logits_in = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto logits_out = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto qkv_buf = Tensor::buffer(dt_logits, {ntok, (nh + nkvh * 2) * dh}, rsrc.memory_pool);
    auto gate_up_buf = Tensor::buffer(dt_logits, {ntok, 2 * di_expert}, rsrc.memory_pool); // ?
    auto o_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, rsrc.memory_pool);
    auto prob_buf = Tensor::buffer(dt_logits, {nreq, dvoc}, rsrc.memory_pool);
    auto result_buf = Tensor::buffer(INFINI_DTYPE_I64, {nreq}, rsrc.memory_pool);
    auto result_cpu = std::vector<int64_t>(nreq);
    auto qkv_rope = qkv_buf->view({ntok, nh + nkvh * 2, dh});
    auto q_buf = qkv_rope->slice(1, 0, nh);
    auto k_buf = qkv_rope->slice(1, nh, nkvh);

    // ====================================================================
    // 1. 输入预处理和Token嵌入
    // ====================================================================

    // 1.1 位置编码准备 - 生成序列中每个token的位置ID
    // requests = [
    // "Hello world",      # req_idx=0, 长度=2, 起始位置=0
    // "How are you",      # req_idx=1, 长度=3, 起始位置=2  
    // "Fine thanks",      # req_idx=2, 长度=2, 起始位置=5
    // ]

    // # 输入参数:
    // req_pos = [0, 2, 5]     # 每个请求的起始位置
    // req_lens = [2, 3, 2]    # 每个请求的长度
    // nreq = 3                # 3个请求
    // ntok = 7                # 总共7个token

    // # 执行后结果:
    // batch_pos_ids = [0, 1, 2, 3, 4, 5, 6]
    auto batch_pos_ids = std::vector<uint32_t>(ntok);
    // 根据请求信息填充位置ID (req_pos指示每个请求的起始位置)
    std::cout << "PreFill Id " << "req len is " << nreq << std::endl;
    for (uint32_t req_idx = 0; req_idx < nreq; ++req_idx) {
        uint32_t start_pos = req_pos[req_idx];
        uint32_t req_len = req_lens[req_idx];
        for (uint32_t i = 0; i < req_len; ++i) {
            batch_pos_ids[start_pos + i] = start_pos + i;
        }
    }

    // 1.2 将位置ID复制到GPU (如果是GPU设备)
    std::shared_ptr<Tensor> pos_ids_buf;
    if (rsrc.device == INFINI_DEVICE_CPU) {
        pos_ids_buf = Tensor::weight(batch_pos_ids.data(), INFINI_DTYPE_U32, {ntok});
    } else { // GPU
        pos_ids_buf = Tensor::buffer(INFINI_DTYPE_U32, {ntok}, rsrc.memory_pool);
        RUN_INFINI(infinirtMemcpyAsync(pos_ids_buf->data(), batch_pos_ids.data(), sizeof(uint32_t) * ntok,
                                       INFINIRT_MEMCPY_H2D, stream));
    }

    // 1.3 Token嵌入查找 - 将token ID转换为对应的嵌入向量
    // 输入: tokens[token_count] -> 输出: [token_count, hidden_dim]
    std::shared_ptr<Tensor> hidden_states = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    for (uint32_t i = 0; i < ntok; i++) {
        // 查找词嵌入表: [vocab_size, d] -> 输出token i的嵌入向量
        RUN_INFINI(infinirtMemcpyAsync(hidden_states->data(i * d),
                                       rsrc.w_in_embd->data(tokens[i] * d),
                                       dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
    }

    // ====================================================================
    // 2. 位置编码应用 - RoPE (Rotary Position Embedding)
    // ====================================================================

    // 2.1 从预计算的RoPE表中获取当前位置的正弦余弦值
    // sin_table: [max_seq_len, head_dim/2], cos_table: [max_seq_len, head_dim/2]
    std::shared_ptr<Tensor> sin_cos_for_pos = Tensor::buffer(dt_logits, {ntok, dh}, rsrc.memory_pool);
    // TODO: 根据batch_pos_ids从sin_table和cos_table中提取对应的正弦余弦值

    // ====================================================================
    // 3. Transformer层处理循环
    // ====================================================================

    // 设置推理上下文 - 必须在调用rmsnorm之前

    CacheManager cache_manager(100);
    InferenceContext ctx(rsrc.handle, rsrc.memory_pool, &cache_manager, rsrc.stream);
    setInferenceContext(&ctx);
    std::cout << "[DEBUG] InferenceContext created and set successfully" << std::endl;

    // 保存每层的输入，用于残差连接
    std::shared_ptr<Tensor> layer_input = hidden_states;

    for (uint32_t layer_idx = 0; layer_idx < nlayer; ++layer_idx) {
        std::cout << "Processing layer " << layer_idx << std::endl;

        // ====================================================================
        // 3.1 注意力前归一化 (Input Layernorm)
        // ====================================================================
       std::cout << "Input LayerNorm" << std::endl;
        // RMSNorm: hidden_states = hidden_states * rms_norm_weight / sqrt(mean(hidden_states^2) + eps)
        std::shared_ptr<Tensor> attn_input = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
        // 实现RMSNorm归一化 - 参考jiuge.cpp:218
        std::cout << "Layer Idx " << layer_idx << " Do rmsnorm" << std::endl;

        // 检查权重是否为空
        if (!rsrc.w_attn_norm[layer_idx]) {
            std::cerr << "ERROR: rsrc.w_attn_norm[" << layer_idx << "] is null!" << std::endl;
            return;
        }
        if (!layer_input) {
            std::cerr << "ERROR: layer_input is null!" << std::endl;
            return;
        }
        if (!attn_input) {
            std::cerr << "ERROR: attn_input is null!" << std::endl;
            return;
        }
        if (!rsrc.w_attn_norm[layer_idx]->data()) {
            std::cerr << "ERROR: rsrc.w_attn_norm[" << layer_idx << "] data pointer is null!" << std::endl;
            return;
        }

        std::cout << "All pointers are valid, calling rmsnorm..." << std::endl;
        rmsnorm(attn_input, layer_input, rsrc.w_attn_norm[layer_idx], meta.epsilon);

        // ====================================================================
        // 3.2 QKV线性投影
        // ====================================================================
        // 输入: [ntok, d] -> Q: [ntok, nh, dh], K: [ntok, nkvh, dh], V: [ntok, nkvh, dh]
        std::cout << "Initlilizing qkv buffer " << std::endl;
        std::shared_ptr<Tensor> qkv_buf = Tensor::buffer(dt_logits, {ntok, (nh + nkvh * 2) * dh}, rsrc.memory_pool);

        
        // 执行线性投影: [ntok, d] × [d, (nh + 2*nkvh)*dh] = [ntok, (nh + 2*nkvh)*dh]
        // TODO: 实现QKV矩阵乘法
        // RUN_INFINI(infiniopGemm(..., attn_input, rsrc.w_attn_qkv[layer_idx], qkv_buf, ...));
        std::cout << "Linear project qkv matrix" << std::endl;
        linear(qkv_buf, attn_input, rsrc.w_attn_qkv[layer_idx], 1.0, 0.0, nullptr, nullptr);
        // 如果有QKV偏置，加上偏置
        // if (has_qkv_bias) {
        //     // TODO: 实现偏置加法
        //     // RUN_INFINI(infiniopAdd(..., qkv_buf, rsrc.b_attn_qkv[layer_idx], qkv_buf, ...));
        // }
        // ====================================================================
        // 3.3 QKV重塑和分离
        // ====================================================================
        std::cout << "Initlilizing qkv split  " << std::endl;
        auto qkv_rope = qkv_buf->view({ntok, nh + nkvh * 2, dh});
        auto q_buf = qkv_rope->slice(1, 0, nh);           // [ntok, nh, dh] - Query部分
        auto k_buf = qkv_rope->slice(1, nh, nkvh);       // [ntok, nkvh, dh] - Key部分
        auto v_buf = qkv_rope->slice(1, nh+nkvh, nkvh);  // [ntok, nkvh, dh] - Value部分

        // ====================================================================
        // 3.4 QK归一化 (可选)
        // ====================================================================
        std::cout << "QK Normalization" << std::endl;
        if (has_qk_norm) {
            // 对Q和K分别应用RMSNorm
            // TODO: 实现QK归一化
            // RUN_INFINI(infiniopRMSNorm(..., q_buf, rsrc.w_attn_q_norm[layer_idx], q_buf, ...));
            // RUN_INFINI(infiniopRMSNorm(..., k_buf, rsrc.w_attn_k_norm[layer_idx], k_buf, ...));
            rmsnorm(q_buf, q_buf, rsrc.w_attn_q_norm[layer_idx], meta.epsilon);
            rmsnorm(k_buf, k_buf, rsrc.w_attn_k_norm[layer_idx], meta.epsilon);
        }

        // ====================================================================
        // 3.5 RoPE位置编码应用
        // ====================================================================
        // 将正弦余弦编码应用到Q和K张量
        // 输入: q_buf: [ntok, nh, dh], k_buf: [ntok, nkvh, dh]
        // 位置编码: sin_cos_for_pos: [ntok, dh]
        // TODO: 实现RoPE旋转位置编码
        std::cout << "Position Embedding" << std::endl;
        rope(q_buf, q_buf, pos_ids_buf, rsrc.sin_table, rsrc.cos_table);
        rope(k_buf, k_buf, pos_ids_buf, rsrc.sin_table, rsrc.cos_table);
        // 注意：rope函数会同时处理q和k，所以第二个调用可能不需要 buffer created
        // ====================================================================
        // 3.6 注意力计算
        // ====================================================================
        size_t token_offset = 0;
        for(uint32_t req = 0; req < nreq; req ++){
            auto past_len = req_pos[req];
            auto seq_len  = req_lens[req];
            auto total = past_len + seq_len; 
            auto o = o_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});
            auto q = qkv_rope->slice({{0, token_offset, seq_len}, {1, 0, nh}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});
            auto k = qkv_rope->slice({{0, token_offset, seq_len}, {1, nh, nkvh}});
            auto v = qkv_rope->slice({{0, token_offset, seq_len}, {1, nh + nkvh, nkvh}});

            // self attention
            // concat
            rearrange(kv_caches[req]->k[idev][layer_idx]->slice(0, past_len, seq_len), k);
            rearrange(kv_caches[req]->v[idev][layer_idx]->slice(0, past_len, seq_len), v);
            // qk
            rearrange(q_rearrange->slice(2, 0, seq_len), q);
            auto qk_gemm = qk_buf->slice(0, 0, nh * seq_len * total_len)->view({nkvh, ngroup * seq_len, total_len});
            auto k_gemm = kv_caches[req]->k[idev][layer]->slice(0, 0, total_len)->permute({1, 2, 0});
            linear(qk_gemm, rearrange_q_buf->slice(1, 0, ngroup * seq_len), k_gemm, 1.f / float(sqrt(dh)), 0.f, nullptr, nullptr);
            // softmax
            auto qk_softmax = qk_gemm->view({nh, seq_len, total_len});
            causalSoftmax(qk_softmax, qk_softmax);
            auto v_gemm = kv_caches[req]->v[idev][layer]->slice(0, 0, total_len)->permute({1, 0, 2});
            linear(attn_val_buf->slice(1, 0, ngroup * seq_len), qk_gemm, v_gemm, 1.f, 0.f, nullptr, nullptr);
            // rearrange attn val
            rearrange(o, attn_val_gemm->slice(2, 0, seq_len));

            token_offset += seq_len;
        }
        // 3.6.1 KV缓存处理
        // 将当前的K和V写入到KV缓存中，用于自回归生成
        // TODO: 实现KV缓存写入
        // update_kv_cache(kv_caches, layer_idx, k_buf, v_buf, ...);

        // 3.6.2 读取历史KV (如果有缓存)
        // TODO: 从KV缓存中读取历史K和V
        // [seq_len_cached + ntok, nkvh, dh]

        // 3.6.3 GQA处理 - 重复KV以匹配Q的维度
        // 将[n_tok, nkvh, dh]的K和V重复ngroup次变为[n_tok, nh, dh]
        std::shared_ptr<Tensor> k_repeated = Tensor::buffer(dt_logits, {ntok, nh, dh}, rsrc.memory_pool);
        std::shared_ptr<Tensor> v_repeated = Tensor::buffer(dt_logits, {ntok, nh, dh}, rsrc.memory_pool);
        // TODO: 实现KV重复操作
        // repeat_kv(k_buf, k_repeated, ngroup);
        // repeat_kv(v_buf, v_repeated, ngroup);

        // 3.6.4 注意力权重计算
        // Q @ K^T / sqrt(dh) -> [ntok, nh, seq_len]
        std::shared_ptr<Tensor> attn_weights = Tensor::buffer(dt_logits, {ntok, nh, ntok}, rsrc.memory_pool);
        // TODO: 实现注意力权重计算
        // RUN_INFINI(infiniopBatchedGemm(..., q_buf, k_repeated, attn_weights, ...));
        // attn_weights = attn_weights / sqrt(dh);

        // 3.6.5 注意力归一化
        // TODO: 实现Softmax归一化
        // RUN_INFINI(infiniopSoftmax(..., attn_weights, attn_weights, ...));

        // 3.6.6 注意力输出计算
        // attn_weights @ V -> [ntok, nh, dh]
        std::shared_ptr<Tensor> attn_output = Tensor::buffer(dt_logits, {ntok, nh, dh}, rsrc.memory_pool);
        // TODO: 实现注意力输出计算
        // RUN_INFINI(infiniopBatchedGemm(..., attn_weights, v_repeated, attn_output, ...));

        // 3.6.7 输出投影
        // 将多头输出合并: [ntok, nh, dh] -> [ntok, d]
        // 然后通过输出投影层: [ntok, d] × [d, d] = [ntok, d]
        std::shared_ptr<Tensor> attn_proj = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
        // TODO: 实现注意力输出投影
        // RUN_INFINI(infiniopGemm(..., attn_output, rsrc.w_attn_out[layer_idx], attn_proj, ...));

        // ====================================================================
        // 3.7 残差连接和后注意力归一化
        // ====================================================================
        // 残差连接: layer_input + attn_proj
        std::shared_ptr<Tensor> residual1 = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
        // TODO: 实现残差加法
        // RUN_INFINI(infiniopAdd(..., layer_input, attn_proj, residual1, ...));

        // 后注意力归一化
        std::shared_ptr<Tensor> mlp_input = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
        // TODO: 实现RMSNorm
        // RUN_INFINI(infiniopRMSNorm(..., residual1, rsrc.w_ffn_norm[layer_idx], mlp_input, ...));

        // ====================================================================
        // 3.8 专家混合(MoE)处理
        // ====================================================================

        // 3.8.1 路由器计算
        // 输入: [ntok, d] -> 输出: [ntok, num_experts] (每个token对每个专家的权重)
        std::shared_ptr<Tensor> router_logits = Tensor::buffer(dt_logits, {ntok, meta.num_experts}, rsrc.memory_pool);
        // TODO: 实现路由器线性层
        // RUN_INFINI(infiniopGemm(..., mlp_input, w_router, router_logits, ...));

        // 3.8.2 TopK专家选择
        // 对每个token选择top_k个专家
        // TODO: 实现TopK选择和Softmax归一化
        // top_k_experts, top_k_weights = topk_softmax(router_logits, k=2);

        // 3.8.3 专家计算
        // 初始化MoE输出为0
        std::shared_ptr<Tensor> moe_output = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
        // RUN_INFINI(infinirtMemsetAsync(moe_output->data(), 0, ...));

        // 对每个专家进行计算
        for (uint32_t expert_idx = 0; expert_idx < meta.num_experts; ++expert_idx) {
            // 找到被路由到当前专家的token
            // TODO: 找到被路由到当前专家的token索引

            // 3.8.3.1 专家FFN计算
            // Gate + Up投影: [token_count, d] × [d, 2*di_expert] = [token_count, 2*di_expert]
            // std::shared_ptr<Tensor> gate_up_output = Tensor::buffer(dt_logits, {1, 2 * di_expert}, rsrc.memory_pool);
            // TODO: 实现Gate+Up投影
            // RUN_INFINI(infiniopGemm(..., expert_tokens, w_gate_up[layer_idx][expert_idx], gate_up_output, ...));

            // 激活函数: 通常使用SiLU (Swish)
            // TODO: 实现SiLU激活函数
            // RUN_INFINI(infiniopSilu(..., gate_up_output, gate_up_output, ...));

            // Down投影: [token_count, 2*di_expert] × [2*di_expert, d] = [token_count, d]
            // std::shared_ptr<Tensor> expert_output = Tensor::buffer(dt_logits, {expert_token_count, d}, rsrc.memory_pool);
            // TODO: 实现Down投影
            // RUN_INFINI(infiniopGemm(..., gate_up_output, w_down[layer_idx][expert_idx], expert_output, ...));

            // 3.8.3.2 加权求和
            // 将专家输出按路由权重加权加到总输出中
            // TODO: 实现加权求和
            // moe_output += expert_output * expert_weights
        }

        // ====================================================================
        // 3.9 第二个残差连接
        // ====================================================================
        // residual1 + moe_output -> next_layer_input
        // std::shared_ptr<Tensor> next_layer_input = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
        // TODO: 实现残差加法
        // RUN_INFINI(infiniopAdd(..., residual1, moe_output, next_layer_input, ...));

        // 更新下一层的输入
        // layer_input = next_layer_input;

        // 同步当前层的计算
        // RUN_INFINI(infinirtStreamSynchronize(stream));
    }

    // ====================================================================
    // 4. 最终输出层
    // ====================================================================

    // 4.1 最终归一化
    // std::shared_ptr<Tensor> final_norm = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    // TODO: 实现最终RMSNorm
    // RUN_INFINI(infiniopRMSNorm(..., layer_input, rsrc.w_out_norm, final_norm, ...));

    // 4.2 输出投影到词汇表
    // 输入: [ntok, d] × 输出权重: [d, dvoc] = [ntok, dvoc]
    // std::shared_ptr<Tensor> logits = Tensor::buffer(dt_logits, {ntok, dvoc}, rsrc.memory_pool);
    // TODO: 实现输出投影
    // RUN_INFINI(infiniopGemm(..., final_norm, rsrc.w_out_embd, logits, ...));

    // ====================================================================
    // 5. 采样/解码
    // ====================================================================

    // // 5.1 如果是推理模式，对每个序列的最后一个token进行采样
    // for (uint32_t req_idx = 0; req_idx < nreq; ++req_idx) {
    //     uint32_t last_token_pos = req_pos[req_idx] + req_lens[req_idx] - 1;

    //     // 获取最后一个token的logits
    //     std::shared_ptr<Tensor> last_token_logits = logits->slice(0, last_token_pos, last_token_pos + 1); // [1, dvoc]

    //     // 5.2 应用温度
    //     if (temperature[req_idx] != 0.0) {
    //         // TODO: 实现温度缩放: logits = logits / temperature
    //     }

    //     // 5.3 TopK/TopP采样
    //     if (topk[req_idx] > 1) {
    //         // TODO: 实现TopK采样
    //     } else if (topp[req_idx] < 1.0) {
    //         // TODO: 实现TopP (Nucleus) 采样
    //     } else {
    //         // 直接取argmax
    //         // TODO: 实现Argmax
    //         // RUN_INFINI(infiniopArgmax(..., last_token_logits, &output[req_idx], ...));
    //     }
    // }

    // 同步所有计算
    RUN_INFINI(infinirtStreamSynchronize(stream));

    // 清理推理上下文
    setInferenceContext(nullptr);

    std::cout << "InferDeviceBatch completed" << std::endl;