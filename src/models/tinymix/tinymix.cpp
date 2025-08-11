#include "tinymix_impl.hpp"
#include "tinymix_weight.hpp"
#include "infiniccl.h"

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "infinicore_infer/models/tinymix.h"

#include <random>
#include <thread>
#include <vector>
#include <iostream>
#include <numeric>

// --- Debugging Macro ---
// Set to 0 to disable all debug logs
#define DEBUG 1

#if DEBUG
#define DEBUG_LOG(...)                                                         \
    do {                                                                       \
        printf("DEBUG: ");                                                     \
        printf(__VA_ARGS__);                                                   \
        printf("\n");                                                          \
    } while (0)
#else
#define DEBUG_LOG(...)
#endif
// -----------------------

void createDeviceResource(DeviceResource *rsrc, const TinyMixMeta *meta,
                          const TinyMixWeights *weights,
                          infiniDevice_t device, int idev,
                          int ndev, int dev_id,
                          infinicclComm_t comm) {
    DEBUG_LOG("Enter createDeviceResource for idev: %d, dev_id: %d", idev, dev_id);
    RUN_INFINI(infinirtSetDevice(device, dev_id));
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle);
    infinirtStream_t stream;
    infinirtStreamCreate(&stream);

    std::vector<std::shared_ptr<Tensor>> w_attn_norm, w_attn_qkv, b_attn_qkv, w_attn_out,
        w_ffn_norm, w_ffn_gate;
    std::vector<std::vector<std::shared_ptr<Tensor>>> w_ffn_gate_up, w_ffn_down;
    
    for (size_t layer = 0; layer < meta->nlayer; layer++) {
		//DEBUG_LOG("L%zu: --- Starting weight loading ---", layer);
	
		//DEBUG_LOG("L%zu: Loading w_attn_norm", layer);
		w_attn_norm.push_back(getAttnNorm(meta, weights, layer));
	
		//DEBUG_LOG("L%zu: Loading w_attn_qkv", layer);
		w_attn_qkv.push_back(getAttnQKV(meta, weights, layer, idev, ndev));
	
		if (weights->attn_qkv_b != nullptr) {
			//DEBUG_LOG("L%zu: Loading b_attn_qkv (bias)", layer);
			b_attn_qkv.push_back(getAttnQKVBias(meta, weights, layer, idev, ndev));
		}
	
		//DEBUG_LOG("L%zu: Loading w_attn_out", layer);
		w_attn_out.push_back(getAttnO(meta, weights, layer, idev, ndev));
	
		//DEBUG_LOG("L%zu: Loading w_ffn_norm", layer);
		w_ffn_norm.push_back(getFFNNorm(meta, weights, layer));
	
		if (meta->nexpert > 1) {
			//DEBUG_LOG("L%zu: MoE model detected with %zu experts", layer, (size_t)meta->nexpert);
			//DEBUG_LOG("L%zu: Loading w_ffn_gate", layer);
			w_ffn_gate.push_back(getFFNGate(meta, weights, layer));
			
			std::vector<std::shared_ptr<Tensor>> gate_up_experts, down_experts;
			for (size_t expert = 0; expert < meta->nexpert; ++expert) {
				//DEBUG_LOG("L%zu E%zu: Loading gate_up tensor", layer, expert);
				gate_up_experts.push_back(getFFNGateUp(meta, weights, layer, expert, idev, ndev));
				
				//DEBUG_LOG("L%zu E%zu: Loading down tensor", layer, expert);
				down_experts.push_back(getFFNDown(meta, weights, layer, expert, idev, ndev));
			}
			w_ffn_gate_up.push_back(gate_up_experts);
			w_ffn_down.push_back(down_experts);
		} else {
			//DEBUG_LOG("L%zu: Standard FFN model detected", layer);
			std::vector<std::shared_ptr<Tensor>> gate_up, down;
			
			//DEBUG_LOG("L%zu: Loading FFN gate_up tensor", layer);
			gate_up.push_back(getFFNGateUp(meta, weights, layer, 0, idev, ndev));
			
			//DEBUG_LOG("L%zu: Loading FFN down tensor", layer);
			down.push_back(getFFNDown(meta, weights, layer, 0, idev, ndev));
			
			w_ffn_gate_up.push_back(gate_up);
			w_ffn_down.push_back(down);
		}
		//DEBUG_LOG("L%zu: --- Weight loading complete ---", layer);
	}
	//printf("enter memory pool\n");
    auto memory_pool = std::make_shared<MemoryPool>(128 * 1024 * 1024);
	//printf("memory pool created\n");
	//printf("Model's expected data type (enum value): %d\n", meta->dt_logits);

    *rsrc = DeviceResource{
        device,
        dev_id,
        handle,
        getInEmbd(meta, weights),
        getOutNorm(meta, weights),
        getOutEmbd(meta, weights),
        getSinTable(meta),
        getCosTable(meta),
        w_attn_norm,
        w_attn_qkv,
        b_attn_qkv,
        w_attn_out,
        w_ffn_norm,
        w_ffn_gate,
        w_ffn_gate_up,
        w_ffn_down,
        stream,
        comm,
        memory_pool,
    };
	printf("device resource created\n");
    RUN_INFINI(infinirtDeviceSynchronize());
    DEBUG_LOG("Exit createDeviceResource for idev: %d", idev);
}

// Define as static to limit scope to this file and prevent linker errors.
static void releaseDeviceResource(DeviceResource &res, int dev_id) {
    DEBUG_LOG("Releasing device resources for dev_id: %d", dev_id);
    // Similar to jiuge's implementation
    infinirtDeviceSynchronize();
    res.w_in_embd.reset();
    res.w_out_norm.reset();
    res.w_out_embd.reset();
    res.sin_table.reset();
    res.cos_table.reset();
    res.w_attn_norm.clear();
    res.w_attn_qkv.clear();
    res.b_attn_qkv.clear();
    res.w_attn_out.clear();
    res.w_ffn_norm.clear();
    res.w_ffn_gate.clear();
    res.w_ffn_gate_up.clear();
    res.w_ffn_down.clear();
    infiniopDestroyHandle(res.handle);
    res.handle = nullptr;
    infinirtStreamDestroy(res.stream);
    res.stream = nullptr;
    infinicclCommDestroy(res.comm);
    res.comm = nullptr;
    DEBUG_LOG("Finished releasing resources for dev_id: %d", dev_id);
}

void inferDeviceBatch(const TinyMixMeta &meta, DeviceResource &rsrc,
                      uint32_t idev, uint32_t ndev,
                      const uint32_t *tokens, uint32_t ntok,
                      const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                      struct KVCache **kv_caches,
                      const float *temperature, const uint32_t *topk, const float *topp,
                      uint32_t *output) {
    DEBUG_LOG("Enter inferDeviceBatch on idev %u. ntok: %u, nreq: %u", idev, ntok, nreq);
    auto nlayer = meta.nlayer;
    auto nkvh = meta.nkvh / ndev;
    auto nh = meta.nh / ndev;
    auto ngroup = nh / nkvh;
    (void)ngroup;
    auto dh = meta.d / meta.nh;
    auto d = meta.d;
    auto dt_logits = meta.dt_logits;
    auto di = meta.di / ndev;
    auto dvoc = meta.dvoc;
    auto stream = rsrc.stream;
    bool has_qkv_bias = rsrc.b_attn_qkv.size() > 0;
    (void)has_qkv_bias;

    // Allocate buffers
    auto logits_in = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto logits_out = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto qkv_buf = Tensor::buffer(dt_logits, {ntok, (nh + nkvh * 2) * dh}, rsrc.memory_pool);
    auto o_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, rsrc.memory_pool);
    auto prob_buf = Tensor::buffer(dt_logits, {nreq, dvoc}, rsrc.memory_pool);
    auto result_buf = Tensor::buffer(INFINI_DTYPE_I64, {nreq}, rsrc.memory_pool);
 
    // Prepare inputs
    auto batch_pos_ids = std::vector<uint32_t>(ntok);
    size_t req_start = 0;
    for (uint32_t req = 0; req < nreq; req++) {
        for (uint32_t i = 0; i < req_lens[req]; i++) {
            batch_pos_ids[req_start + i] = req_pos[req] + i;
        }
        req_start += req_lens[req];
    }

    std::shared_ptr<Tensor> pos_ids_buf;
    pos_ids_buf = Tensor::buffer(INFINI_DTYPE_U32, {ntok}, rsrc.memory_pool);
    RUN_INFINI(infinirtMemcpyAsync(pos_ids_buf->data(), batch_pos_ids.data(), sizeof(uint32_t) * ntok,
                                   INFINIRT_MEMCPY_H2D, stream));

    for (uint32_t i = 0; i < ntok; i++) {
        RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d),
                                       rsrc.w_in_embd->data(tokens[i] * d),
                                       dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
    }
    DEBUG_LOG("Input embeddings prepared. logits_in shape: {%u, %u}", ntok, d);

    // Prepare operators and workspace
    infiniopRMSNormDescriptor_t desc_norm_attn;
    RUN_INFINI(infiniopCreateRMSNormDescriptor(rsrc.handle, &desc_norm_attn, logits_in->desc(), logits_out->desc(), rsrc.w_attn_norm[0]->desc(), meta.epsilon));
    infiniopGemmDescriptor_t desc_attn_qkv;
    RUN_INFINI(infiniopCreateGemmDescriptor(rsrc.handle, &desc_attn_qkv, qkv_buf->desc(), logits_out->desc(), rsrc.w_attn_qkv[0]->desc()));
    infiniopAddDescriptor_t desc_add_qkv_bias;
    if (has_qkv_bias)
        RUN_INFINI(infiniopCreateAddDescriptor(rsrc.handle, &desc_add_qkv_bias, qkv_buf->desc(), qkv_buf->desc(), rsrc.b_attn_qkv[0]->desc()));
    
    infiniopRoPEDescriptor_t desc_rope;
    RUN_INFINI(infiniopCreateRoPEDescriptor(rsrc.handle, &desc_rope, qkv_buf->desc(), qkv_buf->desc(), rsrc.sin_table->desc(), rsrc.cos_table->desc(), pos_ids_buf->desc()));

    auto desc_kv_rearranges = std::vector<infiniopRearrangeDescriptor_t>(nreq);
    auto desc_q_rearranges = std::vector<infiniopRearrangeDescriptor_t>(nreq);
    auto desc_qk_gemms = std::vector<infiniopGemmDescriptor_t>(nreq);
    auto desc_qk_softmaxs = std::vector<infiniopCausalSoftmaxDescriptor_t>(nreq);
    auto desc_attn_v_gemms = std::vector<infiniopGemmDescriptor_t>(nreq);
    auto desc_attn_v_rearranges = std::vector<infiniopRearrangeDescriptor_t>(nreq);
    size_t token_offset_desc = 0;
    size_t max_qk_size = 0;
    size_t max_seq_len = 0;
    o_buf->dimSplit(1, {nh, dh});
    auto temp_qkv_buf = Tensor::buffer(dt_logits, {ntok, (nh + nkvh * 2) * dh}, rsrc.memory_pool);
    temp_qkv_buf->dimSplit(1, {nh + 2 * nkvh, dh});

    for (uint32_t req = 0; req < nreq; req++) {
        auto past_len = req_pos[req];
        auto seq_len = req_lens[req];
        auto total_len = past_len + seq_len;
        auto q = temp_qkv_buf->slice({{0, token_offset_desc, seq_len}, {1, 0, nh}});
        auto k = temp_qkv_buf->slice({{0, token_offset_desc, seq_len}, {1, nh, nkvh}});

        auto full_kv = kv_caches[req]->k[idev][0]->slice(0, 0, total_len)->permute({1, 2, 0});
        auto cache_kv = kv_caches[req]->k[idev][0]->slice(0, past_len, seq_len);

        RUN_INFINI(infiniopCreateRearrangeDescriptor(rsrc.handle, &desc_kv_rearranges[req],
                                                     cache_kv->desc(), k->desc()));

        q->dimSplit(1, {nkvh, ngroup})->permute({1, 2, 0, 3});
        auto q_t = TensorDesc::create(dt_logits, {nkvh, ngroup, seq_len, dh});
        RUN_INFINI(infiniopCreateRearrangeDescriptor(rsrc.handle, &desc_q_rearranges[req],
                                                     q_t->desc(), q->desc()));

        auto attn_v = TensorDesc::createWithOrder(dt_logits, {nkvh, ngroup, seq_len, dh}, {1, 2, 0, 3});
        RUN_INFINI(infiniopCreateRearrangeDescriptor(rsrc.handle, &desc_attn_v_rearranges[req],
                                                     attn_v->desc(), q_t->desc()));
        q_t = TensorDesc::create(dt_logits, {nkvh, ngroup * seq_len, dh});
        auto qk = TensorDesc::create(dt_logits, {nkvh, ngroup * seq_len, total_len});
        max_qk_size = std::max(max_qk_size, size_t(seq_len * total_len));
        max_seq_len = std::max(max_seq_len, size_t(seq_len));
        RUN_INFINI(infiniopCreateGemmDescriptor(
            rsrc.handle, &desc_qk_gemms[req], qk->desc(), q_t->desc(), full_kv->desc()));

        auto full_v = kv_caches[req]->v[idev][0]->slice(0, 0, total_len)->permute({1, 0, 2});
        RUN_INFINI(infiniopCreateGemmDescriptor(
            rsrc.handle, &desc_attn_v_gemms[req], q_t->desc(), qk->desc(), full_v->desc()));

        qk = TensorDesc::create(dt_logits, {nkvh * ngroup, seq_len, total_len});
        RUN_INFINI(infiniopCreateCausalSoftmaxDescriptor(
            rsrc.handle, &desc_qk_softmaxs[req], qk->desc(), qk->desc()));

        token_offset_desc += seq_len;
    }
    temp_qkv_buf->dimMerge(1, 2);
    o_buf->dimMerge(1, 2);
    auto qk_buf = Tensor::buffer(dt_logits, {nh, max_qk_size}, rsrc.memory_pool);
    auto rearrange_q_buf = Tensor::buffer(dt_logits, {nkvh, ngroup * max_seq_len, dh}, rsrc.memory_pool);
    auto attn_val_buf = Tensor::buffer(dt_logits, {nh, max_seq_len, dh}, rsrc.memory_pool);

    // The output projection GEMM descriptor, which also handles the residual addition
    infiniopGemmDescriptor_t desc_attn_out;
    RUN_INFINI(infiniopCreateGemmDescriptor(rsrc.handle, &desc_attn_out, logits_in->desc(), o_buf->desc(), rsrc.w_attn_out[0]->desc()));

    // FFN descriptors
    infiniopRMSNormDescriptor_t desc_norm_ffn;
    RUN_INFINI(infiniopCreateRMSNormDescriptor(rsrc.handle, &desc_norm_ffn, logits_in->desc(), logits_out->desc(), rsrc.w_ffn_norm[0]->desc(), meta.epsilon));
    
    // Descriptors for standard FFN path
    auto gate_up_buf = Tensor::buffer(dt_logits, {ntok, 2 * di}, rsrc.memory_pool);
    auto gate_buf = gate_up_buf->slice(1, 0, di);
    auto up_buf = gate_up_buf->slice(1, di, di);

    infiniopGemmDescriptor_t desc_ffn_gate_up, desc_ffn_down;
    infiniopSwiGLUDescriptor_t desc_swiglu;

    RUN_INFINI(infiniopCreateGemmDescriptor(rsrc.handle, &desc_ffn_gate_up, gate_up_buf->desc(), logits_out->desc(), rsrc.w_ffn_gate_up[0][0]->desc()));
    RUN_INFINI(infiniopCreateSwiGLUDescriptor(rsrc.handle, &desc_swiglu, gate_buf->desc(), up_buf->desc(), gate_buf->desc()));
    RUN_INFINI(infiniopCreateGemmDescriptor(rsrc.handle, &desc_ffn_down, logits_in->desc(), gate_buf->desc(), rsrc.w_ffn_down[0][0]->desc()));
 
    // Compute
    for (uint32_t layer = 0; layer < nlayer; layer++) {
        DEBUG_LOG("Starting layer %u", layer);
        // 1. Attention
        DEBUG_LOG("  (1) Attention");
        RUN_INFINI(infiniopRMSNorm(desc_norm_attn, nullptr, 0, logits_out->data(), logits_in->data(), rsrc.w_attn_norm[layer]->data(), stream));
        DEBUG_LOG("    - RMSNorm for attention done.");

        if (has_qkv_bias) {
            RUN_INFINI(infiniopGemm(desc_attn_qkv, nullptr, 0, qkv_buf->data(), logits_out->data(), rsrc.w_attn_qkv[layer]->data(), 1.0, 0.0, stream));
            RUN_INFINI(infiniopAdd(desc_add_qkv_bias, nullptr, 0, qkv_buf->data(), qkv_buf->data(), rsrc.b_attn_qkv[layer]->data(), stream));
        } else {
            RUN_INFINI(infiniopGemm(desc_attn_qkv, nullptr, 0, qkv_buf->data(), logits_out->data(), rsrc.w_attn_qkv[layer]->data(), 1.0, 0.0, stream));
        }
        DEBUG_LOG("    - QKV GEMM done. qkv_buf shape: {%u, %u}", ntok, (nh + nkvh * 2) * dh);
        
        qkv_buf->dimSplit(1, {nh + 2 * nkvh, dh});
        auto q_buf = qkv_buf->slice(1, 0, nh);
        auto k_buf = qkv_buf->slice(1, nh, nkvh);
        auto v_buf = qkv_buf->slice(1, nh + nkvh, nkvh);

        RUN_INFINI(infiniopRoPE(desc_rope, nullptr, 0, q_buf->data(), k_buf->data(), rsrc.sin_table->data(), rsrc.cos_table->data(), pos_ids_buf->data(), stream));
        DEBUG_LOG("    - RoPE done.");
        
        size_t token_offset = 0;
        o_buf->dimSplit(1, {nh, dh});
        for (uint32_t req = 0; req < nreq; req++) {
            auto past_len = req_pos[req];
            auto seq_len = req_lens[req];
            DEBUG_LOG("    - Processing attention for req %u, seq_len: %u, past_len: %u", req, seq_len, past_len);
            auto o = o_buf->slice({{0, token_offset, seq_len}});
            auto q = q_buf->slice({{0, token_offset, seq_len}});
            auto k = k_buf->slice({{0, token_offset, seq_len}});
            auto v = v_buf->slice({{0, token_offset, seq_len}});
            
            RUN_INFINI(infiniopRearrange(
                desc_kv_rearranges[req],
                kv_caches[req]->k[idev][layer]->data(past_len * nkvh * dh),
                k->data(), stream));
            RUN_INFINI(infiniopRearrange(
                desc_kv_rearranges[req],
                kv_caches[req]->v[idev][layer]->data(past_len * nkvh * dh),
                v->data(), stream));
            
            RUN_INFINI(infiniopRearrange(desc_q_rearranges[req], rearrange_q_buf->data(), q->data(), stream));
            
            RUN_INFINI(infiniopGemm(
                desc_qk_gemms[req], nullptr, 0,
                qk_buf->data(), rearrange_q_buf->data(), kv_caches[req]->k[idev][layer]->data(), 1. / sqrt(dh), 0.0, stream));
            
            RUN_INFINI(infiniopCausalSoftmax(
                desc_qk_softmaxs[req], nullptr, 0,
                qk_buf->data(), qk_buf->data(), stream));
            
            RUN_INFINI(infiniopGemm(
                desc_attn_v_gemms[req], nullptr, 0,
                attn_val_buf->data(), qk_buf->data(), kv_caches[req]->v[idev][layer]->data(), 1.0, 0.0, stream));
            
            RUN_INFINI(infiniopRearrange(
                desc_attn_v_rearranges[req],
                o->data(),
                attn_val_buf->data(), stream));

            token_offset += seq_len;
        }
        o_buf->dimMerge(1, 2);
        qkv_buf->dimMerge(1, 2);
        DEBUG_LOG("    - Per-request attention calculation done.");

        RUN_INFINI(infiniopGemm(desc_attn_out, nullptr, 0, logits_in->data(), o_buf->data(), rsrc.w_attn_out[layer]->data(), 1.0, 1.0, stream)); // Add residual
        DEBUG_LOG("    - Output projection and residual add done.");

        // 2. FFN / MoE
        DEBUG_LOG("  (2) FFN/MoE");
        RUN_INFINI(infiniopRMSNorm(desc_norm_ffn, nullptr, 0, logits_out->data(), logits_in->data(), rsrc.w_ffn_norm[layer]->data(), stream));
        DEBUG_LOG("    - RMSNorm for FFN done.");
        
        if (meta.nexpert > 1) {
            DEBUG_LOG("    - Taking MoE path with %u experts.", meta.nexpert);
            // MOE LOGIC
            // 步骤 1: Gating GEMM - 计算每个令牌对于每个专家的分数
            auto gating_scores = Tensor::buffer(dt_logits, {ntok, meta.nexpert}, rsrc.memory_pool);
            infiniopGemmDescriptor_t desc_gating_gemm;
            RUN_INFINI(infiniopCreateGemmDescriptor(rsrc.handle, &desc_gating_gemm, gating_scores->desc(), logits_out->desc(), rsrc.w_ffn_gate[layer]->desc()));
            RUN_INFINI(infiniopGemm(desc_gating_gemm, nullptr, 0, gating_scores->data(), logits_out->data(), rsrc.w_ffn_gate[layer]->data(), 1.0, 0.0, stream));
            infiniopDestroyGemmDescriptor(desc_gating_gemm);
            DEBUG_LOG("    - MoE: Gating GEMM done. gating_scores shape: {%u, %u}", ntok, meta.nexpert);
 
            // 步骤 2: TopK - 找出分数最高的 k (topk) 个专家
            auto topk_val = Tensor::buffer(dt_logits, {ntok, meta.topk}, rsrc.memory_pool);
            auto topk_ind = Tensor::buffer(INFINI_DTYPE_I32, {ntok, meta.topk}, rsrc.memory_pool);
            infiniopTopKDescriptor_t desc_topk;
            RUN_INFINI(infiniopCreateTopKDescriptor(rsrc.handle, &desc_topk, gating_scores->desc(), topk_val->desc(), topk_ind->desc(), nullptr, meta.topk, 1, 1, 1));
            size_t workspace_size = infiniopGetTopKWorkspaceSize(desc_topk);
            auto workspace = Tensor::buffer(INFINI_DTYPE_U8, {workspace_size}, rsrc.memory_pool);
            RUN_INFINI(infiniopTopKCalculate(desc_topk, gating_scores->data(), topk_val->data(), topk_ind->data(), nullptr, workspace->data(), stream));
            infiniopDestroyTopKDescriptor(desc_topk);
            DEBUG_LOG("    - MoE: TopK done. topk_ind shape: {%u, %u}", ntok, meta.topk);
 
            // 步骤 3: Dispatch - 根据 TopK 结果，将令牌的输入数据重新排列，发往对应专家
            auto permuted_output = Tensor::buffer(dt_logits, {ntok * meta.topk, d}, rsrc.memory_pool);
            auto aux_info = Tensor::buffer(INFINI_DTYPE_I32, {ntok, meta.topk, 4}, rsrc.memory_pool);
            infiniopMoEDispatchDescriptor_t desc_dispatch;
            RUN_INFINI(infiniopCreateMoEDispatchDescriptor(rsrc.handle, &desc_dispatch, meta.nexpert, logits_out->desc(), topk_ind->desc(), permuted_output->desc(), aux_info->desc()));
            RUN_INFINI(infiniopMoEDispatch(desc_dispatch, permuted_output->data(), aux_info->data(), logits_out->data(), topk_ind->data(), stream));
            infiniopDestroyMoEDispatchDescriptor(desc_dispatch);
            DEBUG_LOG("    - MoE: Dispatch done.");
 
            // 步骤 3.5: 【关键修复】计算每个专家需要处理的令牌数和数据偏移量
            // 这部分逻辑在您原来的代码中是缺失的
            auto expert_counts_gpu = Tensor::buffer(INFINI_DTYPE_I32, {meta.nexpert}, rsrc.memory_pool);
            auto expert_offsets_gpu = Tensor::buffer(INFINI_DTYPE_I32, {meta.nexpert}, rsrc.memory_pool);
            infiniopMoEExpertInfoDescriptor_t desc_expert_info;
            RUN_INFINI(infiniopCreateMoEExpertInfoDescriptor(rsrc.handle, &desc_expert_info, topk_ind->desc(), expert_counts_gpu->desc(), expert_offsets_gpu->desc()));
            RUN_INFINI(infiniopMoEExpertInfoCalculate(rsrc.handle, desc_expert_info, topk_ind->data(), expert_counts_gpu->data(), expert_offsets_gpu->data(), stream));
            infiniopDestroyMoEExpertInfoDescriptor(desc_expert_info);
 
            // 将计数和偏移量从 GPU 拷贝到 CPU，用于控制循环
            std::vector<int> expert_counts_cpu(meta.nexpert);
            std::vector<int> expert_offsets_cpu(meta.nexpert);
            RUN_INFINI(infinirtMemcpy(expert_counts_cpu.data(), expert_counts_gpu->data(), sizeof(int) * meta.nexpert, INFINIRT_MEMCPY_D2H));
            RUN_INFINI(infinirtMemcpy(expert_offsets_cpu.data(), expert_offsets_gpu->data(), sizeof(int) * meta.nexpert, INFINIRT_MEMCPY_D2H));
            DEBUG_LOG("    - MoE: Expert counts calculated.");
#if DEBUG
            for(size_t i = 0; i < meta.nexpert; ++i) {
                if (expert_counts_cpu[i] > 0) {
                    DEBUG_LOG("      - Expert %zu has %d tokens.", i, expert_counts_cpu[i]);
                }
            }
#endif
 
            // 步骤 4: 【关键修复】在数据切片上运行专家 FFN
            auto expert_gate_up_buf = Tensor::buffer(dt_logits, {ntok * meta.topk, 2 * di}, rsrc.memory_pool);
            auto expert_output_buf = Tensor::buffer(dt_logits, {ntok * meta.topk, d}, rsrc.memory_pool);
 
            for (size_t expert_id = 0; expert_id < meta.nexpert; ++expert_id) {
                int num_tokens_for_expert = expert_counts_cpu[expert_id];
                if (num_tokens_for_expert == 0) {
                    continue; // 如果没有令牌分配给这个专家，则跳过
                }
                int offset = expert_offsets_cpu[expert_id];
                DEBUG_LOG("    - MoE: Running FFN for expert %zu on %d tokens.", expert_id, num_tokens_for_expert);
 
                // 为当前专家的数据定义缓冲区切片
                auto input_slice = permuted_output->slice({{0, (uint32_t)offset, (uint32_t)num_tokens_for_expert}});
                auto gate_up_slice = expert_gate_up_buf->slice({{0, (uint32_t)offset, (uint32_t)num_tokens_for_expert}});
                auto gate_slice = gate_up_slice->slice(1, 0, di);
                auto up_slice = gate_up_slice->slice(1, di, di);
                auto output_slice = expert_output_buf->slice({{0, (uint32_t)offset, (uint32_t)num_tokens_for_expert}});
 
                // 为切片创建临时的算子描述符
                infiniopGemmDescriptor_t desc_ffn_gate_up_expert, desc_ffn_down_expert;
                infiniopSwiGLUDescriptor_t desc_swiglu_expert;
                RUN_INFINI(infiniopCreateGemmDescriptor(rsrc.handle, &desc_ffn_gate_up_expert, gate_up_slice->desc(), input_slice->desc(), rsrc.w_ffn_gate_up[layer][expert_id]->desc()));
                RUN_INFINI(infiniopCreateSwiGLUDescriptor(rsrc.handle, &desc_swiglu_expert, gate_slice->desc(), gate_slice->desc(), up_slice->desc()));
                RUN_INFINI(infiniopCreateGemmDescriptor(rsrc.handle, &desc_ffn_down_expert, output_slice->desc(), gate_slice->desc(), rsrc.w_ffn_down[layer][expert_id]->desc()));
 
                // 在数据切片上为当前专家执行 FFN 计算
                RUN_INFINI(infiniopGemm(desc_ffn_gate_up_expert, nullptr, 0, gate_up_slice->data(), input_slice->data(), rsrc.w_ffn_gate_up[layer][expert_id]->data(), 1.0, 0.0, stream));
                RUN_INFINI(infiniopSwiGLU(desc_swiglu_expert, nullptr, 0, gate_slice->data(), gate_slice->data(), up_slice->data(), stream));
                RUN_INFINI(infiniopGemm(desc_ffn_down_expert, nullptr, 0, output_slice->data(), gate_slice->data(), rsrc.w_ffn_down[layer][expert_id]->data(), 1.0, 0.0, stream));
 
                // 销毁临时描述符
                infiniopDestroyGemmDescriptor(desc_ffn_gate_up_expert);
                infiniopDestroySwiGLUDescriptor(desc_swiglu_expert);
                infiniopDestroyGemmDescriptor(desc_ffn_down_expert);
            }
 
            // 步骤 5: Combine - 将所有专家的输出与门控分数结合，并加上残差连接
            infiniopMoECombineDescriptor_t desc_combine;
            RUN_INFINI(infiniopCreateMoECombineDescriptor(rsrc.handle, &desc_combine, expert_output_buf->desc(), topk_val->desc(), aux_info->desc(), logits_in->desc()));
            RUN_INFINI(infiniopMoECombine(desc_combine, logits_in->data(), expert_output_buf->data(), topk_val->data(), aux_info->data(), stream));
            infiniopDestroyMoECombineDescriptor(desc_combine);
            DEBUG_LOG("    - MoE: Combine done.");
 
        } else {
            DEBUG_LOG("    - Taking standard FFN path.");
            // Standard FFN Logic (这部分保持不变)
            RUN_INFINI(infiniopGemm(desc_ffn_gate_up, nullptr, 0, gate_up_buf->data(), logits_out->data(), rsrc.w_ffn_gate_up[layer][0]->data(), 1.0, 0.0, stream));
            RUN_INFINI(infiniopSwiGLU(desc_swiglu, nullptr, 0, gate_buf->data(), up_buf->data(), gate_buf->data(), stream));
            RUN_INFINI(infiniopGemm(desc_ffn_down, nullptr, 0, logits_in->data(), gate_buf->data(), rsrc.w_ffn_down[layer][0]->data(), 1.0, 1.0, stream)); // Add residual
            DEBUG_LOG("    - Standard FFN computation done.");
        }

        // AllReduce across devices
        if (ndev > 1) {
            DEBUG_LOG("  (3) AllReduce");
            RUN_INFINI(infinicclAllReduce(logits_in->data(), logits_in->data(),
                                          std::accumulate(logits_in->shape().begin(), logits_in->shape().end(), size_t{1}, std::multiplies<size_t>()), meta.dt_logits,
                                          INFINICCL_SUM, rsrc.comm, stream));
            DEBUG_LOG("    - AllReduce done.");
        }
        DEBUG_LOG("Finished layer %u", layer);
    }

    // Final RMSNorm
    DEBUG_LOG("Starting final operations.");
    infiniopRMSNormDescriptor_t desc_norm_out;
    RUN_INFINI(infiniopCreateRMSNormDescriptor(rsrc.handle, &desc_norm_out, logits_in->slice(0, 0, 1)->desc(), logits_out->slice(0, 0, 1)->desc(), rsrc.w_out_norm->desc(), meta.epsilon));

    // Get logits for the last token of each request
    auto final_logits = Tensor::buffer(dt_logits, {nreq, d}, rsrc.memory_pool);
    size_t cum_pos = 0;
    for (uint32_t i = 0; i < nreq; i++) {
        cum_pos += req_lens[i];
        RUN_INFINI(infiniopRMSNorm(desc_norm_out, nullptr, 0, final_logits->data(i * d), logits_in->data((cum_pos - 1) * d), rsrc.w_out_norm->data(), stream));
    }
    DEBUG_LOG("Final RMSNorm done. final_logits shape: {%u, %u}", nreq, d);

    // Multiply by output embedding
    infiniopGemmDescriptor_t desc_output_gemm;
    RUN_INFINI(infiniopCreateGemmDescriptor(rsrc.handle, &desc_output_gemm, prob_buf->desc(), final_logits->desc(), rsrc.w_out_embd->desc()));
    RUN_INFINI(infiniopGemm(desc_output_gemm, nullptr, 0, prob_buf->data(), final_logits->data(), rsrc.w_out_embd->data(), 1.0, 0.0, stream));
    infiniopDestroyGemmDescriptor(desc_output_gemm);
    DEBUG_LOG("Output embedding GEMM done. prob_buf shape: {%u, %u}", nreq, dvoc);
 
    // Argmax sampling
    infiniopRandomSampleDescriptor_t desc_sample;
    RUN_INFINI(infiniopCreateRandomSampleDescriptor(rsrc.handle, &desc_sample, result_buf->desc(), prob_buf->desc()));
    
    std::random_device _rd;
    std::mt19937 gen(_rd());
    for (uint32_t req = 0; req < nreq; req++) {
        float random_val = std::uniform_real_distribution<float>(0, 1)(gen);
        RUN_INFINI(infiniopRandomSample(
            desc_sample, nullptr, 0,
            result_buf->data(req),
            prob_buf->data(req * dvoc),
            random_val,
            topp[req], topk[req], temperature[req],
            stream));
    }
    infiniopDestroyRandomSampleDescriptor(desc_sample);
    DEBUG_LOG("Sampling done.");

    // Copy result to output
    RUN_INFINI(infinirtMemcpyAsync(output, result_buf->data(), sizeof(uint32_t) * nreq, INFINIRT_MEMCPY_D2H, stream));
    RUN_INFINI(infinirtDeviceSynchronize());
    DEBUG_LOG("Result copied to host. Exiting inferDeviceBatch.");
 
    // Clean up descriptors
    infiniopDestroyRMSNormDescriptor(desc_norm_attn);
    infiniopDestroyGemmDescriptor(desc_attn_qkv);
    if(has_qkv_bias) infiniopDestroyAddDescriptor(desc_add_qkv_bias);
    infiniopDestroyRoPEDescriptor(desc_rope);
    for (uint32_t req = 0; req < nreq; req++) {
        infiniopDestroyRearrangeDescriptor(desc_kv_rearranges[req]);
        infiniopDestroyRearrangeDescriptor(desc_q_rearranges[req]);
        infiniopDestroyGemmDescriptor(desc_qk_gemms[req]);
        infiniopDestroyCausalSoftmaxDescriptor(desc_qk_softmaxs[req]);
        infiniopDestroyGemmDescriptor(desc_attn_v_gemms[req]);
        infiniopDestroyRearrangeDescriptor(desc_attn_v_rearranges[req]);
    }
    infiniopDestroyGemmDescriptor(desc_attn_out);
    infiniopDestroyRMSNormDescriptor(desc_norm_ffn);
    infiniopDestroyRMSNormDescriptor(desc_norm_out);
    infiniopDestroyGemmDescriptor(desc_ffn_gate_up);
    infiniopDestroySwiGLUDescriptor(desc_swiglu);
    infiniopDestroyGemmDescriptor(desc_ffn_down);
}

// Boilerplate code for model creation, destruction, and thread management
// This part is very similar to jiuge.cpp and can be adapted directly.
// To save space, I will omit the full copy-paste here but it should include:
// - launchDevice
// - TinyMixModel::TinyMixModel
// - createTinyMixModel
// - destroyTinyMixModel
// - inferBatchTinyMix

void launchDevice(const TinyMixMeta &meta, const TinyMixWeights *weights, DeviceResource *rsrc, InferState &state, InferRequest &req,
                  infiniDevice_t device, int idev, int ndev, int dev_id, infinicclComm_t comm) {
    createDeviceResource(rsrc, &meta, weights, device, idev, ndev, dev_id, comm);
    {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.loaded = true;
        lock.unlock();
        state.cv_load.notify_one();
    }

    while (true) {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.cv_start.wait(lock, [&] { return state.proceed || state.exit_flag; });
        if (state.exit_flag) {
            DEBUG_LOG("Exit flag received in launchDevice for dev_id: %d. Exiting thread.", dev_id);
            break;
        }

        DEBUG_LOG("Starting inference on dev_id: %d.", dev_id);
        inferDeviceBatch(meta, *rsrc, idev, ndev, req.tokens, req.ntok, req.req_lens, req.nreq, req.req_pos, req.kv_caches, req.temperature, req.topk, req.topp, req.output);
        DEBUG_LOG("Finished inference on dev_id: %d.", dev_id);

        state.proceed = false;
        lock.unlock();
        state.cv_done.notify_one();
    }

    releaseDeviceResource(*rsrc, dev_id);
}

TinyMixModel::TinyMixModel(const TinyMixMeta *_meta, const TinyMixWeights *weights, infiniDevice_t device_, std::vector<int> device_ids) : meta(*_meta) {
    int ndev = int(device_ids.size());
    DEBUG_LOG("Creating TinyMixModel with %d devices.", ndev);
    device = device_;
    dev_ids = device_ids;
    dev_resources = std::vector<DeviceResource>(ndev);
    states = std::vector<InferState>(ndev);
    threads.resize(ndev);
    RUN_INFINI(infinirtInit());
    auto comms = std::vector<infinicclComm_t>(ndev, nullptr);
    if (ndev > 1) {
        DEBUG_LOG("Initializing NCCL communicators.");
        RUN_INFINI(infinicclCommInitAll(device, comms.data(), ndev, dev_ids.data()));
    }

    for (int i = 0; i < ndev; i++) {
        threads[i] = std::thread(launchDevice, std::cref(meta), weights, &dev_resources[i], std::ref(states[i]), std::ref(req), device, i, ndev, dev_ids[i], comms[i]);
    }
    for (int i = 0; i < ndev; i++) {
        std::unique_lock<std::mutex> lock(states[i].mtx);
        states[i].cv_load.wait(lock, [&] { return states[i].loaded; });
        lock.unlock();
    }
    DEBUG_LOG("All device threads launched and resources loaded.");
}

__C struct TinyMixModel *
createTinyMixModel(const TinyMixMeta *meta,
                   const TinyMixWeights *weights,
                   infiniDevice_t device,
                   int ndev,
                   const int *dev_ids) {
    std::vector<int> device_ids(ndev);
    std::copy(dev_ids, dev_ids + ndev, device_ids.begin());
    TinyMixModel *model = new TinyMixModel(meta, weights, device, device_ids);
    return model;
}

__C void destroyTinyMixModel(struct TinyMixModel *model) {
    DEBUG_LOG("Destroying TinyMixModel.");
    auto ndev = model->dev_resources.size();

    for (size_t idev = 0; idev < ndev; idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].exit_flag = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }

    for (size_t idev = 0; idev < ndev; idev++) {
        model->threads[idev].join();
        DEBUG_LOG("Thread for device %zu joined.", idev);
    }

    delete model;
    DEBUG_LOG("TinyMixModel destroyed successfully.");
}

__C void
inferBatchTinyMix(struct TinyMixModel *model,
                  const uint32_t *tokens, uint32_t ntok,
                  const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                  struct KVCache **kv_caches,
                  const float *temperature, const uint32_t *topk, const float *topp,
                  uint32_t *output) {
    DEBUG_LOG("Enter inferBatchTinyMix. ntok: %u, nreq: %u", ntok, nreq);
    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.kv_caches = kv_caches;
    model->req.output = output;
    model->req.temperature = temperature;
    model->req.topk = topk;
    model->req.topp = topp;

    for (size_t idev = 0; idev < model->dev_ids.size(); idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].proceed = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }
    DEBUG_LOG("Notified all device threads to start inference.");
    for (size_t i = model->dev_ids.size(); i > 0; i--) {
        auto idev = i - 1;
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].cv_done.wait(lock, [&] { return !(model->states[idev].proceed); });
        lock.unlock();
    }
    DEBUG_LOG("All device threads finished inference. Exiting inferBatchTinyMix.");
}
