#include "qwen3vl_impl.hpp"

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../inference_context.hpp"
#include "infinicore_infer.h"

#include <random>
#include <thread>
#include <vector>

void createDeviceResource(Qwen3vlDeviceResource *rsrc, const Qwen3vlMeta *meta,
                          std::shared_ptr<Qwen3vlDeviceWeights> weights,
                          infiniDevice_t device, int idev,
                          int ndev, int dev_id,
                          infinicclComm_t comm) {
    RUN_INFINI(infinirtSetDevice(device, dev_id));
    RUN_INFINI(infinirtStreamSynchronize(weights->load_stream));
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle);
    infinirtStream_t stream;
    infinirtStreamCreate(&stream);

    auto memory_pool = std::make_shared<MemoryPool>();

    *rsrc = Qwen3vlDeviceResource{
        device,
        dev_id,
        handle,
        weights,
        stream,
        comm,
        memory_pool,
    };
    RUN_INFINI(infinirtDeviceSynchronize());
}

void releaseDeviceResource(Qwen3vlDeviceResource &res) {
    infinirtDeviceSynchronize();

    res.weights.reset();

    infiniopDestroyHandle(res.handle);
    res.handle = nullptr;
    infinirtStreamDestroy(res.stream);
    res.stream = nullptr;
    infinicclCommDestroy(res.comm);
    res.comm = nullptr;
}

//todo:
// pd分离
// flashattn + batching
// triron跨平台
// pageattn


void inferDeviceBatch(const Qwen3vlMeta &meta, Qwen3vlDeviceResource &rsrc,
                      uint32_t idev, uint32_t ndev,
                      const uint32_t *tokens, uint32_t ntok,
                      const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                      struct Qwen3vlCache **caches,
                      const float *temperature, const uint32_t *topk, const float *topp,
                      uint32_t *output, void *last_logits) {
    assert(meta.text_meta.num_attention_heads % ndev == 0);
    assert(meta.text_meta.num_key_value_heads % ndev == 0);

    auto dtype = meta.dtype;
    printf("meta dtype: %d \n",dtype);
    auto nlayer = meta.text_meta.num_hidden_layers;
    size_t nh = meta.text_meta.num_attention_heads / size_t(ndev);
    size_t nkvh = meta.text_meta.num_key_value_heads / size_t(ndev);
    auto ngroup = nh / nkvh;
    auto dh = meta.text_meta.head_dim;
    auto d = meta.text_meta.hidden_size;
    auto di = meta.text_meta.intermediate_size / size_t(ndev);
    auto dvoc = meta.text_meta.vocab_size;
    float epsilon = meta.text_meta.rms_norm_eps;
    auto stream = rsrc.stream;
    auto weights = rsrc.weights;

    //Allocate buffers
    auto logits_in = Tensor::buffer(dtype, {ntok, d}, rsrc.memory_pool);
    auto logits_out = Tensor::buffer(dtype, {ntok, d}, rsrc.memory_pool);

    //所有请求的当前token
    auto qkv_buf = Tensor::buffer(dtype, {ntok, (nh + nkvh * 2) * dh}, rsrc.memory_pool);
    auto o_buf = Tensor::buffer(dtype, {ntok, nh * dh}, rsrc.memory_pool);
    auto gate_up_buf = Tensor::buffer(dtype, {ntok, 2*di}, rsrc.memory_pool);

    auto prob_buf = Tensor::buffer(dtype, {nreq, dvoc}, rsrc.memory_pool);
    auto result_buf = Tensor::buffer(INFINI_DTYPE_I64, {nreq}, rsrc.memory_pool);
    auto result_cpu = std::vector<int64_t>(nreq);

    //Prepare inputs
    auto batch_pos_ids = std::vector<uint32_t>(ntok);
    size_t req_start = 0;
    for (uint32_t req = 0; req < nreq; req++) {
        for (uint32_t i = 0; i < req_lens[req]; i++) { // req_len 本次query长度，req_pos 历史长度
            batch_pos_ids[req_start + i] = req_pos[req] + i;  //batch_pos_ids 展平后每个token的pos
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

    //convert tokens to embeddings
    for (uint32_t i = 0; i < ntok; i++) {
        RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d),
                                       weights->w_lang->in_embd->data(tokens[i] * d),
                                       dsize(dtype) * d, INFINIRT_MEMCPY_D2D, stream));
    }

    // attention inner
    size_t max_qk_size = 0;
    size_t max_seq_len = 0;

    for (uint32_t req = 0; req < nreq; req++) {
        auto past_len = req_pos[req];
        auto seq_len = req_lens[req];
        auto total_len = past_len + seq_len;

        max_qk_size = std::max(max_qk_size, size_t(seq_len * total_len));
        max_seq_len = std::max(max_seq_len, size_t(seq_len));
    }
    
    auto attn_score_buf = Tensor::buffer(dtype, {nh * max_qk_size}, rsrc.memory_pool);
    auto attn_val_buf = Tensor::buffer(dtype, {nh, max_seq_len, dh}, rsrc.memory_pool);
    auto rearrange_q_buf = Tensor::buffer(dtype, {nkvh, ngroup, max_seq_len, dh}, rsrc.memory_pool);
    auto qkv_rope = qkv_buf->view({ntok, nh + nkvh * 2, dh});
    auto q_buf = qkv_rope->slice(1,0,nh);
    auto k_buf = qkv_rope->slice(1,nh,nkvh);

    auto gate_buf = gate_up_buf->slice(1, 0, di);
    auto up_buf = gate_up_buf->slice(1, di, di);

    //Compute
    for (uint32_t i = 0; i < nlayer; i++){
        // attn norm
        rmsnorm(logits_out,logits_in,weights->w_lang->layers[i].attn_norm,epsilon);
        // qkv_proj
        linear(qkv_buf,logits_out,weights->w_lang->layers[i].attn_qkv_proj,1.0,0.0,nullptr,nullptr);
        // qk_norm
        rmsnorm(q_buf,q_buf,weights->w_lang->layers[i].attn_q_norm,epsilon);
        rmsnorm(k_buf,k_buf,weights->w_lang->layers[i].attn_k_norm,epsilon);
        // rope 
        rope_v2(q_buf,q_buf,pos_ids_buf,weights->sin_table,weights->cos_table);
        rope_v2(k_buf,k_buf,pos_ids_buf,weights->sin_table,weights->cos_table);
        
        // 逐个req处理
        size_t token_offset = 0;
        for(uint32_t req=0; req < nreq; req++){
            auto past_len = req_pos[req];
            auto seq_len = req_lens[req];
            auto total_len = past_len + seq_len;
            
            auto o = o_buf->slice(0,token_offset,seq_len)->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});// [nkvh, ngroup, seq_len, dh]
            auto q = qkv_rope->slice({{0,token_offset,seq_len},{1,0,nh}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});// [nkvh, ngroup, seq_len, dh]
            auto k = qkv_rope->slice({{0,token_offset,seq_len},{1,nh,nkvh}});// [ntok, nkvh, dh]
            auto v = qkv_rope->slice({{0,token_offset,seq_len},{1,nh+nkvh,nkvh}});// [ntok, nkvh, dh]

            // concat to cache 
            rearrange(caches[req]->k_rot[idev][i]->slice(0,past_len,seq_len),k);
            rearrange(caches[req]->v[idev][i]->slice(0,past_len,seq_len),v);

            //fill full_k full_v
            auto full_k_buff = caches[req]->k_rot[idev][i]->slice(0,0,total_len)->permute({1,2,0});// [nkvh, dh, total_len]
            auto full_v_buff = caches[req]->v[idev][i]->slice(0,0,total_len)->permute({1,0,2});// [nkvh, total_len, dh]

            //self-attn
            auto attn_score_req = attn_score_buf->slice(0,0,nh*seq_len*total_len)->view({nkvh, ngroup*seq_len, total_len});
            auto rearrange_q = rearrange_q_buf->slice(2,0,seq_len);
            rearrange(rearrange_q,q);
            // [nkvh, ngroup * seq_len, dh] @ [nkvh, dh, total_len] = [nkvh, ngroup * seq_len, total_len]
            linear(attn_score_req,rearrange_q->view({nkvh, ngroup * seq_len, dh}),full_k_buff,1.f / float(sqrt(dh)), 0.f, nullptr, nullptr);
            // softmax
            auto qk_softmax = attn_score_req->view({nh, seq_len, total_len});
            causalSoftmax(qk_softmax,qk_softmax);
            auto attn_val_req = attn_val_buf->slice(1,0,seq_len)->view({nkvh, ngroup * seq_len, dh});
            // [nkvh, ngroup * seq_len, total_len] @ [nkvh, total_len, dh] = [nkvh, ngroup * seq_len, dh]
            linear(attn_val_req, attn_score_req, full_v_buff, 1.0, 0.0, nullptr, nullptr);
            rearrange(o,attn_val_req->view({nkvh, ngroup, seq_len, dh}));
            token_offset += seq_len;
        }
        linear(logits_in, o_buf, weights->w_lang->layers[i].attn_o_proj, 1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr);
        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dtype,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }

        // mlp norm
        rmsnorm(logits_out,logits_in,weights->w_lang->layers[i].mlp_norm,epsilon);
        // mlp gate_up
        linear(gate_up_buf,logits_out,weights->w_lang->layers[i].mlp_gate_up,1.0,0.0,nullptr,nullptr);
        // silu
        silu(gate_buf,gate_buf);
        mul(gate_buf,gate_buf,up_buf);
        // mlp down
        linear(logits_in,gate_buf,weights->w_lang->layers[i].mlp_down,1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr);
        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dtype,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }
    }
    // sample and output
    if (idev == 0) {
        if (last_logits != nullptr) {
            rmsnorm(logits_out, logits_in, weights->w_lang->out_norm, meta.text_meta.rms_norm_eps);
            auto last_logits_buf = Tensor::buffer(dtype, {ntok, dvoc}, rsrc.memory_pool);
            linear(last_logits_buf, logits_out, weights->w_lang->out_embd, 1.0, 0.0, nullptr, nullptr);
            RUN_INFINI(infinirtStreamSynchronize(stream));
            RUN_INFINI(infinirtMemcpy(last_logits, last_logits_buf->data(), dsize(dtype) * ntok * dvoc, INFINIRT_MEMCPY_D2H));
        }
        if (output != nullptr) {
            size_t token_offset = 0;
            for (uint32_t req = 0; req < nreq; req++) {
                auto seq_len = req_lens[req];
                token_offset += seq_len;
                rmsnorm(logits_out->slice(0, req, 1),
                        logits_in->slice(0, token_offset - 1, 1),
                        weights->w_lang->out_norm,
                        meta.text_meta.rms_norm_eps);
            }
            //logits_out->slice(0,0,nreq)->debug(); different from transformers
            linear(prob_buf, logits_out->slice(0, 0, nreq), weights->w_lang->out_embd, 1.0, 0.0, nullptr, nullptr);
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

__C void
inferBatchQwen3vl(struct Qwen3vlModel *model,
                    const uint32_t *tokens, uint32_t ntok,
                    const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                    struct Qwen3vlCache **kv_caches,
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

__C void
forwardBatchQwen3vl(struct Qwen3vlModel *model,
                    const uint32_t *tokens, uint32_t ntok,
                    const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                    struct Qwen3vlCache **kv_caches,
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

void launchDevice(const Qwen3vlMeta &meta, std::shared_ptr<Qwen3vlDeviceWeights> weights, Qwen3vlDeviceResource *rsrc, InferState &state, InferRequest &req,
                  infiniDevice_t device, int idev, int ndev, int dev_id, infinicclComm_t comm) {
    // Create Device Resource
    createDeviceResource(rsrc, &meta, weights, device, idev, ndev, dev_id, comm);

    CacheManager cache_manager(100);
    InferenceContext ctx(rsrc->handle, rsrc->memory_pool, &cache_manager, rsrc->stream);

    // Set the inference context for this thread
    setInferenceContext(&ctx);

    {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.loaded = true;
        lock.unlock();
        state.cv_load.notify_one();
    }

    // Infer Loop
    while (true) {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.cv_start.wait(lock, [&] { return state.proceed || state.exit_flag; });
        // quit if exit_flag is set
        if (state.exit_flag) {
            break;
        }

        inferDeviceBatch(meta, *rsrc, idev, ndev, req.tokens, req.ntok,
                         req.req_lens, req.nreq, req.req_pos, req.kv_caches,
                         req.temperature, req.topk, req.topp, req.output, req.logits);

        state.proceed = false;
        lock.unlock();
        state.cv_done.notify_one();
    }

    // Clean-Up
    releaseDeviceResource(*rsrc);
    setInferenceContext(nullptr); // Clear the context when done
}


Qwen3vlModel::Qwen3vlModel(const Qwen3vlMeta *_meta, const Qwen3vlWeights *weights) : meta(*_meta) {
    auto device_weights = weights->device_weights;
    int ndev = device_weights.size();
    device = device_weights[0]->device;
    dev_ids.resize(ndev);
    for (int i = 0; i < ndev; i++) {
        dev_ids[i] = device_weights[i]->dev_id;
    }
    dev_resources = std::vector<Qwen3vlDeviceResource>(ndev);
    states = std::vector<InferState>(ndev);
    threads.resize(ndev);
    RUN_INFINI(infinirtInit());
    auto comms = std::vector<infinicclComm_t>(ndev, nullptr);
    if (ndev > 1) {
        RUN_INFINI(infinicclCommInitAll(device, comms.data(), ndev, dev_ids.data()));
    }
    for (int i = 0; i < ndev; i++) {
        threads[i] = std::thread(launchDevice, std::cref(meta), device_weights[i], &dev_resources[i], std::ref(states[i]), std::ref(req), device, i, ndev, dev_ids[i], comms[i]);
    }
    for (int i = 0; i < ndev; i++) {
        std::unique_lock<std::mutex> lock(states[i].mtx);
        states[i].cv_load.wait(lock, [&] { return states[i].loaded; });
        lock.unlock();
    }
}

__C struct Qwen3vlModel *
createQwen3vlModel(const Qwen3vlMeta *_meta,
                      const Qwen3vlWeights *weights) {
    Qwen3vlModel *model = new Qwen3vlModel(_meta, weights);
    return model;
}

__C void
destroyQwen3vlModel(struct Qwen3vlModel *model) {
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
