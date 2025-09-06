#include "qwen_hybrid.hpp"

#include "../../context/inference_context.hpp"
#include "../../tensor.hpp"
#include "../../utils.hpp"

#include <cmath>

struct DeviceResource {
public:
    // Device
    infiniDevice_t device;
    int device_id;
    // Op handle
    infiniopHandle_t handle;
    // Streams
    infinirtStream_t stream;
    // Communicator
    infinicclComm_t comm;
    // Memory pool
    std::shared_ptr<MemoryPool> memory_pool;

    DeviceResource(infiniDevice_t device_,
                   int dev_id_,
                   infinicclComm_t comm_);

    ~DeviceResource();
};

DeviceResource::DeviceResource(
    infiniDevice_t device_,
    int dev_id_,
    infinicclComm_t comm_) : device(device_), device_id(dev_id_), comm(comm_) {
    RUN_INFINI(infinirtSetDevice(device, device_id));
    infiniopCreateHandle(&handle);
    infinirtStreamCreate(&stream);

    memory_pool = std::make_shared<MemoryPool>(128 * 1024 * 1024);

    RUN_INFINI(infinirtDeviceSynchronize());
}

DeviceResource::~DeviceResource() {
    infinirtDeviceSynchronize();

    infiniopDestroyHandle(handle);
    handle = nullptr;
    infinirtStreamDestroy(stream);
    stream = nullptr;
    infinicclCommDestroy(comm);
    comm = nullptr;
}

QwenHybridLayer::QwenHybridLayer(const QwenHybridMeta *meta, size_t layer, int rank, int nranks, infinicore::weights::Loader &weights_loader) {
    input_norm = infinicore::nn::module::RMSNorm::init(meta->d, meta->dt_logits, meta->epsilon);
    input_norm->register_weights(weights_loader, "model.layers." + std::to_string(layer) + ".input_layernorm", rank);
    multi_head_attn = infinicore::nn::module::MultiHeadAttention::init(meta->dt_logits, meta->d, meta->nh, meta->nkvh, meta->dh, meta->dh, nranks, meta->has_qkv_bias, false);
    multi_head_attn->register_weights(weights_loader, "model.layers." + std::to_string(layer) + ".self_attn", rank);
    post_attn_norm = infinicore::nn::module::RMSNorm::init(meta->d, meta->dt_logits, meta->epsilon);
    post_attn_norm->register_weights(weights_loader, "model.layers." + std::to_string(layer) + ".post_attention_layernorm", rank);
    mlp = infinicore::nn::module::MLP::init(meta->d, meta->di, meta->dt_logits, nranks);
    mlp->register_weights(weights_loader, "model.layers." + std::to_string(layer) + ".mlp", rank);
}

inline std::shared_ptr<Tensor> getSinTable(QwenHybridMeta const *meta) {
    auto half_dh = meta->dh / 2;
    auto unit = dsize(meta->dt_logits);
    void *table = std::malloc(meta->dctx * half_dh * unit);

    for (size_t i = 0; i < meta->dctx; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            float _sin = std::sin(
                static_cast<float>(i) / std::pow(meta->theta, static_cast<float>(j) / half_dh));
            if (meta->dt_logits == INFINI_DTYPE_F16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_f16(_sin);
            } else if (meta->dt_logits == INFINI_DTYPE_BF16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_bf16(_sin);
            } else if (meta->dt_logits == INFINI_DTYPE_F32) {
                ((float *)table)[i * half_dh + j] = _sin;
            } else {
                std::cout << "unsupported data type" << std::endl;
                exit(1);
            }
        }
    }
    auto shape = std::vector<size_t>({meta->dctx, half_dh});
    auto tensor = Tensor::weight(table, meta->dt_logits, shape);
    std::free(table);
    return tensor;
}

inline std::shared_ptr<Tensor> getCosTable(QwenHybridMeta const *meta) {
    auto half_dh = meta->dh / 2;
    auto unit = dsize(meta->dt_logits);
    void *table = std::malloc(meta->dctx * half_dh * unit);

    for (size_t i = 0; i < meta->dctx; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            float _cos = std::cos(
                static_cast<float>(i) / std::pow(meta->theta, static_cast<float>(j) / half_dh));
            if (meta->dt_logits == INFINI_DTYPE_F16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_f16(_cos);
            } else if (meta->dt_logits == INFINI_DTYPE_BF16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_bf16(_cos);
            } else if (meta->dt_logits == INFINI_DTYPE_F32) {
                ((float *)table)[i * half_dh + j] = _cos;
            } else {
                std::cout << "unsupported data type" << std::endl;
                exit(1);
            }
        }
    }
    auto shape = std::vector<size_t>({meta->dctx, half_dh});
    auto tensor = Tensor::weight(table, meta->dt_logits, shape);
    std::free(table);
    return tensor;
}

QwenHybridDeviceModel::QwenHybridDeviceModel(const QwenHybridMeta *meta, int rank_, int nranks_, infinicore::weights::Loader &weights_loader)
    : rank(rank_), nranks(nranks_) {
    input_embedding = Tensor::weight(nullptr, meta->dt_logits, {meta->dvoc, meta->d});
    sin_table = getSinTable(meta);
    cos_table = getCosTable(meta);
    weights_loader.register_weight("model.embed_tokens.weight", input_embedding, rank);

    layers.reserve(meta->nlayer);
    for (size_t layer = 0; layer < meta->nlayer; layer++) {
        layers.emplace_back(std::make_shared<QwenHybridLayer>(meta, layer, rank, nranks, weights_loader));
    }
    output_norm = infinicore::nn::module::RMSNorm::init(meta->d, meta->dt_logits, meta->epsilon);
    output_norm->register_weights(weights_loader, "model.norm", rank);
    output_embedding = infinicore::nn::module::Linear::init(meta->d, meta->dvoc, meta->dt_logits);
    output_embedding->register_weights(weights_loader, "lm_head", rank);
}

void QwenHybridDeviceModel::infer(InferRequest *req, DeviceResource &rsrc) {
    auto ntok = req->ntok;
    auto nreq = req->nreq;
    auto d = input_embedding->shape()[1];
    auto dt_logits = input_embedding->dtype();
    auto stream = rsrc.stream;
    auto nlayer = layers.size();

    // Allocate buffers
    auto logits_in = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto logits_out = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);

    // Prepare inputs
    auto batch_pos_ids = std::vector<uint32_t>(ntok);
    size_t req_start = 0;
    for (uint32_t req_idx = 0; req_idx < nreq; req_idx++) {
        for (uint32_t i = 0; i < req->req_lens[req_idx]; i++) {
            batch_pos_ids[req_start + i] = req->req_pos[req_idx] + i;
        }
        req_start += req->req_lens[req_idx];
    }

    std::shared_ptr<Tensor> pos_ids_buf;
    if (rsrc.device == INFINI_DEVICE_CPU) {
        pos_ids_buf = Tensor::weight(batch_pos_ids.data(), INFINI_DTYPE_U32, {ntok});
    } else {
        pos_ids_buf = Tensor::buffer(INFINI_DTYPE_U32, {ntok}, rsrc.memory_pool);
        RUN_INFINI(infinirtMemcpyAsync(pos_ids_buf->data(), batch_pos_ids.data(), sizeof(uint32_t) * ntok,
                                       INFINIRT_MEMCPY_H2D, stream));
    }

    // Input embedding
    for (uint32_t i = 0; i < ntok; i++) {
        RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d),
                                       input_embedding->data(req->tokens[i] * d),
                                       dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
    }

    // Get attention parameters from the first layer
    auto nh = layers[0]->multi_head_attn->n_q_heads;
    auto nkvh = layers[0]->multi_head_attn->n_kv_heads;
    auto dh = layers[0]->multi_head_attn->head_dim;
    auto ngroup = nh / nkvh;

    // Allocate attention buffers
    auto qkv_buf = Tensor::buffer(dt_logits, {ntok, (nh + nkvh * 2) * dh}, rsrc.memory_pool);
    auto o_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, rsrc.memory_pool);

    // Get MLP intermediate dimension from the first layer
    auto di = layers[0]->mlp->gate->weight->shape()[0];

    // Allocate MLP buffers
    auto gate_up_buf = Tensor::buffer(dt_logits, {ntok, 2 * di}, rsrc.memory_pool);

    // Compute max sequence length for attention
    size_t max_seq_len = 0;
    size_t max_qk_size = 0;
    for (uint32_t req_idx = 0; req_idx < nreq; req_idx++) {
        size_t req_len = req->req_lens[req_idx];
        size_t past_len = req->req_pos[req_idx];
        max_seq_len = std::max(max_seq_len, req_len);
        max_qk_size = std::max(max_qk_size, req_len * (past_len + req_len));
    }

    // Allocate attention score and value buffers
    auto attn_score_buf = Tensor::buffer(dt_logits, {nh * max_qk_size}, rsrc.memory_pool);
    auto attn_val_buf = Tensor::buffer(dt_logits, {nh * max_seq_len * dh}, rsrc.memory_pool);

    // Prepare KV cache vectors for each layer
    std::vector<std::vector<std::shared_ptr<Tensor>>> k_caches_per_layer(nlayer);
    std::vector<std::vector<std::shared_ptr<Tensor>>> v_caches_per_layer(nlayer);

    for (uint32_t layer = 0; layer < nlayer; layer++) {
        k_caches_per_layer[layer].reserve(nreq);
        v_caches_per_layer[layer].reserve(nreq);

        for (uint32_t req_idx = 0; req_idx < nreq; req_idx++) {
            auto past_len = req->req_pos[req_idx];
            auto seq_len = req->req_lens[req_idx];

            // Extract the appropriate slices from KV cache
            auto k_cache = req->kv_caches[req_idx]->k[rank][layer]->slice(0, 0, past_len + seq_len);
            auto v_cache = req->kv_caches[req_idx]->v[rank][layer]->slice(0, 0, past_len + seq_len);

            k_caches_per_layer[layer].push_back(k_cache);
            v_caches_per_layer[layer].push_back(v_cache);
        }
    }

    std::cout << "nlayer: " << nlayer << std::endl;
    // Layer loop
    for (uint32_t layer = 0; layer < nlayer; layer++) {
        auto &layer_module = layers[layer];

        // 1. Attention
        // RMS norm
        layer_module->input_norm->forward(logits_out, logits_in);

        // Multi-head attention
        layer_module->multi_head_attn->forward(
            o_buf, logits_out, logits_in, // output, input, residual
            attn_score_buf, attn_val_buf,
            pos_ids_buf, sin_table, cos_table,                          // pos_ids, sin_table, cos_table
            k_caches_per_layer[layer],                                  // k_caches for this layer
            v_caches_per_layer[layer],                                  // v_caches for this layer
            std::vector<uint32_t>(req->req_lens, req->req_lens + nreq), // req_lens
            std::vector<uint32_t>(req->req_pos, req->req_pos + nreq),   // req_pos
            nreq);

        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }

        // 2. FFN
        // RMS norm
        layer_module->post_attn_norm->forward(logits_out, logits_in);

        // MLP
        layer_module->mlp->forward(
            logits_in, logits_out, logits_in, // output, input, residual
            gate_up_buf);

        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }
    }

    // Output processing
    if (rsrc.device_id == 0) {
        if (req->logits != nullptr) {
            // Forward pass: output logits
            output_norm->forward(logits_out, logits_in);
            auto dvoc = output_embedding->weight->shape()[0];
            auto last_logits_buf = Tensor::buffer(dt_logits, {ntok, dvoc}, rsrc.memory_pool);
            output_embedding->forward(last_logits_buf, logits_out);
            RUN_INFINI(infinirtStreamSynchronize(stream));
            RUN_INFINI(infinirtMemcpy(req->logits, last_logits_buf->data(),
                                      dsize(dt_logits) * ntok * dvoc, INFINIRT_MEMCPY_D2H));
        }
        if (req->output != nullptr) {
            // Inference: sample next tokens
            auto dvoc = output_embedding->weight->shape()[0];
            auto prob_buf = Tensor::buffer(dt_logits, {nreq, dvoc}, rsrc.memory_pool);
            auto result_buf = Tensor::buffer(INFINI_DTYPE_I64, {nreq}, rsrc.memory_pool);
            auto result_cpu = std::vector<int64_t>(nreq);

            // Get last tokens for each request
            size_t token_offset = 0;
            for (uint32_t req_idx = 0; req_idx < nreq; req_idx++) {
                auto seq_len = req->req_lens[req_idx];
                output_norm->forward(
                    logits_out->slice(0, token_offset + seq_len - 1, 1),
                    logits_in->slice(0, token_offset + seq_len - 1, 1));
                token_offset += seq_len;
            }

            output_embedding->forward(prob_buf, logits_out->slice(0, 0, nreq));

            // Sample (using a simple argmax for now)
            RUN_INFINI(infinirtStreamSynchronize(stream));

            // Copy probabilities to CPU and do argmax
            auto prob_cpu = std::vector<float>(nreq * dvoc);
            RUN_INFINI(infinirtMemcpy(prob_cpu.data(), prob_buf->data(),
                                      dsize(dt_logits) * nreq * dvoc, INFINIRT_MEMCPY_D2H));

            for (uint32_t req_idx = 0; req_idx < nreq; req_idx++) {
                float max_prob = -1.0f;
                uint32_t max_idx = 0;
                for (uint32_t i = 0; i < dvoc; i++) {
                    if (prob_cpu[req_idx * dvoc + i] > max_prob) {
                        max_prob = prob_cpu[req_idx * dvoc + i];
                        max_idx = i;
                    }
                }
                req->output[req_idx] = max_idx;
            }
        }
    }
}

void launchDevice(const QwenHybridMeta *meta, InferState *state, InferRequest *req,
                  infiniDevice_t device, int idev, int ndev, int dev_id, infinicclComm_t comm, infinicore::weights::Loader *weights_loader) {
    // Create Device Resource
    auto rsrc = DeviceResource(device, dev_id, comm);

    CacheManager cache_manager(100);
    InferenceContext ctx(rsrc.handle, rsrc.memory_pool, &cache_manager, rsrc.stream);

    // Set the inference context for this thread
    setInferenceContext(&ctx);

    QwenHybridDeviceModel *model = new QwenHybridDeviceModel(meta, idev, ndev, *weights_loader);

    // Infer Loop
    while (true) {
        std::unique_lock<std::mutex> lock(state->mtx);
        state->cv_start.wait(lock, [&] { return state->proceed || state->exit_flag; });
        // quit if exit_flag is set
        if (state->exit_flag) {
            break;
        }

        model->infer(req, rsrc);

        state->proceed = false;
        lock.unlock();
        state->cv_done.notify_one();
    }

    // Clean-Up
    delete model;
    setInferenceContext(nullptr); // Clear the context when done
}

QwenHybridModel::QwenHybridModel(const QwenHybridMeta *meta_, infiniDevice_t device_, const std::vector<int> &dev_ids_)
    : meta(*meta_), device(device_), dev_ids(dev_ids_), weights_loader(device_, dev_ids_) {
    int ndev = int(dev_ids.size());
    states = std::vector<InferState>(ndev);
    threads.resize(ndev);

    auto comms = std::vector<infinicclComm_t>(ndev, nullptr);
    if (ndev > 1) {
        RUN_INFINI(infinicclCommInitAll(device, comms.data(), ndev, dev_ids.data()));
    }

    for (int i = 0; i < ndev; i++) {
        threads[i] = std::thread(launchDevice, &meta, &(states[i]), &req, device, i, ndev, dev_ids[i], comms[i], &weights_loader);
    }
}

__C struct QwenHybridModel *
createQwenHybridModel(const QwenHybridMeta *meta,
                      infiniDevice_t device,
                      int ndev,
                      const int *dev_ids) {
    std::vector<int> dev_id_vec(dev_ids, dev_ids + ndev);
    return new QwenHybridModel(meta, device, dev_id_vec);
}

__C struct ModelWeights *getQwenHybridModelWeights(QwenHybridModel *model) {
    RUN_INFINI(infinirtDeviceSynchronize());
    return (struct ModelWeights *)(&model->weights_loader);
}

__C void
destroyQwenHybridModel(struct QwenHybridModel *model) {
    auto ndev = model->dev_ids.size();

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

__C void
inferBatchQwenHybrid(struct QwenHybridModel *model,
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

__C void
forwardBatchQwenHybrid(struct QwenHybridModel *model,
                       const uint32_t *tokens, uint32_t ntok,
                       const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                       struct KVCache **kv_caches,
                       void *logits) {}
