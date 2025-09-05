#include "qwen_hybrid.hpp"

#include "../../context/inference_context.hpp"
#include "../../tensor.hpp"
#include "../../utils.hpp"

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
    input_norm = infinicore::nn::module::RMSNorm::init(meta->d, meta->dt_logits);
    input_norm->register_weights(weights_loader, "model.layers." + std::to_string(layer) + ".input_layernorm", rank);
    multi_head_attn = infinicore::nn::module::MultiHeadAttention::init(meta->dt_logits, meta->d, meta->nh, meta->nkvh, meta->dh, meta->dh, nranks, meta->has_qkv_bias, false);
    multi_head_attn->register_weights(weights_loader, "model.layers." + std::to_string(layer) + ".multi_head_attn", rank);
    post_attn_norm = infinicore::nn::module::RMSNorm::init(meta->d, meta->dt_logits);
    post_attn_norm->register_weights(weights_loader, "model.layers." + std::to_string(layer) + ".post_attention_layernorm", rank);
    mlp = infinicore::nn::module::MLP::init(meta->d, meta->di, meta->dt_logits, nranks);
    mlp->register_weights(weights_loader, "model.layers." + std::to_string(layer) + ".mlp", rank);
}

QwenHybridDeviceModel::QwenHybridDeviceModel(const QwenHybridMeta *meta, int rank, int nranks, infinicore::weights::Loader &weights_loader) {
    input_embedding = Tensor::weight(nullptr, meta->dt_logits, {meta->dvoc, meta->d});
    weights_loader.register_weight("model.embed_tokens.weight", input_embedding, rank);

    layers.reserve(meta->nlayer);
    for (size_t layer = 0; layer < meta->nlayer; layer++) {
        layers[layer] = std::make_shared<QwenHybridLayer>(meta, layer, rank, nranks, weights_loader);
    }
    output_norm = infinicore::nn::module::RMSNorm::init(meta->d, meta->dt_logits);
    output_norm->register_weights(weights_loader, "model.norm.weight", rank);
    output_embedding = infinicore::nn::module::Linear::init(meta->d, meta->dvoc, meta->dt_logits);
    output_embedding->register_weights(weights_loader, "lm_head.weight", rank);
}

void QwenHybridDeviceModel::infer(InferRequest *req, DeviceResource &rsrc) {
    // TODO
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
