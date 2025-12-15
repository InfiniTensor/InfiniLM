#ifndef _QWEN_DEVICE_RESOURCE_
#define _QWEN_DEVICE_RESOURCE_

#include "../inference_context.hpp"
#include "../jiuge/jiuge_impl.hpp"
#include <condition_variable>
#include <cstdio>
#include <memory>
#include <mutex>
#include <stdexcept>

template <typename WeightsTensor>
struct DeviceResource {
    // Device
    infiniDevice_t device;
    int device_id;
    infiniopHandle_t handle;
    // Streams
    infinirtStream_t stream;
    // Communicator
    infinicclComm_t comm;
    std::shared_ptr<MemoryPool> memory_pool;

    // Pointer to the GPU parameters of the model
    std::unique_ptr<const WeightsTensor> weights_tensor_ptr{nullptr};
};

/**
 * @brief Create and initialize device resource for model inference
 * @tparam WeightsTensor Type of weights tensor
 * @tparam Meta Type of model metadata
 * @tparam Weights Type of model weights
 * @param rsrc Pointer to DeviceResource to initialize (must not be nullptr)
 * @param meta Pointer to model metadata (must not be nullptr)
 * @param weights Pointer to model weights (must not be nullptr)
 * @param device Device type
 * @param idev Device index
 * @param ndev Total number of devices
 * @param dev_id Physical device ID
 * @param comm Communication handle for multi-device
 * @throws std::invalid_argument if any required pointer is nullptr
 * @throws std::runtime_error if resource creation fails
 */
template <typename WeightsTensor, typename Meta, typename Weights>
void createDeviceResource(DeviceResource<WeightsTensor> *rsrc, const Meta *meta,
                          const Weights *weights,
                          infiniDevice_t device, int idev,
                          int ndev, int dev_id,
                          infinicclComm_t comm) {
    // Input validation
    if (rsrc == nullptr) {
        throw std::invalid_argument("createDeviceResource: rsrc cannot be nullptr");
    }
    if (meta == nullptr) {
        throw std::invalid_argument("createDeviceResource: meta cannot be nullptr");
    }
    if (weights == nullptr) {
        throw std::invalid_argument("createDeviceResource: weights cannot be nullptr");
    }
    if (ndev <= 0) {
        throw std::invalid_argument("createDeviceResource: ndev must be positive");
    }
    if (idev < 0 || idev >= ndev) {
        throw std::invalid_argument("createDeviceResource: idev out of range");
    }

    RUN_INFINI(infinirtSetDevice(device, dev_id));
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle);
    infinirtStream_t stream;
    infinirtStreamCreate(&stream);

    auto memory_pool = std::make_shared<MemoryPool>(128 * 1024 * 1024);
    if (!memory_pool) {
        throw std::runtime_error("createDeviceResource: memory pool allocation failed");
    }

    // Use member-wise assignment instead of aggregate initialization to avoid stack smashing
    rsrc->device = device;
    rsrc->device_id = dev_id;
    rsrc->handle = handle;
    rsrc->stream = stream;
    rsrc->comm = comm;
    rsrc->memory_pool = memory_pool;
    rsrc->weights_tensor_ptr = std::make_unique<const WeightsTensor>(meta, weights, idev, ndev);

    RUN_INFINI(infinirtDeviceSynchronize());
}

/**
 * @brief Release device resource and clean up allocated resources
 * @tparam WeightsTensor Type of weights tensor
 * @param res DeviceResource reference to release
 * @note This function is safe to call multiple times or with partially initialized resources
 */
template <typename WeightsTensor>
void releaseDeviceResource(DeviceResource<WeightsTensor> &res) {
    infinirtDeviceSynchronize();

    // Release weights tensor (smart pointer will automatically free memory)
    res.weights_tensor_ptr.reset();

    // Release device handles
    if (res.handle != nullptr) {
        infiniopDestroyHandle(res.handle);
        res.handle = nullptr;
    }
    if (res.stream != nullptr) {
        infinirtStreamDestroy(res.stream);
        res.stream = nullptr;
    }
    if (res.comm != nullptr) {
        infinicclCommDestroy(res.comm);
        res.comm = nullptr;
    }
}

/**
 * @brief Launch device thread for model inference
 * @tparam WeightsTensor Type of weights tensor
 * @tparam Meta Type of model metadata
 * @tparam Weights Type of model weights
 * @param meta Model metadata reference, it is config of the model
 * @param weights Pointer to model weights, it is cpu pointer, it will be copied to gpu memory
 * @param rsrc Pointer to DeviceResource to initialize (must not be nullptr)
 * @param state Inference state for synchronization
 * @param req Inference request structure
 * @param device Device type
 * @param idev Device index
 * @param ndev Total number of devices
 * @param dev_id Physical device ID
 * @param comm Communication handle for multi-device
 * @param inferDeviceBatch Function pointer to device batch inference function (must not be nullptr)
 * @throws std::invalid_argument if any required pointer is nullptr
 */
template <typename WeightsTensor, typename Meta, typename Weights>
void launchDevice(const Meta &meta, const Weights *weights, DeviceResource<WeightsTensor> *rsrc, InferState &state, InferRequest &req,
                  infiniDevice_t device, int idev, int ndev, int dev_id, infinicclComm_t comm,
                  void (*inferDeviceBatch)(const Meta *, DeviceResource<WeightsTensor> &, uint32_t, uint32_t, const uint32_t *, uint32_t, const uint32_t *, uint32_t, const uint32_t *, struct KVCache **kv_caches, const float *, const uint32_t *, const float *, uint32_t *, void *)) {
    // Input validation
    if (rsrc == nullptr) {
        throw std::invalid_argument("launchDevice: rsrc cannot be nullptr");
    }
    if (weights == nullptr) {
        throw std::invalid_argument("launchDevice: weights cannot be nullptr");
    }
    if (inferDeviceBatch == nullptr) {
        throw std::invalid_argument("launchDevice: inferDeviceBatch cannot be nullptr");
    }

    // Create Device Resource
    createDeviceResource<WeightsTensor, Meta, Weights>(rsrc, &meta, weights, device, idev, ndev, dev_id, comm);

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

        inferDeviceBatch(&meta, *rsrc, idev, ndev, req.tokens, req.ntok,
                         req.req_lens, req.nreq, req.req_pos, req.kv_caches,
                         req.temperature, req.topk, req.topp, req.output, req.logits);

        state.proceed = false;
        lock.unlock();
        state.cv_done.notify_one();
    }

    // Clean-Up
    releaseDeviceResource<WeightsTensor>(*rsrc);
    setInferenceContext(nullptr); // Clear the context when done
}

/**
 * @brief Perform batch inference on the model
 */
template <typename Model>
void inferBatch(Model *model,
                const uint32_t *tokens, uint32_t ntok,
                const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                KVCache **kv_caches,
                const float *temperature, const uint32_t *topk, const float *topp,
                uint32_t *output) {
    if (model == nullptr) {
        throw std::invalid_argument("inferBatch: model cannot be nullptr");
    }

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

/**
 * @brief Perform forward pass (compute logits) for batch inference
 */
template <typename Model>
void forwardBatch(Model *model,
                  const uint32_t *tokens, uint32_t ntok,
                  const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                  KVCache **kv_caches,
                  void *logits) {
    if (model == nullptr) {
        throw std::invalid_argument("forwardBatch: model cannot be nullptr");
    }

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

#endif
