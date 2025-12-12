#ifndef _QWEN_MODEL_H_
#define _QWEN_MODEL_H_

// #include "infinicore_infer/models/qwen3.h"


#include "qwen3moe/qwen3moe_model.hpp"

#include <stdexcept>
#include <vector>

/**
 * @brief Create a model instance
 * @tparam Model Model type
 * @tparam Meta Metadata type
 * @tparam Weights Weights type
 * @param meta Pointer to model config metadata (must not be nullptr)
 * @param weights Pointer to model weights, it is a cpu pointer, it will be copied to gpu memory
 * @param device Device type
 * @param ndev Number of devices (must be positive)
 * @param dev_ids Array of device IDs (must not be nullptr if ndev > 0)
 * @return Pointer to the created model instance
 * @throws std::invalid_argument if any input parameter is invalid
 * @throws std::bad_alloc if memory allocation fails
 */
template <typename Model, typename Meta, typename Weights>
Model *createModel(const Meta *meta,
                   const Weights *weights,
                   infiniDevice_t device,
                   int ndev,
                   const int *dev_ids) {
    // Input validation
    if (meta == nullptr) {
        throw std::invalid_argument("createModel: meta cannot be nullptr");
    }
    if (weights == nullptr) {
        throw std::invalid_argument("createModel: weights cannot be nullptr");
    }
    if (ndev <= 0) {
        throw std::invalid_argument("createModel: ndev must be positive");
    }
    if (dev_ids == nullptr) {
        throw std::invalid_argument("createModel: dev_ids cannot be nullptr");
    }

    // Copy device IDs
    std::vector<int> device_ids(dev_ids, dev_ids + ndev);

    // Create model instance
    Model *model = new Model(meta, weights, device, device_ids);
    if (model == nullptr) {
        throw std::bad_alloc();
    }

    return model;
}

/**
 * @brief Destroy a model instance and clean up resources
 * @tparam Model Model type
 * @param model Pointer to the model instance to destroy (must not be nullptr)
 * @throws std::invalid_argument if model is nullptr
 */
template <typename Model>
void destroyModel(Model *model) {
    if (model == nullptr) {
        throw std::invalid_argument("destroyModel: model cannot be nullptr");
    }

    auto ndev = model->dev_resources.size();

    // Signal all device threads to exit
    for (size_t idev = 0; idev < ndev; idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].exit_flag = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }

    // Wait for all device threads to finish
    for (size_t idev = 0; idev < ndev; idev++) {
        model->threads[idev].join();
    }

    // Delete the model instance
    delete model;
}

#endif
