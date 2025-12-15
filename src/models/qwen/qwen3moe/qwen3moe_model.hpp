#ifndef _QWEN3MOE_MODEL_HPP_
#define _QWEN3MOE_MODEL_HPP_
#include "../qwen_device_resource.hpp"
#include "qwen3moe_weight.hpp"

namespace Qwen3MoE {

struct Model {
    Meta meta;
    infiniDevice_t device;
    std::vector<int> dev_ids;
    std::vector<DeviceResource<WeightsTensor>> dev_resources;
    std::vector<InferState> states;
    std::vector<std::thread> threads;
    InferRequest req;

    Model(const Meta *, const Weights *, infiniDevice_t device, std::vector<int> device_ids);
};
}; // namespace Qwen3MoE
#endif
