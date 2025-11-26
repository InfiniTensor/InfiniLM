#ifndef LLADAMOE_IMPL_H
#define LLADAMOE_IMPL_H

#include "infinicore_infer.h"
#include "../../../include/infinicore_infer/models/llada_moe.h"  // 先包含meta定义
#include "../../allocator.hpp"
#include "../../tensor.hpp"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

struct LLaDAMoEDeviceResource {

};

struct InferState {

};

struct InferRequest {

};

struct LLaDAMoEModel {
    LLaDAMoEMeta meta;
    infiniDevice_t device;
    std::vector<int> dev_ids;
    std::vector<LLaDAMoEDeviceResource> dev_resources; // # TODO: Lack of LLaDAMoEDeviceResource
    std::vector<InferState> states;
    std::vector<std::thread> threads;
    InferRequest req;
    // TODO
    LLaDAMoEModel(const LLaDAMoEMeta *, const JiugeWeights *, infiniDevice_t device, std::vector<int> device_ids);
};

#include "../../cache.hpp"
#endif