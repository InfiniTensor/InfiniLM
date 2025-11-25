#ifndef JIUGE_IMPL_H
#define JIUGE_IMPL_H

#include "infinicore_infer.h"

#include "../../allocator.hpp"
#include "../../tensor.hpp"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

struct LLaDAMoEDeviceResource {

}

struct InferState {

}

struct InferRequest {

}

struct LLaDAMoEModel {
    LLaDAMeta meta;
    infiniDevice_t device;
    std::vector<int> dev_ids;
    std::vector<LLaDAMoEDeviceResource> dev_resources; // # TODO: Lack of LLaDAMoEDeviceResource
    std::vector<InferState> states;
    std::vector<std::thread> threads;
    InferRequest req;

    // TODO
    LLaDAMoEModel(const LLaDAMoEMeta *, const JiugeWeights *, infiniDevice_t device, std::vector<int> device_ids);
}