#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../inference_context.hpp"
#include "infinicore_infer.h"
#include "llada_moe.h"

#include <random>
#include <thread>
#include <vector>
void createDeviceResource(){

}

void releaseDeviceResource(){

}

void inferDeviceBatch(){

}

__C void
inferBatchLLaDAMoE(){

}

__C void
forwardBatchLLaDAMoE(){

}

void launchDevice(){

}


// TODO: not void just for tmp
void LLaDAMoEModel()  {

}


__C struct LLaDAMoEModel * createLLaDAMoEModel(const LLaDAMoEMeta *meta, const ModelWeights *weights_) {
    auto weights = (LLaDAWeight *)(weights_);
    dev_resources = std::vector<DeviceResource>(1);
    threads.resize(1);
    launchDevice(meta, weights->device_weights()[i], &dev_resources[i], std::ref(states[i]), std::ref(req), device, i, ndev, dev_ids[i], comms[i]);
}


__C void destroyLaDAMoEModel(){

}