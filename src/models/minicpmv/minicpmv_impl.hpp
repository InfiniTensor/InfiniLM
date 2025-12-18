#ifndef MINICPMV_IMPL_HPP
#define MINICPMV_IMPL_HPP

#include "infinicore_infer/models/minicpmv.h"

#include <vector>

struct MiniCPMVModel {
    MiniCPMVMeta meta;
    const MiniCPMVWeights *weights;
    infiniDevice_t device;
    std::vector<int> dev_ids;
    infiniopHandle_t op_handle = nullptr;
    infinirtStream_t stream = nullptr;

    MiniCPMVModel(const MiniCPMVMeta *meta_,
                  const MiniCPMVWeights *weights_,
                  infiniDevice_t device_,
                  std::vector<int> device_ids)
        : meta(*meta_), weights(weights_), device(device_), dev_ids(std::move(device_ids)) {}
};

#endif
