#include "weights_loader.hpp"
#include "infinicore_infer/weights_loader.h"

#include "../utils.hpp"

#include <infinirt.h>
#include <numeric>

namespace infinicore::weights {

void Weight::load(const void *host_data, infinirtStream_t stream) {
    if (_dist_type == DistributionType::FULL) {
        _tensor->load(host_data, stream);
    } else if (_dist_type == DistributionType::ROW || _tensor->ndim() == 1) { // 1D column-distributed is same as row-distributed
        _tensor->load((const char *)host_data + _rank * _tensor->numel() * dsize(_tensor->dtype()), stream);
    } else if (_dist_type == DistributionType::COLUMN && _tensor->ndim() > 1) { // _dist_type == DistributionType::COLUMN
        void *rearranged_ptr;
        RUN_INFINI(infinirtMallocHost(&rearranged_ptr, _tensor->numel() * dsize(_tensor->dtype())));
        /// TODO: here assume weight is stored as W^T, and has been permuted {1, 0}
        size_t row_size = _tensor->shape()[_tensor->ndim() - 2] * dsize(_tensor->dtype());
        size_t host_offset = _rank * row_size;
        size_t host_row_size = _nrank * row_size;
        size_t rows = _tensor->shape()[_tensor->ndim() - 1];
        for (size_t row = 0; row < rows; row++) {
            memcpy((char *)rearranged_ptr + row * row_size,
                   (char *)host_data + host_offset + row * host_row_size,
                   row_size);
        }
        _tensor->load(rearranged_ptr, stream);
        RUN_INFINI(infinirtFreeHost(rearranged_ptr));
    } else {
        std::cerr << "Unsupported distribution type: " << _dist_type << std::endl;
        std::abort();
    }
};

Loader::Loader(infiniDevice_t dev, const std::vector<int> &dev_ids) : _device(dev), _dev_ids(dev_ids) {
    _streams.resize(_dev_ids.size());
    _weights_maps.resize(_dev_ids.size());
    for (int rank = 0; rank < int(_dev_ids.size()); rank++) {
        RUN_INFINI(infinirtSetDevice(_device, _dev_ids[rank]));
        _weights_maps[rank] = std::unordered_map<std::string, std::shared_ptr<Weight>>();
        RUN_INFINI(infinirtStreamCreate(&_streams[rank]));
    }
}
void Loader::register_weight(const std::string &name, std::shared_ptr<Tensor> tensor, int rank, DistributionType dist_type) {
    _weights_maps[rank][name] = std::make_shared<Weight>(tensor, rank, _dev_ids.size(), dist_type);
}
void Loader::load(const std::string &name, const void *host_data) {
    for (int rank = 0; rank < int(_dev_ids.size()); rank++) {
        RUN_INFINI(infinirtSetDevice(_device, _dev_ids[rank]));
        auto it = _weights_maps[rank].find(name);
        if (it == _weights_maps[rank].end()) {
            std::cerr << "Weight " << name << " not found in rank " << rank << std::endl;
            std::abort();
        }

        _weights_maps[rank][name]->load(host_data, _streams[rank]);
    }
    for (int rank = int(_dev_ids.size() - 1); rank >= 0; rank--) {
        RUN_INFINI(infinirtSetDevice(_device, _dev_ids[rank]));
        RUN_INFINI(infinirtStreamSynchronize(_streams[rank]));
    }
}

void Loader::finalize() {
    int dev_id;
    RUN_INFINI(infinirtGetDevice(nullptr, &dev_id));
    for (int rank = 0; rank < int(_dev_ids.size()); rank++) {
        RUN_INFINI(infinirtSetDevice(_device, _dev_ids[rank]));
        RUN_INFINI(infinirtStreamSynchronize(_streams[rank]));
        RUN_INFINI(infinirtStreamDestroy(_streams[rank]));
    }
    RUN_INFINI(infinirtSetDevice(_device, dev_id));
}
std::shared_ptr<Tensor> Loader::get(const std::string &name, int rank) {
    return _weights_maps[rank][name]->tensor();
}

} // namespace infinicore::weights

__C void
loadModelWeight(struct ModelWeights *weights_, const char *name, void *data) {
    std::string name_str(name);
    // std::cout << "Loading weight: " << name_str << std::endl;
    auto weights = reinterpret_cast<infinicore::weights::Loader *>(weights_);
    weights->load(name_str, data);
}
