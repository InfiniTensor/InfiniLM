#ifndef WEIGHTS_LOADER_HPP
#define WEIGHTS_LOADER_HPP

#include "../tensor.hpp"

#include <unordered_map>
#include <vector>

namespace infinicore {

namespace weights {
enum DistributionType {
    FULL,
    ROW,
    COLUMN
};
class Weight {
private:
    Tensor *_tensor;
    int _rank;
    int _nrank;
    DistributionType _dist_type;

public:
    Weight(std::shared_ptr<Tensor> tensor,
           int rank = 0,
           int nrank = 1,
           DistributionType dist_type = DistributionType::FULL)
        : _tensor(tensor.get()), _rank(rank), _nrank(nrank), _dist_type(dist_type) {}
    Tensor *tensor() const { return _tensor; }
    int rank() const { return _rank; }
    int nrank() const { return _nrank; }
    void load(const void *host_data, infinirtStream_t stream = nullptr);
};

class Loader {
protected:
    std::vector<std::unordered_map<std::string, std::shared_ptr<Weight>>> _weights_maps;
    infiniDevice_t _device;
    std::vector<int> _dev_ids;
    std::vector<infinirtStream_t> _streams;

public:
    Loader(infiniDevice_t, const std::vector<int> &dev_ids);

    /// @brief register a tensor to the loader
    /// @param name name (aka key) of the tensor
    /// @param tensor
    /// @param rank the rank of the weight tensor (default 0)
    /// @param dist_type either FULL, or distributed by ROW or COLUMN (default FULL)
    void register_weight(const std::string &name, std::shared_ptr<Tensor> tensor, int rank = 0, DistributionType dist_type = DistributionType::FULL);
    void load(const std::string &name, const void *host_data);
    void finalize();
    Tensor *get(const std::string &name, int rank = 0);
    const std::vector<int> &devIds() const { return _dev_ids; }
    infiniDevice_t device() const { return _device; }
};
} // namespace weights
} // namespace infinicore

#endif // WEIGHTS_LOADER_HPP
