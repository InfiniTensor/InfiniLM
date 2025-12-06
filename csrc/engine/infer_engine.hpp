#pragma once

#include "distributed/distributed.hpp"
#include "infinicore/tensor.hpp"
#include "rank_worker.hpp"

#include <any>
#include <vector>

namespace infinilm::engine {

class InferEngine {
public:
    InferEngine(
        const std::any &config,
        const distributed::DistConfig &distributed_config = distributed::DistConfig(),
        infinicore::Device::Type device_type = infinicore::context::getDevice().getType());

    // Load a parameter to all workers (each can extract its shard inside RankWorker)
    void load_param(const std::string &name, const infinicore::Tensor &param);

    // return the parameters (i.e. weights and biases).
    std::unordered_map<std::string, infinicore::nn::Parameter> state_dict();

    // Run a single forward pass on all workers and return the outputs from all ranks
    infinicore::Tensor generate(const infinicore::Tensor &input_ids,
                                const infinicore::Tensor &position_ids);

    // Reset the internal cache in all workers (clears state between generations)
    void reset_cache(bool full_reset = true);

    ~InferEngine();

    const distributed::DistConfig &get_dist_config() const;

protected:
    std::vector<std::unique_ptr<RankWorker>> workers_;
    distributed::CommunicationGroup communication_group_;
    std::any model_config_;
};

} // namespace infinilm::engine
