#include "dispatcher_factory.hpp"

#include "../ep/allgather_reduce_scatter_dispatcher.hpp"
#include "../ep/deepep_dispatcher.hpp"
#include "../ep/local_allreduce_dispatcher.hpp"
#include "standard_dispatcher.hpp"

#include <stdexcept>

namespace infinilm::layers::moe {

std::shared_ptr<BaseDispatcher> make_dispatcher(const EPConfig &ep_config,
                                                size_t num_experts) {
    switch (ep_config.backend) {
    case EPBackend::Disabled:
        return std::make_shared<StandardDispatcher>();
    case EPBackend::AllGatherReduceScatter:
        return std::make_shared<AllGatherReduceScatterDispatcher>(ep_config, num_experts);
    case EPBackend::LocalAllReduce:
        return std::make_shared<LocalAllReduceDispatcher>(ep_config, num_experts);
    case EPBackend::DeepEP:
        return std::make_shared<DeepEPDispatcher>(ep_config);
    default:
        throw std::runtime_error("Unsupported MoE EP backend");
    }
}

} // namespace infinilm::layers::moe
