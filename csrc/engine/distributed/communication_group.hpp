#pragma once

#include "dist_config.hpp"

#include <infiniccl.h>
#include <infinicore/context/context.hpp>

#include <vector>

namespace infinilm::engine::distributed {

// Communicator each rank will hold
struct RankCommunicator {
    RankInfo info;
    infinicclComm_t comm;
};

// The communication group managed by model infer engine
class CommunicationGroup {
public:
    explicit CommunicationGroup(const DistConfig &dist_config);

    const DistConfig &getDistConfig() const;

    RankCommunicator getRankCommunicator(int rank) const;

    ~CommunicationGroup();

protected:
    DistConfig dist_config_;
    std::vector<infinicclComm_t> communicators_;
};

} // namespace infinilm::engine::distributed
