#pragma once

#include "../../../engine/distributed/communication_group.hpp"
#include "../ep/ep_config.hpp"
#include "base_dispatcher.hpp"

#include <cstddef>
#include <memory>

namespace infinilm::layers::moe {

std::shared_ptr<BaseDispatcher> make_dispatcher(const EPConfig &ep_config,
                                                size_t num_experts,
                                                const engine::distributed::RankInfo &rank_info);

} // namespace infinilm::layers::moe
