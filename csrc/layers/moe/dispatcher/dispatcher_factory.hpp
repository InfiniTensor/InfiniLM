#pragma once

#include "../ep/ep_config.hpp"
#include "base_dispatcher.hpp"

#include <cstddef>
#include <memory>

namespace infinilm::layers::moe {

std::shared_ptr<BaseDispatcher> make_dispatcher(const EPConfig &ep_config,
                                                size_t num_experts);

} // namespace infinilm::layers::moe
