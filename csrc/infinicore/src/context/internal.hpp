#pragma once

#include "infinicore/device.hpp"
#include "infinicore/memory.hpp"

#include "infinicore/graph/graph.hpp"

namespace infinicore::context {
std::shared_ptr<Memory> reinstantiateBlob(std::shared_ptr<Memory> blob);
void retainGraphMemory(const std::shared_ptr<Memory> &memory);
}; // namespace infinicore::context
