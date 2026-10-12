#pragma once

#include <infini/rt.h>

#include <cstddef>
#include <string>

namespace infinilm::runtime {

using DataType = infini::rt::DataType;

std::string toString(const DataType &dtype);
std::size_t dsize(const DataType &dtype);

} // namespace infinilm::runtime
