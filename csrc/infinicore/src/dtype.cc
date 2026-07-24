#include <infinicore.hpp>

namespace infinicore {

std::string toString(const DataType &dtype) {
    return std::string{infini::rt::kDataTypeToDesc.at(dtype)};
}

std::size_t dsize(const DataType &dtype) {
    return infini::rt::kDataTypeToSize.at(dtype);
}

} // namespace infinicore
