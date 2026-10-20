#include "data_type.h"

#include <iomanip>
#include <iostream>
#include <vector>

static void PrintDataTypeInfo(const infini::ops::DataType& dtype) {}

int main() {
  using namespace infini::ops;

  static const std::vector<DataType> kDataTypes{
      DataType::kInt8,     DataType::kInt16,   DataType::kInt32,
      DataType::kInt64,    DataType::kUInt8,   DataType::kUInt16,
      DataType::kUInt32,   DataType::kUInt64,  DataType::kFloat16,
      DataType::kBFloat16, DataType::kFloat32, DataType::kFloat64};

  std::cout << std::left << std::setw(10) << "Name" << std::left
            << std::setw(10) << "Element Size\n";

  for (const auto& dtype : kDataTypes) {
    std::cout << std::left << std::setw(10) << kDataTypeToDesc.at(dtype)
              << std::left << std::setw(10) << kDataTypeToSize.at(dtype)
              << '\n';
  }

  return 0;
}
