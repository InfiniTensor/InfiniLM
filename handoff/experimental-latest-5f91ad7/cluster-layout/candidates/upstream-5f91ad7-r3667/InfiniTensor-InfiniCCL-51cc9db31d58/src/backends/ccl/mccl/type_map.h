#ifndef INFINI_CCL_BACKENDS_CCL_MCCL_TYPE_MAP_H_
#define INFINI_CCL_BACKENDS_CCL_MCCL_TYPE_MAP_H_

#include <mccl.h>

#include <string>

#include "backends/ccl/common/api.h"
#include "backends/ccl/mccl/api.h"
#include "comm_impl.h"
#include "data_type_impl.h"
#include "logging.h"

namespace infini::ccl {

template <Device::Type device>
struct McclDataTypeMap {
  static constexpr ConstexprMap<DataType, mcclDataType_t, 12> kMap{{{
      {DataType::kInt8, mcclInt8},
      {DataType::kInt16, mcclNumTypes},
      {DataType::kInt32, mcclInt32},
      {DataType::kInt64, mcclInt64},
      {DataType::kUInt8, mcclUint8},
      {DataType::kUInt16, mcclNumTypes},
      {DataType::kUInt32, mcclUint32},
      {DataType::kUInt64, mcclUint64},
      {DataType::kFloat32, mcclFloat32},
      {DataType::kFloat64, mcclFloat64},
      {DataType::kFloat16, mcclFloat16},
      {DataType::kBFloat16, McclDataTypeTraits<device>::kBFloat16},
  }}};
};

static const ConstexprMap<ReductionOpType, mcclRedOp_t, 5> kMcclOpMap{{{
    {ReductionOpType::kSum, mcclSum},
    {ReductionOpType::kProd, mcclProd},
    {ReductionOpType::kMax, mcclMax},
    {ReductionOpType::kMin, mcclMin},
    {ReductionOpType::kAvg, mcclAvg},
}}};

template <Device::Type device>
inline mcclDataType_t DataTypeToMcclType(DataType dtype) {
  auto mccl_dtype = McclDataTypeMap<device>::kMap.at(dtype);

  if (mccl_dtype == mcclNumTypes) {
    LOG(("DataType '" + std::string(kDataTypeToDesc.at(dtype)) +
         "' is not supported by the MCCL backend")
            .c_str());
  }

  return mccl_dtype;
}

inline mcclRedOp_t RedOpToMcclOp(ReductionOpType red_op) {
  return kMcclOpMap.at(red_op);
}

template <Device::Type device>
struct CclTypeMap<BackendType::kMccl, device> {
  using Api = CclApi<BackendType::kMccl, device>;

  static bool ToBackendDataType(DataType dtype,
                                typename Api::DataType *backend_dtype) {
    auto mccl_dtype = DataTypeToMcclType<device>(dtype);
    if (mccl_dtype == mcclNumTypes) {
      return false;
    }
    *backend_dtype = mccl_dtype;
    return true;
  }

  static bool ToBackendRedOp(ReductionOpType red_op,
                             typename Api::RedOp *backend_op) {
    *backend_op = RedOpToMcclOp(red_op);
    return true;
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_MCCL_TYPE_MAP_H_
