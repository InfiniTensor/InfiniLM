#ifndef INFINI_CCL_BACKENDS_CCL_NCCL_TYPE_MAP_H_
#define INFINI_CCL_BACKENDS_CCL_NCCL_TYPE_MAP_H_

#include <string>

#include "backends/ccl/common/api.h"
#include "backends/ccl/nccl/api.h"
#include "comm_impl.h"
#include "data_type_impl.h"
#include "logging.h"

namespace infini::ccl {

template <Device::Type device>
struct NcclDataTypeMap {
  static constexpr ConstexprMap<DataType, ncclDataType_t, 12> kMap{{{
      {DataType::kInt8, ncclInt8},
      {DataType::kInt16, ncclNumTypes},
      {DataType::kInt32, ncclInt32},
      {DataType::kInt64, ncclInt64},
      {DataType::kUInt8, ncclUint8},
      {DataType::kUInt16, ncclNumTypes},
      {DataType::kUInt32, ncclUint32},
      {DataType::kUInt64, ncclUint64},
      {DataType::kFloat32, ncclFloat32},
      {DataType::kFloat64, ncclFloat64},
      {DataType::kFloat16, ncclFloat16},
      {DataType::kBFloat16, NcclDataTypeTraits<device>::kBFloat16},
  }}};
};

static const ConstexprMap<ReductionOpType, ncclRedOp_t, 5> kNcclOpMap{{{
    {ReductionOpType::kSum, ncclSum},
    {ReductionOpType::kProd, ncclProd},
    {ReductionOpType::kMax, ncclMax},
    {ReductionOpType::kMin, ncclMin},
    {ReductionOpType::kAvg, ncclAvg},
}}};

template <Device::Type device>
inline ncclDataType_t DataTypeToNcclType(DataType dtype) {
  auto nccl_dtype = NcclDataTypeMap<device>::kMap.at(dtype);

  if (nccl_dtype == ncclNumTypes) {
    // This means the requested data type is not supported by NCCL.
    // TODO(lzm): change to use `glog`.
    LOG(("DataType '" + std::string(kDataTypeToDesc.at(dtype)) +
         "' is not supported by the NCCL backend")
            .c_str());
  }

  return nccl_dtype;
}

inline ncclRedOp_t RedOpToNcclOp(ReductionOpType red_op) {
  return kNcclOpMap.at(red_op);
}

template <Device::Type device>
struct CclTypeMap<BackendType::kNccl, device> {
  using Api = CclApi<BackendType::kNccl, device>;

  static bool ToBackendDataType(DataType dtype,
                                typename Api::DataType *backend_dtype) {
    auto nccl_dtype = DataTypeToNcclType<device>(dtype);
    if (nccl_dtype == ncclNumTypes) {
      return false;
    }
    *backend_dtype = nccl_dtype;
    return true;
  }

  static bool ToBackendRedOp(ReductionOpType red_op,
                             typename Api::RedOp *backend_op) {
    *backend_op = RedOpToNcclOp(red_op);
    return true;
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_NCCL_TYPE_MAP_H_
