#ifndef INFINI_CCL_BACKENDS_CCL_CNCL_TYPE_MAP_H_
#define INFINI_CCL_BACKENDS_CCL_CNCL_TYPE_MAP_H_

#include <cncl.h>

#include <string>

#include "backends/ccl/common/api.h"
#include "comm_impl.h"
#include "data_type_impl.h"
#include "logging.h"

namespace infini::ccl {

template <Device::Type device>
struct CnclDataTypeMap {
  static constexpr ConstexprMap<DataType, cnclDataType_t, 12> kMap{{{
      {DataType::kInt8, cnclInt8},
      {DataType::kInt16, cnclInt16},
      {DataType::kInt32, cnclInt32},
      {DataType::kInt64, cnclInt64},
      {DataType::kUInt8, cnclUint8},
      {DataType::kUInt16, cnclUint16},
      {DataType::kUInt32, cnclUint32},
      {DataType::kUInt64, cnclUint64},
      {DataType::kFloat32, cnclFloat32},
      {DataType::kFloat64, cnclInvalid},
      {DataType::kFloat16, cnclFloat16},
      {DataType::kBFloat16, cnclBfloat16},
  }}};
};

template <Device::Type device>
inline cnclDataType_t DataTypeToCnclType(DataType dtype) {
  auto cncl_dtype = CnclDataTypeMap<device>::kMap.at(dtype);

  if (cncl_dtype == cnclInvalid) {
    LOG(("DataType '" + std::string(kDataTypeToDesc.at(dtype)) +
         "' is not supported by the CNCL backend")
            .c_str());
  }

  return cncl_dtype;
}

static const ConstexprMap<ReductionOpType, cnclReduceOp_t, 4> kCnclOpMap{{{
    {ReductionOpType::kSum, cnclSum},
    {ReductionOpType::kProd, cnclProd},
    {ReductionOpType::kMax, cnclMax},
    {ReductionOpType::kMin, cnclMin},
}}};

inline cnclReduceOp_t RedOpToCnclOp(ReductionOpType red_op) {
  return kCnclOpMap.at(red_op);
}

template <>
struct CclTypeMap<BackendType::kCncl, Device::Type::kCambricon> {
  using Api = CclApi<BackendType::kCncl, Device::Type::kCambricon>;

  static bool ToBackendDataType(DataType dtype,
                                typename Api::DataType* backend_dtype) {
    auto cncl_dtype = DataTypeToCnclType<Device::Type::kCambricon>(dtype);
    if (cncl_dtype == cnclInvalid) {
      return false;
    }
    *backend_dtype = cncl_dtype;
    return true;
  }

  static bool ToBackendRedOp(ReductionOpType red_op,
                             typename Api::RedOp* backend_op) {
    if (red_op == ReductionOpType::kAvg) {
      return false;
    }
    *backend_op = RedOpToCnclOp(red_op);
    return true;
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_CNCL_TYPE_MAP_H_
