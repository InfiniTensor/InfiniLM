#ifndef INFINI_OPS_ASCEND_DATA_TYPE__H_
#define INFINI_OPS_ASCEND_DATA_TYPE__H_

#include <cassert>

#include "acl/acl.h"
#include "data_type.h"
#include "native/ascend/device_.h"

namespace infini::ops::ascend {

inline aclDataType ToAclDtype(DataType dt) {
  switch (dt) {
    case DataType::kFloat16:
      return ACL_FLOAT16;
    case DataType::kBFloat16:
      return ACL_BF16;
    case DataType::kFloat32:
      return ACL_FLOAT;
    case DataType::kInt8:
      return ACL_INT8;
    case DataType::kInt16:
      return ACL_INT16;
    case DataType::kInt32:
      return ACL_INT32;
    case DataType::kInt64:
      return ACL_INT64;
    case DataType::kUInt8:
      return ACL_UINT8;
    case DataType::kUInt16:
      return ACL_UINT16;
    case DataType::kUInt32:
      return ACL_UINT32;
    case DataType::kUInt64:
      return ACL_UINT64;
    default:
      assert(false && "unsupported dtype for Ascend backend");
      return ACL_DT_UNDEFINED;
  }
}

// Returns true for integer (signed or unsigned) DataType values.
inline bool IsIntegerDtype(DataType dt) {
  switch (dt) {
    case DataType::kInt8:
    case DataType::kInt16:
    case DataType::kInt32:
    case DataType::kInt64:
    case DataType::kUInt8:
    case DataType::kUInt16:
    case DataType::kUInt32:
    case DataType::kUInt64:
      return true;
    default:
      return false;
  }
}

}  // namespace infini::ops::ascend

#endif
