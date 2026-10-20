#ifndef INFINI_CCL_BACKENDS_MPI_OMPI_TYPE_MAP_H_
#define INFINI_CCL_BACKENDS_MPI_OMPI_TYPE_MAP_H_

#include <mpi.h>

#include "comm_impl.h"
#include "data_type_impl.h"

namespace infini::ccl {

static const ConstexprMap<DataType, MPI_Datatype, 12> kOmpiTypeMap{{{
    {DataType::kInt8, MPI_INT8_T},
    {DataType::kInt16, MPI_INT16_T},
    {DataType::kInt32, MPI_INT32_T},
    {DataType::kInt64, MPI_INT64_T},
    {DataType::kUInt8, MPI_UINT8_T},
    {DataType::kUInt16, MPI_UINT16_T},
    {DataType::kUInt32, MPI_UINT32_T},
    {DataType::kUInt64, MPI_UINT64_T},
    {DataType::kFloat32, MPI_FLOAT},
    {DataType::kFloat64, MPI_DOUBLE},
    {DataType::kFloat16, MPI_BYTE},
    {DataType::kBFloat16, MPI_BYTE},
}}};

static const ConstexprMap<ReductionOpType, MPI_Op, 5> kOmpiOpMap{{{
    {ReductionOpType::kSum, MPI_SUM},
    {ReductionOpType::kProd, MPI_PROD},
    {ReductionOpType::kMax, MPI_MAX},
    {ReductionOpType::kMin, MPI_MIN},
    // OpenMPI does not support native average reduction, so we map it to sum
    // and handle the scaling manually.
    {ReductionOpType::kAvg, MPI_SUM},
}}};

inline MPI_Datatype DataTypeToOmpiType(DataType dtype) {
  return kOmpiTypeMap.at(dtype);
}

inline MPI_Op RedOpToOmpiOp(ReductionOpType red_op) {
  return kOmpiOpMap.at(red_op);
}

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_MPI_OMPI_TYPE_MAP_H_
