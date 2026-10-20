#ifndef INFINI_OPS_CAMBRICON_DENSE_TRIANGLE_LAUNCH_H_
#define INFINI_OPS_CAMBRICON_DENSE_TRIANGLE_LAUNCH_H_
#include <cnrt.h>
#include <cstddef>
#include <cstdint>
namespace infini::ops {
void DenseTriangleLaunch(cnrtQueue_t,const void*,void*,std::size_t,std::size_t,std::size_t,int64_t,bool,std::size_t);
}
#endif
