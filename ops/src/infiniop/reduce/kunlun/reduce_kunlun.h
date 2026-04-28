#ifndef __INFINIOP_REDUCE_KUNLUN_H__
#define __INFINIOP_REDUCE_KUNLUN_H__

#include "../../devices/kunlun/kunlun_kernel_common.h"

namespace op::common_kunlun::reduce_op {

using namespace device::kunlun::kernel;

// Sum(x^2) on contiguous data of length count
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__device__ inline Tcompute sumSquared(__shared_ptr__ const Tdata *data_ptr, size_t count) {
    Tcompute ss = 0;

    for (size_t i = core_id(); i < count; i += BLOCK_SIZE) {
        Tdata xi = data_ptr[i];
        ss += Tcompute(xi) * Tcompute(xi);
    }

    __shared__ Tcompute temp_storage;
    if (core_id() == 0) {
        temp_storage = Tcompute(0.f);
    }
    sync_cluster();

    atomicAdd(&temp_storage, ss);
    sync_cluster();

    return temp_storage;
}

// Sum(x) on contiguous data of length count
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__device__ inline Tcompute sum(__shared_ptr__ const Tdata *data_ptr, size_t count) {
    Tcompute ss = 0;

    for (size_t i = core_id(); i < count; i += BLOCK_SIZE) {
        Tdata xi = data_ptr[i];
        ss += Tcompute(xi);
    }

    __shared__ Tcompute temp_storage;
    if (core_id() == 0) {
        temp_storage = Tcompute(0.f);
    }
    sync_cluster();

    atomicAdd(&temp_storage, ss);
    sync_cluster();

    return temp_storage;
}

// Max(x) on contiguous data of length count
template <unsigned int BLOCK_SIZE, typename Tdata>
__device__ inline Tdata max(__shared_ptr__ const Tdata *data_ptr, size_t count) {
    Tdata max_val = data_ptr[0];

    for (size_t i = core_id(); i < count; i += BLOCK_SIZE) {
        Tdata xi = data_ptr[i];
        max_val = fmax(max_val, Tdata(xi));
    }

    __shared__ Tdata temp_storage;
    if (core_id() == 0) {
        temp_storage = data_ptr[0];
    }
    sync_cluster();

    atomicMax(&temp_storage, max_val);
    sync_cluster();

    return temp_storage;
}

} // namespace op::common_kunlun::reduce_op

#endif
