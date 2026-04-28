#ifndef __RMS_NORM_KUNLUN_KERNEL_H__
#define __RMS_NORM_KUNLUN_KERNEL_H__

#include "../../../devices/kunlun/kunlun_kernel_common.h"
#include "../../../reduce/kunlun/reduce_kunlun.h"

using namespace device::kunlun::kernel;

template <unsigned int BLOCK_SIZE, typename Tcompute, typename Tdata, typename Tweight>
__device__ void rmsnormBlock(
    __shared_ptr__ Tdata *y,
    __shared_ptr__ const Tdata *x,
    __shared_ptr__ const Tweight *w,
    size_t dim,
    float epsilon) {

    // Block reduce sum of x^2
    Tcompute ss = op::common_kunlun::reduce_op::sumSquared<BLOCK_SIZE, Tdata, Tcompute>(x, dim);

    __shared__ Tcompute rms;
    if (core_id() == 0) {
        rms = Tcompute(rsqrt(ss / Tcompute(dim) + epsilon));
    }
    sync_cluster();

    // Copy contiguous x, w into local mem (load from shared memory safely)
    for (size_t i = core_id(); i < dim; i += BLOCK_SIZE) {
        Tdata xi = x[i];
        Tweight wi = w[i];
        y[i] = Tdata(Tcompute(xi) * Tcompute(wi) * rms);
    }
    sync_cluster();
}

#endif
