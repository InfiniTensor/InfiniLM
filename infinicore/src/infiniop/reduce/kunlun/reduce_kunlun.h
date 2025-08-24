#ifndef __INFINIOP_REDUCE_KUNLUN_H__
#define __INFINIOP_REDUCE_KUNLUN_H__

#include "../../devices/kunlun/kunlun_kernel_common.h"

namespace op::common_kunlun::reduce_op {

using namespace device::kunlun::kernel;

// Use 16 floats instruction to calculate reduce
// data_ptr is the pointer of LM
static inline __device__ float sumSquaredF32(float *data_ptr, int count) {
    __local__ float acc_buf[16];
    int remain = count % 16;
    int offset_last = count - remain;
    int mask = lowerBitMask(remain - 1);
    // Load last 16 data
    float32x16_t v_last = vload_lm_float32x16_mz((data_ptr + offset_last), mask);
    // Do v_last * v_last
    v_last = vvmul_float32x16(v_last, v_last);
    // for every 16 float data
    for (int i = 0; i < offset_last; i += 16) {
        float32x16_t v_0 = vload_lm_float32x16_mz(data_ptr + i);
        // Do v_0 * v_0
        v_0 = vvmul_float32x16(v_0, v_0);
        // Add to v_last
        v_last = vvadd_float32x16(v_last, v_0);
    }
    vstore_lm_float32x16_mz(acc_buf, v_last);
    mfence();
    float res = 0.0f;
    for (int i = 0; i < 16; ++i) {
        res += acc_buf[i];
    }
    return res;
}

} // namespace op::common_kunlun::reduce_op

#endif
