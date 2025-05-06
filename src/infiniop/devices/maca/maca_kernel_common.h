#ifdef ENABLE_SUGON_MACA_API
#define INFINIOP_MACA_KERNEL __launch_bounds__(512) __global__ void
#else
#define INFINIOP_MACA_KERNEL __global__ void
#endif

// Posible maximum number of threads per block for MACA architectures
// Used for picking correct kernel launch configuration
#define MACA_BLOCK_SIZE_1024 1024
#define MACA_BLOCK_SIZE_512 512

#define CHECK_MACA(API) CHECK_INTERNAL(API, hcSuccess)

namespace device::maca {

// return the memory offset of original tensor, given the flattened index of broadcasted tensor
__forceinline__ __device__ __host__ size_t
indexToReducedOffset(
    size_t flat_index,
    size_t ndim,
    const ptrdiff_t *broadcasted_strides,
    const ptrdiff_t *target_strides) {
    size_t res = 0;
    for (size_t i = 0; i < ndim; ++i) {
        res += flat_index / broadcasted_strides[i] * target_strides[i];
        flat_index %= broadcasted_strides[i];
    }
    return res;
}

// get the memory offset of the given element in a tensor given its flat index
__forceinline__ __device__ __host__ size_t
indexToOffset(
    size_t flat_index,
    size_t ndim,
    const size_t *shape,
    const ptrdiff_t *strides) {
    size_t res = 0;
    for (size_t i = ndim; i-- > 0;) {
        res += (flat_index % shape[i]) * strides[i];
        flat_index /= shape[i];
    }
    return res;
}
} // namespace device::maca

#ifdef ENABLE_MACA_API
#include <maca_fp16.h>
__forceinline__ __device__ float
exp_(const float val) {
    return expf(val);
}

__forceinline__ __device__ long double
exp_(const long double val) {
    return expl(val);
}

__forceinline__ __device__ double
exp_(const double val) {
    return exp(val);
}

__forceinline__ __device__ __half
exp_(const __half x) {
    return hexp(x);
}
#endif
