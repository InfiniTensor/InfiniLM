#ifndef __KUNLUN_XBLAS_H__
#define __KUNLUN_XBLAS_H__

#include "../../handle.h"
#include "../pool.h"
#include "kunlun_common.h"
#include <cublas_v2.h>
#include <memory>

#define CHECK_CUBLAS(API) CHECK_INTERNAL(API, CUBLAS_STATUS_SUCCESS)

namespace device::kunlun::blas {

struct Handle : public InfiniopHandle {
    class Internal;
    auto internal() const -> const std::shared_ptr<Internal> &;

    Handle(int device_id);

private:
    std::shared_ptr<Internal> _internal;

public:
    static infiniStatus_t create(InfiniopHandle **handle_ptr, int device_id);
};

class Handle::Internal {
    Pool<cublasHandle_t> blas_handles;
    template <typename T>
    using Fn = std::function<infiniStatus_t(T)>;

public:
    infiniStatus_t useCublas(cudaStream_t stream, const Fn<cublasHandle_t> &f) const;
};

} // namespace device::kunlun::blas

#endif // __KUNLUN_XBLAS_H__
