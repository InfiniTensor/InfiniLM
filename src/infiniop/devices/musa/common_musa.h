#include "../../../utils.h"
#include "../pool.h"
#include "musa_handle.h"
#include <mublas.h>
#include <mudnn.h>
#include <musa.h>
#include <musa_fp16_mtgpu.h>
#include <musa_runtime_api.h>

#define CHECK_MUBLAS(API) CHECK_INTERNAL(API, MUBLAS_STATUS_SUCCESS)
#define CHECK_MUDNN(API) CHECK_INTERNAL((int)API, (int)::musa::dnn::Status::SUCCESS)

namespace device::musa {

class Handle::Internal {
    Pool<std::unique_ptr<mublasHandle_t>> mublas_handles;
    Pool<std::unique_ptr<::musa::dnn::Handle>> mudnn_handles;

    template <typename T>
    using Fn = std::function<infiniStatus_t(T)>;

public:
    infiniStatus_t useMublas(musaStream_t stream, const Fn<mublasHandle_t> &f) const;
    infiniStatus_t useMudnn(musaStream_t stream, const Fn<::musa::dnn::Handle &> &f) const;
};

} // namespace device::musa
