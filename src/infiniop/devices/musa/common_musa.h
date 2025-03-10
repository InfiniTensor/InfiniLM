#include "../../../utils.h"
#include "musa_handle.h"
#include "pool.h"
#include <memory>
#include <mublas.h>
#include <mudnn.h>
#include <musa.h>
#include <musa_runtime_api.h>

#define CHECK_MUBLAS(API) CHECK_INTERNAL(API, MUBLAS_STATUS_SUCCESS)
#define CHECK_MUDNN(API) CHECK_INTERNAL_MUDNN(API, ::musa::dnn::Status::SUCCESS, return INFINI_STATUS_INTERNAL_ERROR)

#define CHECK_INTERNAL_MUDNN(API, EXPECT, ACTION)                                    \
    do {                                                                             \
        auto api_result_ = (API);                                                    \
        if (api_result_ != (EXPECT)) {                                               \
            std::cerr << "Error Code " << (int)api_result_ << " in `" << #API << "`" \
                      << " from " << __func__                                        \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
            { ACTION; }                                                              \
        }                                                                            \
    } while (0)

namespace device::musa {

class Handle::Internal {
    Pool<mublasHandle_t> mublas_handles;
    Pool<::musa::dnn::Handle> mudnn_handles;

    template <typename T>
    using Fn = std::function<infiniStatus_t(T)>;

public:
    infiniStatus_t useMublas(MUstream stream, const Fn<mublasHandle_t> &f) const;
    infiniStatus_t useMudnn(musaStream_t stream, const Fn<::musa::dnn::Handle &> &f) const;
};

} // namespace device::musa
