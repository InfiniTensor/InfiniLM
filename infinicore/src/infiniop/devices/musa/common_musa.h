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

    int _warp_size,
        _max_threads_per_block,
        _block_size[3],
        _grid_size[3];

    template <typename T>
    using Fn = std::function<infiniStatus_t(T)>;

public:
    Internal(int);
    infiniStatus_t useMublas(musaStream_t stream, const Fn<mublasHandle_t> &f) const;
    infiniStatus_t useMudnn(musaStream_t stream, const Fn<::musa::dnn::Handle &> &f) const;

    int warpSize() const;
    int maxThreadsPerBlock() const;
    int blockSizeX() const;
    int blockSizeY() const;
    int blockSizeZ() const;
    int gridSizeX() const;
    int gridSizeY() const;
    int gridSizeZ() const;
};

} // namespace device::musa
