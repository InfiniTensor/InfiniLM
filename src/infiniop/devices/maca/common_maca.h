#include "../../../utils.h"
#include "../pool.h"
#include "maca_handle.h"
#include <hcblas/hcblas.h>
#include <hcdnn/hcdnn.h>
#include <memory>

#define CHECK_MCBLAS(API) CHECK_INTERNAL(API, HCBLAS_STATUS_SUCCESS)
#define CHECK_MCDNN(API) CHECK_INTERNAL(API, HCDNN_STATUS_SUCCESS)

namespace device::maca {

class Handle::Internal {
    Pool<hcblasHandle_t> mcblas_handles;
    Pool<hcdnnHandle_t> mcdnn_handles;

    template <typename T>
    using Fn = std::function<infiniStatus_t(T)>;

    int _warp_size,
        _max_threads_per_block,
        _block_size[3],
        _grid_size[3];

public:
    Internal(int);
    infiniStatus_t useMcblas(hcStream_t stream, const Fn<hcblasHandle_t> &f) const;
    infiniStatus_t useMcdnn(hcStream_t stream, const Fn<hcdnnHandle_t> &f) const;

    int warpSize() const;
    int maxThreadsPerBlock() const;
    int blockSizeX() const;
    int blockSizeY() const;
    int blockSizeZ() const;
    int gridSizeX() const;
    int gridSizeY() const;
    int gridSizeZ() const;
};

hcdnnDataType_t getHcdnnDtype(infiniDtype_t dt);

} // namespace device::maca
