#ifndef __COMMON_BANG_H__
#define __COMMON_BANG_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include "../pool.h"
#include "bang_handle.h"
#include "cnnl.h"
#include "cnrt.h"
#include <functional>

#define CHECK_BANG(API) CHECK_INTERNAL(API, CNNL_STATUS_SUCCESS)

#define NRAM_MAX_SIZE 1024 * 240
constexpr size_t ALIGN_SIZE = 128;

namespace device::bang {

class Handle::Internal {
    Pool<cnnlHandle_t> cnnl_handles;

    int _core_per_cluster;
    int _cluster_count;

    template <typename T>
    using Fn = std::function<infiniStatus_t(T)>;

public:
    Internal(int);

    infiniStatus_t useCnnl(cnrtQueue_t queue, const Fn<cnnlHandle_t> &f) const;

    int getCorePerCluster() const;
    int getClusterCount() const;
};

cnnlDataType_t getCnnlDtype(infiniDtype_t dt);

// set cnnl tensor descriptor without strides
infiniStatus_t setCnnlTensor(cnnlTensorDescriptor_t desc,
                             const InfiniopTensorDescriptor *layout);

// set cnnl tensor descriptor with strides
infiniStatus_t setCnnlTensorEx(cnnlTensorDescriptor_t desc,
                               const InfiniopTensorDescriptor *layout);

} // namespace device::bang

#endif // __COMMON_BANG_H__
