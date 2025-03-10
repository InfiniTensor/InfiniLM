#ifndef __INFINIOP_BANG_INTERNAL_H__
#define __INFINIOP_BANG_INTERNAL_H__

#include "../../../utils.h"
#include "../pool.h"
#include "bang_handle.h"
#include "cnnl.h"
#include "cnrt.h"
#include <functional>

namespace device::bang {

class Handle::Internal {
    Pool<cnnlHandle_t> cnnl_handles;

    template <typename T>
    using Fn = std::function<infiniStatus_t(T)>;

public:
    infiniStatus_t useCnnl(cnrtQueue_t queue, const Fn<cnnlHandle_t> &f) const;
};

cnnlDataType_t getCnnlDtype(infiniDtype_t dt);

} // namespace device::bang

#endif // __INFINIOP_BANG_INTERNAL_H__
