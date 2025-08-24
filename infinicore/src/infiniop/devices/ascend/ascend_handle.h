#ifndef __INFINIOP_ASCEND_HANDLE_H__
#define __INFINIOP_ASCEND_HANDLE_H__

#include "../../handle.h"

namespace device::ascend {

class Handle : public InfiniopHandle {

    Handle(int device_id);

public:
    static infiniStatus_t create(InfiniopHandle **handle_ptr, int device_id);
};
} // namespace device::ascend

#endif
