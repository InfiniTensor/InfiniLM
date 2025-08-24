#ifndef __INFINIOP_CPU_HANDLE_H__
#define __INFINIOP_CPU_HANDLE_H__

#include "../../handle.h"

namespace device::cpu {

class Handle : public InfiniopHandle {
    Handle();

public:
    static infiniStatus_t create(InfiniopHandle **handle_ptr, int);
};

} // namespace device::cpu

#endif
