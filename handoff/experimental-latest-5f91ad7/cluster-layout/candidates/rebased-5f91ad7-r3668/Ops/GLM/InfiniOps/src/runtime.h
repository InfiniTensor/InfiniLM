#ifndef INFINI_OPS_RUNTIME_H_
#define INFINI_OPS_RUNTIME_H_

#include <infini/rt.h>

namespace infini::ops {

template <Device::Type device_type>
using Runtime = infini::rt::runtime::Runtime<device_type>;

template <typename Derived>
using DeviceRuntime = infini::rt::runtime::DeviceRuntime<Derived>;

}  // namespace infini::ops

#endif
