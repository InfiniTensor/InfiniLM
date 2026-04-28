#ifndef __KUNLUN_COMMON_H__
#define __KUNLUN_COMMON_H__

#include "../../../utils.h"
#include <xpu/runtime.h>
#include <xpu/runtime_ex.h>
#include <xpu/xdnn.h>

namespace xdnn = baidu::xpu::api;

typedef XPUStream kunlunStream_t;
typedef XPUEvent kunlunEvent_t;
typedef xdnn::Context *xdnnHandle_t;

#define CHECK_KUNLUN(API) CHECK_INTERNAL(API, XPU_SUCCESS)

#endif
