#ifndef __ATANH_MOORE_API_H__
#define __ATANH_MOORE_API_H__

// 1. 修改包含路径，指向 moore 平台的 elementwise API 定义
#include "../../../elementwise/moore/elementwise_moore_api.h"

// 2. 使用 ELEMENTWISE_DESCRIPTOR 宏，平台参数改为 moore
// 这将自动生成 op::atanh::moore::Descriptor 类的声明
ELEMENTWISE_DESCRIPTOR(atanh, moore)

#endif // __ATANH_MOORE_API_H__
