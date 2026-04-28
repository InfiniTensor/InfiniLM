#ifndef __RECIPROCAL_MOORE_API_H__
#define __RECIPROCAL_MOORE_API_H__

// 1. 切换到 Moore 平台的 elementwise API 定义文件
#include "../../../elementwise/moore/elementwise_moore_api.h"

// 2. 调用宏生成 op::reciprocal::moore::Descriptor
// 宏展开后会包含 create 和 calculate 的标准声明
ELEMENTWISE_DESCRIPTOR(reciprocal, moore)

#endif // __RECIPROCAL_MOORE_API_H__
